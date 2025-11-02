import os
import itertools
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import AutoModel, AutoTokenizer

from model.attention import TransformerDecoder, TransformerDecoderLayer
from model.fusion_block import GAFMBlock
from model.layers import Mlp, SmallUpsampleNet, FeatureUpsampler
from model.gaf_conv import GafConv
from renderer.gaussian_render import Gaussian_Renderer
from renderer.render_utils import depth_to_rgb

# Disable tokenizer multi-thread warning from HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Branch2D(nn.Module):
    """
    The 2D Affordance Branch of GEAL model.
    This branch takes point clouds (xyz), text queries, and optionally 3D features,
    then renders 2D projections using differentiable Gaussian rendering, extracts
    DINO features, fuses them with language embeddings, and predicts pixel-wise
    affordance activation maps.

    Args:
        cfg (dict): Model configuration dictionary.
        render_cfg (dict): Renderer configuration dictionary.
    """

    def __init__(self, cfg, render_cfg):
        super().__init__()

        # ====== Core model configuration ======
        self.n_groups = cfg["n_groups"]
        self.emb_dim = cfg["emb_dim"]
        self.num_heads = cfg["num_heads"]
        self.freeze_text_encoder = cfg["freeze_text_encoder"]
        self.text_encoder_type = cfg["text_encoder_type"]

        self.dino_dim = cfg["dino_dim"]
        self.llm_dim = cfg["llm_dim"]
        self.num_levels = cfg["level"]

        self.project_dim = cfg.get("project_dim", 64)
        self.stage1 = cfg.get("stage1", False)

        self.normalize_mean = cfg.get("normalize_mean", [0.485, 0.456, 0.406])
        self.normalize_std = cfg.get("normalize_std", [0.229, 0.224, 0.225])
        self.render_resolution = render_cfg["render_resolution"]
        self.fuse_level = cfg.get("fuse_level", False)

        # ====== Text encoder (frozen or fine-tuned) ======
        self.text_encoder = AutoModel.from_pretrained(self.text_encoder_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_type)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.emb_dim, bias=True),
            nn.LayerNorm(self.emb_dim, eps=1e-12)
        )

        # ====== Differentiable Gaussian Renderer ======
        self.renderer = Gaussian_Renderer(
            sh_degree=render_cfg["sh_degree"],
            render_resolution=self.render_resolution
        )

        # ====== DINOv2 Backbone (frozen) ======
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
        for p in self.dino_model.parameters():
            p.requires_grad = False

        # ====== Cross-modal Group Perception Block ======
        self.GAFM_block = GAFMBlock(
            embed_dims=self.llm_dim,
            num_group_token=self.n_groups,
            lan_dim=self.emb_dim
        )

        # ====== Multi-Level DINO Feature Processing ======
        self.dino_linears = nn.ModuleList([nn.Linear(self.dino_dim, self.dino_dim) for _ in range(self.num_levels)])
        self.dino_norms = nn.ModuleList([nn.LayerNorm(self.dino_dim) for _ in range(self.num_levels)])

        if self.fuse_level:
            self.gaf_conv = GafConv(M=self.num_levels, d=self.num_levels * self.dino_dim, K=self.num_levels)

        # ====== Embedding projection heads ======
        self.dino_embed = Mlp(
            in_features=self.dino_dim,
            hidden_features=self.llm_dim,
            out_features=self.llm_dim,
            act_layer=nn.GELU,
            drop=0.
        )
        self.dino_embed_norm = nn.LayerNorm(self.llm_dim)

        self.cls_embed = Mlp(
            in_features=self.dino_dim,
            hidden_features=self.llm_dim,
            out_features=self.llm_dim,
            act_layer=nn.GELU,
            drop=0.
        )
        self.cls_norm = nn.LayerNorm(self.llm_dim)

        # ====== Decoder and positional encoding ======
        decoder_layer = TransformerDecoderLayer(self.llm_dim, nheads=self.num_heads, dropout=0)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=1, norm=nn.LayerNorm(self.llm_dim))
        self.pos1d = nn.Parameter(torch.zeros(1, self.n_groups, self.llm_dim))

        # ====== Upsampling modules ======
        self.learnable_upsample = SmallUpsampleNet(
            self.llm_dim, [self.render_resolution, self.render_resolution]
        )
        if not self.stage1:
            self.feature_upsampler = FeatureUpsampler(self.llm_dim)

    # ----------------------------------------------------------------------

    def forward(self, text, xyz, features_3d=None):
        """
        Forward pass for the 2D branch.
        Args:
            text (list[str]): Batch of text queries.
            xyz (Tensor): Point cloud tensor of shape [B, N, 3].
            features_3d (Tensor): Optional 3D features from the 3D branch.

        Returns:
            Tensor: 2D affordance maps of shape [B*n_view, 1, H, W].
        """
        B = xyz.shape[0]

        # ========== Step 1. Differentiable Rendering ==========
        rendered_images, masks, render_feats = self._render_views(xyz, features_3d)
        Bn, C, H, W = rendered_images.shape

        # ========== Step 2. Extract DINOv2 Features ==========
        dino_layers = self.dino_model.get_intermediate_layers(rendered_images, n=self.num_levels, return_class_token=True)

        # Process multi-level features with layernorms and linear projections
        dense_feats = []
        for i, (patch_feat, _) in enumerate(dino_layers):
            feat_proj = self.dino_linears[i](patch_feat)
            fused_feat = self.dino_norms[i](feat_proj)
            dense_feats.append(fused_feat)

        # Combine multi-level DINO features using Conv gating
        if self.fuse_level:
            concat_feats = torch.cat(dense_feats, dim=-1).transpose(1, 2)
            gates, _ = self.gaf_conv(concat_feats)
            fused_feat = sum(gates[:, i].view(-1, 1, 1) * dense_feats[i] for i in range(len(dense_feats)))
        
        fused_feat = self.dino_embed_norm(self.dino_embed(fused_feat))

        # CLS token projection
        cls_token = dino_layers[-1][1]
        cls_token = self.cls_norm(self.cls_embed(cls_token))
        cls_token = cls_token.view(Bn, -1, 1, 1).expand(Bn, -1, H // 14, W // 14)

        # ========== Step 3. Text Encoding ==========
        # Flatten text batch (nested list) into a sequence
        text_tuples = tuple(zip(*text))
        flat_queries = list(itertools.chain.from_iterable(text_tuples))
        text_embeds, text_mask = self._encode_text(flat_queries, xyz.device)

        # ========== Step 4. Cross-modal Fusion ==========
        cross_modal_feat = self.GAFM_block(text_embeds, fused_feat)

        # ========== Step 5. Affordance Prediction ==========
        if self.stage1:
            # Stage 1: Predict 2D activation maps
            text_feat = self.decoder(text_embeds, cross_modal_feat,
                                     tgt_key_padding_mask=text_mask,
                                     query_pos=self.pos1d)
            text_feat *= text_mask.unsqueeze(-1).float()

            # Attention map generation
            attn = torch.einsum('blc,bcn->bln', text_feat, fused_feat.transpose(1, 2))
            attn = attn.sum(1) / text_mask.float().sum(1).unsqueeze(-1)
            attn_map = attn.reshape(Bn, -1, H // 14, W // 14)
            attn_map = self.learnable_upsample(torch.cat([attn_map, cls_token], dim=1))
            attn_map = torch.sigmoid(attn_map)
            return attn_map

        else:
            # Stage 2: Cross-modal feature fusion (for 3D consistency)
            dense_feat_map = cross_modal_feat.transpose(1, 2).reshape(Bn, -1, H // 14, W // 14)
            fused_features = self.feature_upsampler(dense_feat_map)
            return fused_features, render_feats

    # ----------------------------------------------------------------------

    def _render_views(self, xyz, features_3d):
        """
        Render point clouds into 2D multi-view images using Gaussian splatting.
        """
        rendered_images, masks, feats = [], [], []

        # If no 3D features provided, create iterable of Nones
        if features_3d is None:
            features_3d = [None] * xyz.shape[0]

        for pts, f3d in zip(xyz, features_3d):
            rgb_img, depth_img, _, _, feat = self.renderer(pts, None, f3d)
            depth_vis = depth_to_rgb(depth_img)
            mask = (rgb_img != 0).all(dim=1, keepdim=True).int()
            norm_img = TF.normalize(depth_vis, mean=self.normalize_mean, std=self.normalize_std)

            rendered_images.append(norm_img)
            masks.append(mask)
            feats.append(feat)

        render_tensor = torch.stack(rendered_images)       # [B, n_views, 3, H, W]
        mask_tensor = torch.stack(masks)
        feat_tensor = torch.stack(feats)

        B, V, C, H, W = render_tensor.shape
        render_tensor = render_tensor.view(-1, C, H, W)
        mask_tensor = mask_tensor.view(-1, 1, H, W)
        feat_tensor = feat_tensor.view(-1, self.project_dim, H, W)

        return render_tensor, mask_tensor, feat_tensor

    # ----------------------------------------------------------------------

    def _encode_text(self, queries, device):
        """
        Encode a list of natural language queries using pretrained text encoder.
        """
        tokens = self.tokenizer.batch_encode_plus(
            queries,
            padding='max_length',
            truncation=True,
            max_length=self.n_groups,
            return_tensors='pt'
        ).to(device)

        # Freeze text encoder if configured
        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded = self.text_encoder(**tokens).last_hidden_state

        projected = self.text_proj(encoded)
        attn_mask = tokens.attention_mask.bool()
        return projected, attn_mask
