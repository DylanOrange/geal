import os 
import torch
import torch.nn as nn

from model.pointnet2_utils import PointNetFeaturePropagation
from model.attention import TransformerDecoder, TransformerDecoderLayer
from model.fusion_block import GAFMBlock
from model.layers import PointFeatureDownsampler,PointEncoder
from model.gaf_conv import GafConv
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Branch3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        """
        The 3D Affordance Branch of GEAL.
        This branch encodes 3D point clouds, fuses them with text queries using
        cross-modal attention, and predicts dense affordance activation scores
        for each 3D point.

        Args:
            cfg (dict): Model configuration dictionary containing keys like:
                - n_groups (int): number of group tokens for text
                - emb_dim (int): embedding dimension
                - N_p (int): number of input points
                - num_heads (int): transformer attention heads
                - normal_channel (bool): whether normals are used
                - freeze_text_encoder (bool): freeze or finetune text encoder
                - text_encoder_type (str): e.g., "bert-base-uncased"
                - project_dim (int): projection dimension for point features
                - training (bool): whether training or inference mode
                - level (int): number of point feature levels
                - fuse_level (bool): enable multi-level fusion
        """

        # ====== Core configuration ======
        self.n_groups = cfg["n_groups"]
        self.emb_dim = cfg["emb_dim"]
        self.N_p = cfg["N_p"]
        self.num_heads = cfg["num_heads"]
        self.normal_channel = cfg["normal_channel"]
        self.freeze_text_encoder = cfg["freeze_text_encoder"]
        self.text_encoder_type = cfg["text_encoder_type"]
        self.project_dim = cfg["project_dim"]
        self.training = cfg["training"]
        self.num_levels = cfg["level"]
        self.fuse_level = cfg.get("fuse_level", False)

        # ====== Text encoder (frozen or finetuned) ======
        self.text_encoder = AutoModel.from_pretrained(self.text_encoder_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_type)
        self.text_resizer = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.emb_dim, bias=True),
            nn.LayerNorm(self.emb_dim, eps=1e-12)
        )

        # ====== Point encoder (hierarchical) ======
        # Input may include normals as additional channels
        self.additional_channel = 3 if self.normal_channel else 0
        point_cfg = cfg.get("point_encoder", {})
        self.point_encoder = PointEncoder(
            emb_dim=self.emb_dim,
            normal_channel=self.normal_channel,
            additional_channel=self.additional_channel,
            N_p=self.N_p,
            cfg_layers=point_cfg.get("layers", None),
        )

        # ====== Positional encoding & cross-modal fusion ======
        self.pos1d = nn.Parameter(torch.zeros(1, self.n_groups, self.emb_dim))
        nn.init.trunc_normal_(self.pos1d, std=0.2)

        # Fusion Module for text ↔ point alignment
        self.GAFM_block = GAFMBlock(
            embed_dims=self.emb_dim,
            num_group_token=self.n_groups,
            lan_dim=self.emb_dim
        )

        # ====== Transformer decoder for cross-modal interaction ======
        decoder_layer = TransformerDecoderLayer(self.emb_dim, nheads=self.num_heads, dropout=0)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=1, norm=nn.LayerNorm(self.emb_dim))

        # ====== Feature propagation (hierarchical upsampling) ======
        fp_cfg = cfg.get("fp_layers", {})
        self.fp_layers = nn.ModuleDict()
        for name in ["fp3", "fp2", "fp1"]:
            layer_cfg = fp_cfg.get(name, {})
            in_ch = layer_cfg.get("in_channel", 512)
            if layer_cfg.get("emb_add", False):
                in_ch += self.emb_dim
            if layer_cfg.get("add_normals", False):
                in_ch += self.additional_channel
            mlp = layer_cfg.get("mlp", [512, 512])
            self.fp_layers[name] = PointNetFeaturePropagation(in_channel=in_ch, mlp=mlp)

        # ====== multi-level fusion ======
        if self.fuse_level:
            self.fp_upsample_2 = PointNetFeaturePropagation(in_channel=self.emb_dim * 2, mlp=[512, 512])
            self.fp_upsample_3 = PointNetFeaturePropagation(in_channel=self.emb_dim * 2, mlp=[512, 512])
            self.gaf_conv = GafConv(M=self.num_levels, d=self.num_levels * self.emb_dim, K=self.num_levels)

        # ====== Feature projection head for downstream consistency ======
        if self.training:
            self.feature_downsampler = PointFeatureDownsampler(self.emb_dim, self.project_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)

    # ----------------------------------------------------------------------

    def forward(self, text, xyz):
        """
        Forward pass of the 3D branch.

        Args:
            text (list[str]): Text queries (batch of affordance sentences).
            xyz (Tensor): Input point cloud [B, N, 3].

        Returns:
            - During training: (affordance_pred, downsampled_feats)
            - During inference: affordance_pred
        """
        # ========== Step 1. Normalize input coordinates ==========
        xyz = xyz / 0.5

        # ========== Step 2. Text encoding ==========
        # Extract affordance description portion (after ".")
        text_queries = [sentence.split('.')[1] for sentence in text[0]]
        text_feat, text_mask = self.forward_text(text_queries, xyz.device)

        # ========== Step 3. Point feature encoding ==========
        multi_scale_feats = self.point_encoder(xyz)
        feat_lvl0, feat_lvl1, feat_lvl2, feat_lvl3 = multi_scale_feats

        # ========== Step 4. Cross-modal feature fusion ==========
        # Inject language context into the deepest point features
        feat_lvl3[1] = self.GAFM_block(text_feat, feat_lvl3[1].transpose(-2, -1)).transpose(-2, -1)

        # Hierarchical feature propagation (decoder path)
        feat_up3 = self.fp_layers["fp3"](feat_lvl2[0], feat_lvl3[0], feat_lvl2[1], feat_lvl3[1])  # [B, 512, 128]
        feat_up2 = self.GAFM_block(text_feat, feat_up3.transpose(-2, -1)).transpose(-2, -1)
        feat_up2 = self.fp_layers["fp2"](feat_lvl1[0], feat_lvl2[0], feat_lvl1[1], feat_up2)       # [B, 512, 512]
        feat_up1 = self.GAFM_block(text_feat, feat_up2.transpose(-2, -1)).transpose(-2, -1)

        fused_feat = self.fp_layers["fp1"](
            feat_lvl0[0], feat_lvl1[0],
            torch.cat([feat_lvl0[0], feat_lvl0[1]], dim=1),
            feat_up1
        )  # [B, 512, 2048]

        # ========== Step 5. Multi-level feature fusion ==========
        if self.fuse_level:
            feat_up2_refined = self.fp_upsample_2(feat_lvl0[0], feat_lvl1[0], fused_feat, feat_up2)
            feat_up3_refined = self.fp_upsample_3(feat_lvl0[0], feat_lvl2[0], fused_feat, feat_up3)
            dense_feats = [fused_feat, feat_up2_refined, feat_up3_refined]
            concat_feats = torch.cat(dense_feats, dim=1)
            gates, _ = self.gaf_conv(concat_feats)
            fused_feat = sum(gates[:, i].view(-1, 1, 1) * dense_feats[i] for i in range(len(dense_feats)))

        # ========== Step 6. Transformer-based decoding ==========
        text_decoded = self.decoder(
            text_feat, fused_feat.transpose(-2, -1),
            tgt_key_padding_mask=text_mask,
            query_pos=self.pos1d
        )
        text_decoded *= text_mask.unsqueeze(-1).float()

        # Compute 3D affordance activation per point
        affordance_map = torch.einsum('blc,bcn->bln', text_decoded, fused_feat)
        affordance_map = affordance_map.sum(1) / text_mask.float().sum(1).unsqueeze(-1)
        affordance_map = torch.sigmoid(affordance_map)

        # ========== Step 7. Optional downsampling for stage-2 fusion ==========
        if self.training:
            downsampled_feat = self.feature_downsampler(fused_feat)  # [B, 512, 2048] → [B, 64, 2048]
            downsampled_feat = downsampled_feat.transpose(1, 2)      # [B, 2048, 64]
            return affordance_map, downsampled_feat
        else:
            return affordance_map
        
    # ------------------------------------------------------------------
    def forward_text(self, text_queries, device):
        """
        Encode natural language affordance descriptions.

        """
        tokens = self.tokenizer.batch_encode_plus(
            text_queries,
            padding='max_length',
            truncation=True,
            max_length=self.n_groups,
            return_tensors='pt'
        ).to(device)

        with torch.inference_mode(mode=self.freeze_text_encoder):
            encoded = self.text_encoder(**tokens).last_hidden_state

        projected = self.text_resizer(encoded)
        attn_mask = tokens.attention_mask.bool()
        return projected, attn_mask