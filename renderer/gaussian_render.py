import torch
import numpy as np
from renderer.gaussian_model import Renderer, MiniCam, BasicPointCloud_torch
from renderer.cam_utils import orbit_camera, OrbitCamera
from torchvision import utils as vutils
import open3d as o3d

def save_image(input_tensor, filename):
    """
    Save a rendered tensor image to disk.

    Args:
        input_tensor (torch.Tensor): a 4D tensor with shape [1, C, H, W].
        filename (str): output image file path.

    Notes:
        The tensor is detached and moved to CPU before saving.
    """
    assert len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1, \
        "Expected input tensor shape [1, C, H, W]"
    input_tensor = input_tensor.clone().detach().to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)

class Gaussian_Renderer:

    """
    Differentiable Gaussian Splatting Renderer for point clouds.

    This renderer projects 3D points into 2D multi-view images using a
    differentiable rasterization process. It supports rendering RGB, depth,
    and language-feature maps under multiple predefined viewpoints.

    Args:
        sh_degree (int): spherical harmonics degree for shading (default: 3)
        render_resolution (int): output image size (square)
        device (torch.device): computation device (CUDA or CPU)

    Usage:
        >>> renderer = Gaussian_Renderer(sh_degree=3, render_resolution=112, device='cuda')
        >>> rgb_imgs, depth_imgs, _, _, feats = renderer(points, gt_mask, lang_feat)
    """
        
    def __init__(self, sh_degree=3, render_resolution=112, device=None):
        self.device = device
        self.render = Renderer(sh_degree=sh_degree)

        # Define 12 view angles: 8 horizontal + 4 vertical
        self.hor_angles = [-180, -135, -90, -45, 0, 45, 90, 135]
        self.ver_angles = [-89, -45, 45, 89]

        self.render_resolution = render_resolution
        self.base_cam = OrbitCamera(800, 800, r=2, fovy=49.1)  # base FOV reference camera
        self.depth_image = None
        self.back_project_point = None
    
    def __call__(self, point_set, gt_mask=None, language_feature=None):
        """
        Render multi-view images given a 3D point cloud.

        Args:
            point_set (torch.Tensor): shape [3, N] or [N, 3], point coordinates.
            gt_mask (torch.Tensor or None): ground truth affordance mask for coloring (optional).
            language_feature (torch.Tensor or None): per-point feature vectors (optional).

        Returns:
            Tuple of:
                - gt_image (torch.Tensor): [12, 3, H, W] rendered RGB images
                - depth_image (torch.Tensor): [12, 1, H, W] rendered depth maps
                - render_idx (torch.Tensor): [12, H, W] rendered point indices
                - rendered_contrib (torch.Tensor): [12, H, W] pixel contributions
                - features (torch.Tensor): [12, C, H, W] rendered feature maps
        """
        # Ensure tensor shape is [N, 3]
        if point_set.shape[0] == 3:
            point_set = point_set.transpose(1, 0)
        num_pts = point_set.shape[0]

        # Assign color (affordance mask → white, otherwise position-based color)
        if gt_mask is not None:
            color = gt_mask.reshape(-1, 1).repeat(1, 3)
        else:
            color = point_set + 0.5  # normalized coordinates as color

        # Initialize placeholder normals and language features
        normals = torch.zeros((num_pts, 3), device=point_set.device)
        if language_feature is None:
            language_feature = torch.zeros((num_pts, 64), device=point_set.device)

        # Create differentiable point cloud object
        pcd = BasicPointCloud_torch(
            points=point_set,
            colors=color,
            normals=normals,
            language_feature=language_feature
        )
        self.render.initialize(pcd)

        # Prepare containers for multi-view outputs
        gt_image_list, depth_image_list = [], []
        render_idx_list, rendered_contrib_list, feature_list = [], [], []

        # ----------------- 8 Horizontal Views -----------------
        for hor in self.hor_angles:
            # For 224px resolution use distance=1.5; for 112px use 1.6
            pose = orbit_camera(0, hor, 1.6)
            cur_cam = MiniCam(
                pose, self.render_resolution, self.render_resolution,
                self.base_cam.fovy, self.base_cam.fovx, self.base_cam.near, self.base_cam.far
            )
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=point_set.device)
            out = self.render.render(cur_cam, bg_color=bg_color)

            gt_image_list.append(out["image"])
            depth_image_list.append(out["depth"])
            render_idx_list.append(out["rendered_idx"])
            rendered_contrib_list.append(out["rendered_contrib"])
            feature_list.append(out["language_feature_image"])

        # ----------------- 4 Vertical Views -----------------
        for ver in self.ver_angles:
            pose = orbit_camera(ver, 0, 1.6)
            cur_cam = MiniCam(
                pose, self.render_resolution, self.render_resolution,
                self.base_cam.fovy, self.base_cam.fovx, self.base_cam.near, self.base_cam.far
            )
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=point_set.device)
            out = self.render.render(cur_cam, bg_color=bg_color)

            gt_image_list.append(out["image"])
            depth_image_list.append(out["depth"])
            render_idx_list.append(out["rendered_idx"])
            rendered_contrib_list.append(out["rendered_contrib"])
            feature_list.append(out["language_feature_image"])

        # Stack all 12 view results
        gt_image = torch.stack(gt_image_list, dim=0)
        depth_image = torch.stack(depth_image_list, dim=0)
        render_idx = torch.stack(render_idx_list, dim=0)
        rendered_contrib = torch.stack(rendered_contrib_list, dim=0)
        features = torch.stack(feature_list, dim=0)

        return gt_image, depth_image, render_idx, rendered_contrib, features