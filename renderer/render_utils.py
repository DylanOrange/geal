import torch
import matplotlib.cm as cm
import torch.nn.functional as F

def depth_to_normal(depth_map):

    # Ensure depth_map is of shape (B, H, W)
    B, _, H, W = depth_map.shape
    depth_map = depth_map*10
    # Pad depth map with 1 pixel padding to avoid boundary issues
    depth_map_padded = F.pad(depth_map, (1, 1, 1, 1), mode='replicate')

    # Compute gradients in x (dzdx) and y (dzdy) directions
    dzdx = (depth_map_padded[:, :, 2:, 1:-1] - depth_map_padded[:, :, :-2, 1:-1]) / 2.0  # Horizontal gradient
    dzdy = (depth_map_padded[:, :, 1:-1, 2:] - depth_map_padded[:, :, 1:-1, :-2]) / 2.0  # Vertical gradient

    # Stack gradients into normal map (nx = -dzdx, ny = -dzdy, nz = 1)
    normal_x = -dzdx
    normal_y = -dzdy
    normal_z = torch.ones_like(depth_map)  # Normals in the z-direction

    # Stack the x, y, z components into a single tensor (B, 3, H, W)
    normal_map = torch.cat([normal_x, normal_y, normal_z], dim=1)

    # Normalize the normal vectors to unit length
    normal_map = F.normalize(normal_map, p=2, dim=1)  # Normalize along the channel dimension

    rgb_map = (normal_map + 1) / 2.0
    rgb_map = torch.clamp(rgb_map, 0, 1)

    depth_mask = (depth_map == 0) 
    depth_mask_rgb = depth_mask.expand(-1, 3, -1, -1) 
    rgb_map[depth_mask_rgb] = 1

    return rgb_map  # Shape (B, H, W, 3)

def depth_to_rgb(depth_map):
    """
    Converts a batch of depth maps to RGB using a colormap.

    :param depth_map: torch.Tensor of shape (B, 1, H, W) with depth values.
    :return: rgb_depth: torch.Tensor of shape (B, H, W, 3) with values in [0, 1] for RGB visualization.
    """
    B, _, H, W = depth_map.shape

    # Normalize the depth map to the range [0, 1]
    depth_min = torch.min(depth_map)
    depth_max = torch.max(depth_map)
    normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)  # Add epsilon to avoid division by zero

    rgb_depth = normalized_depth.repeat(1,3,1,1)
    # # Convert the depth map to numpy for use with matplotlib colormaps
    depth_np = normalized_depth.squeeze(1).detach().cpu().numpy()  # Shape (B, H, W)

    # Apply a colormap (viridis is used here, but can choose others like 'plasma', 'inferno', etc.)
    colormap = cm.get_cmap('viridis')

    # Convert depth maps to RGB format
    rgb_depth = []
    for i in range(B):
        depth_rgb = colormap(depth_np[i])[:, :, :3]  # Get RGB values, ignore the alpha channel
        rgb_depth.append(torch.tensor(depth_rgb, dtype=torch.float32, device=depth_map.device).permute(2, 0, 1))  # Convert to tensor and permute to (3, H, W)

    # Stack RGB tensors into a batch
    rgb_depth = torch.stack(rgb_depth)

    return rgb_depth

def map_back(color_map, idx_map, contrib_map):
    
    N_points = 2048
    B, V, C, H, W = color_map.shape
    n_contributor = contrib_map.shape[2]

    # Expand the color_map to match the number of contributors and flatten necessary dimensions
    color_map_expanded = color_map.expand(-1, -1, n_contributor, -1, -1).reshape(B, V * n_contributor, H * W)
    idx_map_flattened = idx_map.view(B, V * n_contributor, H * W).long()
    contribution_map_flattened = contrib_map.view(B, V * n_contributor, H * W)

    # Compute weighted colors by contributions without redundant reshapes
    weighted_colors = color_map_expanded * contribution_map_flattened  # Shape: (B, V * n_contributor, H * W)

    # Flatten to (B, -1) once for efficient scatter_add operations
    idx_map_flattened = idx_map_flattened.flatten(1)
    weighted_colors = weighted_colors.flatten(1)
    contribution_map_flattened = contribution_map_flattened.flatten(1)

    # Initialize accumulation tensors
    point_colors_sum = torch.zeros(B, N_points, device=color_map.device)
    contribution_sums = torch.zeros(B, N_points, device=color_map.device)

    # Accumulate weighted colors and contributions per point using scatter_add
    point_colors_sum.scatter_add_(1, idx_map_flattened, weighted_colors)
    contribution_sums.scatter_add_(1, idx_map_flattened, contribution_map_flattened)

    # Normalize to get the final color for each point
    point_colors = point_colors_sum / (contribution_sums + 1e-8)


    return point_colors