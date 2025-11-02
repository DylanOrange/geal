"""
Render 3D point clouds into multi-view 2D images
-------------------------------------------------------------
This script uses the Gaussian Renderer to visualize each shape
in multiple viewpoints. For each (object, affordance) pair,
it produces several RGB renderings from different camera angles.

Inputs:
    - data_root/anno_train.pkl : annotations (shape_id, class, affordance, mask)
    - data_root/objects_train.pkl : raw 3D point clouds (shape_id -> points)

Outputs:
    - debug/{shape_id}_{cls}_{affordance}_{ver}_{hor}.jpg

Usage:
    python visualize_dataset.py
"""
import os
import pickle
import torch
import numpy as np
from torchvision import utils as vutils

from renderer.gaussian_model import Renderer, MiniCam, BasicPointCloud
from renderer.cam_utils import orbit_camera, OrbitCamera


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    pc = pc * 0.5
    return pc, centroid, m

def initilize_point(annotation, objects):

    shape_id = annotation['shape_id']
    cls = annotation['class']
    affordance = annotation['affordance']
    gt_mask = annotation['mask']

    points = objects[str(shape_id)]
    points,_,_ = pc_normalize(points)#2048,3

    num_pts = points.shape[0]
    color = 192*np.ones((num_pts,3))/255.0
    color = gt_mask.reshape(-1,1)*color
    pcd = BasicPointCloud(points=points, colors=color, normals=np.zeros((num_pts, 3)))
    
    return pcd, shape_id, cls, affordance

def save_image(input_tensor, filename):
    """
    :param input_tensor: tensor
    :param filename:     """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))

    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)

if __name__=='__main__':
    
    render = Renderer(sh_degree=0)
    data_root=''
    split = 'train'

    n_views = 7
    hor_angles = np.linspace(-60, 60, n_views)
    ver = -30  
    render_resolution = 224
    base_cam = OrbitCamera(800, 800, r=2, fovy=49.1)
    
    with open(os.path.join(data_root, f'anno_{split}.pkl'), 'rb') as f:
        anno = pickle.load(f)

    with open(os.path.join(data_root, f'objects_{split}.pkl'), 'rb') as f:
        objects = pickle.load(f)

    for annotation in anno:
        pcd, shape_id, cls, affordance = initilize_point(annotation, objects)
        render.initialize(pcd) 
        for i in range(n_views):
            hor = hor_angles[i]  # Get the horizontal angle for each view

            pose = orbit_camera(ver, hor, 1.5)

            cur_cam = MiniCam(pose, render_resolution, render_resolution, base_cam.fovy, base_cam.fovx, base_cam.near, base_cam.far)    
            bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            out = render.render(cur_cam, bg_color=bg_color)

            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            file_name = f'debug/{shape_id}_{cls}_{affordance}_{ver}_{hor}.jpg'
            save_image(image, file_name)