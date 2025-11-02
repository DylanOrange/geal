"""
Export Best IoU Samples to PLY
-------------------------------
Use pretrained model and dataset from YAML config.
Compute IoU only, select top-N samples per (affordance, class),
export GT & prediction as PLY, and save all paths to a .txt file.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import open3d as o3d
import sys
sys.path.append(".")

from utils.utils import seed_torch, read_yaml
from utils.metrics import calculate_batch_iou
from dataset.data_utils import CLASSES, AFFORDANCES
from model.branch_3d import Branch3D
from dataset.laso import LasoDataset
from dataset.piad import PiadDataset


def select_top_samples(df, top_n=10):
    df['aff'] = df['aff'].apply(lambda x: AFFORDANCES[x] if 0 <= x < len(AFFORDANCES) else 'Unknown')
    df['cls'] = df['cls'].apply(lambda x: CLASSES[x] if 0 <= x < len(CLASSES) else 'Unknown')

    def _top(group):
        return group.nlargest(min(top_n, len(group)), 'iou')

    return df.groupby(['aff', 'cls'], group_keys=False).apply(_top).reset_index(drop=True)

# -------------------------------
# Save to PLY
# -------------------------------
def save_pair_to_ply(points, gt_mask, pred_mask, save_dir, obj, aff, iou, shape_id):
    os.makedirs(save_dir, exist_ok=True)
    back = np.array([190, 190, 190], dtype=np.float64)
    red = np.array([255, 0, 0], dtype=np.float64)

    def colorize(scores):
        return ((red - back) * scores.reshape(-1, 1) + back) / 255.0

    gt_pc = o3d.geometry.PointCloud()
    gt_pc.points = o3d.utility.Vector3dVector(points)
    gt_pc.colors = o3d.utility.Vector3dVector(colorize(gt_mask))

    pred_pc = o3d.geometry.PointCloud()
    pred_pc.points = o3d.utility.Vector3dVector(points)
    pred_pc.colors = o3d.utility.Vector3dVector(colorize(pred_mask))

    tag = f"{obj}_{aff}_{round(float(iou),3)}_{shape_id}"
    gt_path = os.path.join(save_dir, f"{tag}_GT.ply")
    pred_path = os.path.join(save_dir, f"{tag}_Pred.ply")

    o3d.io.write_point_cloud(gt_path, gt_pc)
    o3d.io.write_point_cloud(pred_path, pred_pc)
    return gt_path, pred_path


# -------------------------------
# Evaluation Loop (IoU only)
# -------------------------------
def evaluate_and_export(model, dataset, device, top_n=10, save_dir="./ply_out", path_txt="./ply_paths.txt"):
    loader = DataLoader(dataset, batch_size=8, num_workers=8, shuffle=False)
    all_preds = []
    records = []

    model.eval()
    with torch.no_grad():
        for bid, (point, cls, binary_label, question, aff_label, label) in tqdm(
            enumerate(loader), total=len(loader), ascii=True
        ):
            point = point.float().to(device)
            label = label.float().to(device)
            pred = model(question, point)
            pred_np = pred.cpu().numpy()
            label_np = label.cpu().numpy()
            cls_np = cls.numpy()
            aff_np = aff_label.numpy()

            iou_np = calculate_batch_iou(pred_np, label_np)
            B = pred_np.shape[0]
            base_idx = bid * B
            for i in range(B):
                records.append({
                    'idx': base_idx + i,
                    'aff': int(aff_np[i]),
                    'cls': int(cls_np[i]),
                    'iou': float(iou_np[i]) if not np.isnan(iou_np[i]) else -1
                })
            all_preds.append(pred_np)

    all_preds = np.vstack(all_preds)
    df = pd.DataFrame(records, columns=['idx','aff','cls','iou'])
    top_df = select_top_samples(df, top_n=top_n)

    print(f"Selected {len(top_df)} top samples for export.")

    os.makedirs(save_dir, exist_ok=True)
    lines = []
    anno = dataset.annotations
    for idx, row in tqdm(top_df.iterrows(), total=len(top_df), ascii=True, desc="Saving PLY"):
        idx = int(row['idx'])
        obj = row['cls']
        aff = row['aff']
        iou = float(row['iou'])
        shape = anno[idx]['point']
        gt_mask = anno[idx]['mask']
        pred_mask = all_preds[idx]
        shape_id = anno[idx].get('shape_id', str(idx))
        gt_path, pred_path = save_pair_to_ply(shape, gt_mask, pred_mask, save_dir, obj, aff, iou, shape_id)
        lines += [gt_path, pred_path]

    with open(path_txt, 'w') as f:
        f.write("\n".join(lines))
    print(f"[Done] Saved {len(lines)} .ply paths to {path_txt}")


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Export Top IoU PLY Samples")
    parser.add_argument("--config", type=str, default="config/evaluation.yaml")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="runs/ply/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    seed_torch(cfg.get("seed", 42))
    model = Branch3D(cfg["model_3d"])
    ckpt = torch.load(cfg["ckpt"], map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    print("Loaded checkpoint:", cfg["ckpt"])

    if cfg["dataset"] == "laso":
        dataset = LasoDataset("test", cfg["setting"], data_root=cfg["data_root"])
    else:
        dataset = PiadDataset("test", cfg["setting"], data_root=cfg["data_root"])

    os.makedirs(args.output_dir, exist_ok=True)
    txt_path = os.path.join(args.output_dir, "ply_paths.txt")

    evaluate_and_export(model, dataset, device, top_n=args.top_n,
                        save_dir=args.output_dir, path_txt=txt_path)


if __name__ == "__main__":
    main()