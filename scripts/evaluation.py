"""
3D Affordance Evaluation Script
--------------------------------
Evaluate pretrained models on test sets.
Compute per-category, per-affordance, and overall metrics (IoU, AUC, SIM, MAE).

"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
sys.path.append(".")

from model.branch_3d import Branch3D
from dataset.laso import LasoDataset
from dataset.piad import PiadDataset
from utils.utils import seed_torch, read_yaml
from utils.metrics import calculate_batch_iou_auc, calculate_batch_sim, calculate_batch_mae
from dataset.data_utils import CLASSES, AFFORDANCES

# -------------------------------
# Evaluation core
# -------------------------------

def evaluate(model, dataloader, device):
    """
    Run evaluation loop and compute metrics for:
      - Each object class
      - Each affordance type
      - Overall statistics
    """
    num_cls, num_aff = len(CLASSES), len(AFFORDANCES)

    # Accumulators for metrics
    category_iou = np.zeros(num_cls)
    category_auc = np.zeros(num_cls)
    category_sim = np.zeros(num_cls)
    category_mae = np.zeros(num_cls)
    category_count = np.zeros(num_cls)

    affordance_iou = np.zeros(num_aff)
    affordance_auc = np.zeros(num_aff)
    affordance_sim = np.zeros(num_aff)
    affordance_mae = np.zeros(num_aff)
    affordance_count = np.zeros(num_aff)

    model.eval()

    with torch.no_grad():
        for i, (point, cls, binary_label, question, aff_label, label) in tqdm(
            enumerate(dataloader), total=len(dataloader), ascii=True
        ):
            # Move data to GPU
            point, label, binary_label = point.float().to(device), label.float().to(device), binary_label.float().to(device)

            # Forward pass
            pred = model(question, point)

            # Convert to numpy for metric computation
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()
            cls = cls.numpy()
            aff_label = aff_label.numpy()

            # Compute metrics
            iou, auc = calculate_batch_iou_auc(pred, label)
            sim = calculate_batch_sim(pred, label)
            mae = calculate_batch_mae(pred, label)

            # Accumulate metrics for each category and affordance
            for idx, (cls_id, aff_id) in enumerate(zip(cls, aff_label)):
                if np.isnan(iou[idx]):
                    continue
                category_iou[cls_id] += iou[idx]
                category_auc[cls_id] += auc[idx]
                category_sim[cls_id] += sim[idx]
                category_mae[cls_id] += mae[idx]
                category_count[cls_id] += 1

                affordance_iou[aff_id] += iou[idx]
                affordance_auc[aff_id] += auc[idx]
                affordance_sim[aff_id] += sim[idx]
                affordance_mae[aff_id] += mae[idx]
                affordance_count[aff_id] += 1

    # -------------------------------
    # Compute Mean Metrics
    # -------------------------------
    overall_metrics = {
        "IOU": category_iou.sum() / category_count.sum(),
        "AUC": category_auc.sum() / category_count.sum(),
        "SIM": category_sim.sum() / category_count.sum(),
        "MAE": category_mae.sum() / category_count.sum(),
    }

    # Mean per-class and per-affordance
    category_metrics = {
        i: {
            "IOU": category_iou[i] / category_count[i] if category_count[i] > 0 else np.nan,
            "AUC": category_auc[i] / category_count[i] if category_count[i] > 0 else np.nan,
            "SIM": category_sim[i] / category_count[i] if category_count[i] > 0 else np.nan,
            "MAE": category_mae[i] / category_count[i] if category_count[i] > 0 else np.nan,
        }
        for i in range(num_cls)
    }

    affordance_metrics = {
        i: {
            "IOU": affordance_iou[i] / affordance_count[i] if affordance_count[i] > 0 else np.nan,
            "AUC": affordance_auc[i] / affordance_count[i] if affordance_count[i] > 0 else np.nan,
            "SIM": affordance_sim[i] / affordance_count[i] if affordance_count[i] > 0 else np.nan,
            "MAE": affordance_mae[i] / affordance_count[i] if affordance_count[i] > 0 else np.nan,
        }
        for i in range(num_aff)
    }

    return category_metrics, affordance_metrics, overall_metrics

# -------------------------------
# Result printing and saving
# -------------------------------
def save_metrics(category_metrics, affordance_metrics, overall_metrics, save_path):
    """Print formatted results and save to text file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    category_df = pd.DataFrame(category_metrics).T
    affordance_df = pd.DataFrame(affordance_metrics).T
    category_df.index = CLASSES
    affordance_df.index = AFFORDANCES

    # Console output
    print("\n===== Category Metrics =====")
    print(category_df.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n===== Affordance Metrics =====")
    print(affordance_df.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n===== Overall Metrics =====")
    print({k: f"{v:.3f}" for k, v in overall_metrics.items()})

    # File output
    with open(save_path, "w") as f:
        f.write("Category Metrics:\n")
        f.write(category_df.to_string(float_format=lambda x: f"{x:.3f}"))
        f.write("\n\nAffordance Metrics:\n")
        f.write(affordance_df.to_string(float_format=lambda x: f"{x:.3f}"))
        f.write("\n\nOverall Metrics:\n")
        for k, v in overall_metrics.items():
            f.write(f"{k}: {v:.3f}\n")


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D Affordance Model")
    parser.add_argument("--config", type=str, default="config/evaluation.yaml")
    parser.add_argument("--output", type=str, default="runs/result/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load config and model
    cfg = read_yaml(args.config)

    seed_torch(cfg["seed"])

    model = Branch3D(cfg["model_3d"])
    ckpt = torch.load(cfg["ckpt"], map_location=device)
    status = model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    print("\n Checkpoint loaded:", cfg["ckpt"])
    if status.missing_keys:
        print("Missing keys:", status.missing_keys)
    if status.unexpected_keys:
        print("Unexpected keys:", status.unexpected_keys)

    # Load dataset
    if cfg["dataset"] == "laso":
        dataset = LasoDataset("test", cfg["setting"], data_root=cfg["data_root"])
    else:
        dataset = PiadDataset("test", cfg["setting"], data_root=cfg["data_root"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Evaluate
    category_metrics, affordance_metrics, overall_metrics = evaluate(model, loader, device)

    # Save results
    ckpt_name = os.path.splitext(os.path.basename(cfg["ckpt"]))[0]
    output_file = os.path.join(args.output, f"{ckpt_name}_{cfg['dataset']}_{cfg['setting']}.txt")
    save_metrics(category_metrics, affordance_metrics, overall_metrics, output_file)


if __name__ == "__main__":
    main()
