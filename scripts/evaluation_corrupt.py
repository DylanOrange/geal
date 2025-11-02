"""
3D Affordance Robustness Evaluation Script
------------------------------------------
Evaluate pretrained models on multiple corrupted test sets (scale, jitter, rotate, etc.)
across multiple severity levels (0–4), on either LASO or PIAD datasets.
Compute per-corruption averaged metrics (IoU, AUC, SIM, MAE) and save all results to a .txt file.
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
from dataset.corrupt import CorruptDataset
from utils.utils import seed_torch, read_yaml
from utils.metrics import calculate_batch_iou_auc, calculate_batch_sim, calculate_batch_mae

CORRUPT_TYPES = ['scale', 'jitter', 'rotate', 'dropout_global', 'dropout_local', 'add_global', 'add_local']

# -------------------------------
# Evaluation core
# -------------------------------
def evaluate(model, dataloader, device):
    """Evaluate model and return overall metrics."""
    category_iou = category_auc = category_sim = category_mae = category_count = 0

    model.eval()
    with torch.no_grad():
        for point, cls, label, question, aff_label in tqdm(dataloader, total=len(dataloader), ascii=True):
            point, label = point.float().to(device), label.float().to(device)
            pred, _ = model(question, point)
            pred, label = pred.cpu().numpy(), label.cpu().numpy()

            iou, auc = calculate_batch_iou_auc(pred, label)
            sim, mae = calculate_batch_sim(pred, label), calculate_batch_mae(pred, label)

            valid = ~np.isnan(iou)
            if np.sum(valid) == 0:
                continue
            category_iou += np.sum(iou[valid])
            category_auc += np.sum(auc[valid])
            category_sim += np.sum(sim[valid])
            category_mae += np.sum(mae[valid])
            category_count += np.sum(valid)

    overall = {
        "IOU": category_iou / category_count,
        "AUC": category_auc / category_count,
        "SIM": category_sim / category_count,
        "MAE": category_mae / category_count,
    }
    return overall


# -------------------------------
# Save results (txt, consistent with evaluation.py)
# -------------------------------
def save_summary_txt(results, output_file):
    """Save corruption results in formatted txt file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(results).T
    df.index.name = "Corruption"
    pd.options.display.float_format = '{:.3f}'.format

    with open(output_file, "w") as f:
        f.write("===== Robustness Evaluation Summary (Mean over 5 levels) =====\n")
        f.write(df.to_string(float_format=lambda x: f"{x:.3f}"))
        f.write("\n\n")

    print("\n===== Summary of Robustness Across Corruptions =====")
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n Results saved to: {output_file}")


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D Affordance Model Robustness")
    parser.add_argument("--config", type=str, default="config/evaluation_corrupt.yaml")
    parser.add_argument("--output", type=str, default="runs/result/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = read_yaml(args.config)
    seed_torch(cfg.get("seed", 42))

    # Load model
    model = Branch3D(cfg["model_3d"])
    ckpt = torch.load(cfg["ckpt"], map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    print(f"\n Checkpoint loaded: {cfg['ckpt']}")

    results = {}

    for corrupt_type in CORRUPT_TYPES:
        print(f"\n=== Evaluating Corruption Type: {corrupt_type} ===")
        level_metrics = []

        for level in range(5):
            print(f"\n>>> Level {level}")
            dataset = CorruptDataset(corrupt_type=corrupt_type, level=level, data_root=cfg["data_root"])

            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            metrics = evaluate(model, dataloader, device)
            level_metrics.append(metrics)
            print({k: f"{v:.3f}" for k, v in metrics.items()})

        # average over 5 levels
        avg_metrics = {
            "IOU": np.mean([m["IOU"] for m in level_metrics]),
            "AUC": np.mean([m["AUC"] for m in level_metrics]),
            "SIM": np.mean([m["SIM"] for m in level_metrics]),
            "MAE": np.mean([m["MAE"] for m in level_metrics]),
        }
        results[corrupt_type] = avg_metrics
        print(f"\n--- Average over 5 levels for {corrupt_type}: {avg_metrics} ---")

    # Save and print summary
    ckpt_name = os.path.splitext(os.path.basename(cfg["ckpt"]))[0]
    output_file = os.path.join(args.output, f"{ckpt_name}_{cfg['dataset']}_robustness.txt")
    save_summary_txt(results, output_file)


if __name__ == "__main__":
    main()
