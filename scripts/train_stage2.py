"""
Stage 2 Training Script (3D Affordance Alignment)

This stage distills knowledge from the pretrained 2D Branch
into the 3D branch, aligning multi-view 2D representations with 3D features.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(".")

from utils.utils import seed_torch, read_yaml
from utils.logger import setup_logger
from utils.metrics import evaluating, cal_SIM_3d

from dataset.laso import LasoDataset
from dataset.piad import PiadDataset
from model.branch_2d import Branch2D
from model.branch_3d import Branch3D
from utils.loss import HM_Loss

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def count_trainable_params(model):
    """Print number of trainable parameters per submodule."""
    for name, module in model.named_children():
        if any(p.requires_grad for p in module.parameters()):
            num = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"Module: {name:20s} | Trainable params: {num / 1e6:.2f}M")


def build_dataloader(cfg):
    """Initialize train/test dataloaders."""
    if cfg["category"] == "piad":
        train_dataset = PiadDataset(cfg["train_split"], cfg["setting"])
        test_dataset = PiadDataset(cfg["test_split"])
    elif cfg["category"] == "laso":
        train_dataset = LasoDataset(cfg["train_split"], cfg["setting"], data_root=cfg["data_root"])
        test_dataset = LasoDataset(cfg["test_split"], data_root=cfg["data_root"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
    )
    return train_loader, test_loader


def build_optimizer(model, opt_cfg):
    """Set up optimizer and scheduler for 3D branch."""
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad],
         "lr": opt_cfg["tlr"]},
    ]
    optimizer = torch.optim.Adam(
        params=param_dicts,
        lr=opt_cfg["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt_cfg["step_size"], gamma=opt_cfg["gamma"]
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------
# Training and Evaluation Loops
# ---------------------------------------------------------------------

def train_one_epoch(model_3d, model_2d, loader, optimizer, device, criterion_hm, logger, epoch, train_cfg):
    """
    One training epoch:
      - Freeze 2D branch (teacher)
      - Compute 3D affordance heatmaps and alignment loss
    """
    model_3d.train()
    model_2d.eval()
    loss_sum = 0

    for _, p in model_2d.named_parameters():
        p.requires_grad = False

    for i, (point, _, _, question, _, label) in enumerate(loader):

        optimizer.zero_grad()
        point, label = point.to(device), label.to(device)

        # --- Forward ---
        pred_3d, feat_3d = model_3d(question, point)
        feat_2d, render_feats = model_2d(question, point, feat_3d)

        # --- Losses ---
        loss_kld = nn.MSELoss()(render_feats, feat_2d)
        loss_hm = criterion_hm(pred_3d, label)
        loss = loss_hm + train_cfg["kl_loss_weight"]*loss_kld

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if i % 10 == 0:
            logger.debug(f"[Epoch {epoch}] Iter {i}/{len(loader)} | Loss: {loss.item():.4f}")

    return loss_sum / len(loader)


def evaluate(model_3d, loader, device, criterion_hm, logger):
    """
    Validation loop:
      - Computes IOU, SIM, MAE, and AUC across all test samples.
    """
    model_3d.eval()
    results, targets = [], []
    total_mae, total_points = 0, 0

    with torch.no_grad():
        for i, (point, _, _, question, _, label) in enumerate(loader):

            point, label = point.to(device), label.to(device)
            pred = model_3d(question, point)

            val_loss = criterion_hm(pred, label)
            mae, n_pts = evaluating(pred, label)
            total_mae += mae.item()
            total_points += n_pts

            results.append(pred.cpu().numpy())
            targets.append(label.cpu().numpy())

            logger.debug(f"[Val] Batch {i}/{len(loader)} | Loss: {val_loss.item():.4f}")

    results, targets = np.array(results), np.array(targets)
    mean_mae = total_mae / total_points

    # Compute similarity and AUC/IOU
    sim_scores = np.array([cal_SIM_3d(r, t) for r, t in zip(results, targets)])
    SIM = np.nanmean(sim_scores)

    IOUs, AUCs = [], []
    IOU_thres = np.linspace(0, 1, 20)
    targets_bin = (targets >= 0.5).astype(int)

    for t_true, p_score in zip(targets_bin, results):
        if np.sum(t_true) == 0:
            continue
        auc = roc_auc_score(t_true.flatten(), p_score.flatten())
        AUCs.append(auc)
        temp_iou = []
        for thr in IOU_thres:
            p_mask = (p_score >= thr).astype(int)
            intersect = np.sum(p_mask & t_true)
            union = np.sum(p_mask | t_true)
            temp_iou.append(intersect / (union + 1e-6))
        IOUs.append(np.mean(temp_iou))

    IOU = np.nanmean(IOUs)
    AUC = np.nanmean(AUCs)
    logger.debug(f"Validation → IOU: {IOU:.4f}, AUC: {AUC:.4f}, SIM: {SIM:.4f}, MAE: {mean_mae:.4f}")
    return IOU, mean_mae


# ---------------------------------------------------------------------
# Main Training Entry
# ---------------------------------------------------------------------

def main(cfg_path="config/train_stage2.yaml"):
    """
    Stage 2: 3D Affordance Alignment Training
    """
    cfg = read_yaml(cfg_path)
    train_cfg = cfg["train"]

    # Select device
    gpu_id = str(train_cfg.get("gpu", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"[INFO] Using GPU {gpu_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch(train_cfg["seed"])
    logger, sign = setup_logger(train_cfg)

    device = torch.device(f"cuda:{train_cfg['gpu']}" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_dataloader({**cfg["dataset"], "batch_size": train_cfg["batch_size"]})

    # Build models
    model_2d = Branch2D(cfg["model_2d"], cfg["renderer"]).to(device)
    model_3d = Branch3D(cfg["model_3d"]).to(device)
    criterion_hm = HM_Loss().to(device)

    # Load pretrained 2D weights (frozen teacher)
    if train_cfg.get("pretrained_2d", None):
        ckpt_path = train_cfg["pretrained_2d"]
        logger.debug(f"Loading pretrained 2D model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_2d.load_state_dict(ckpt["model"], strict=False)

    if train_cfg["resume"]:
        ckpt_path = train_cfg["checkpoint_path"]
        logger.debug(f"Resuming 3D model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model_3d.load_state_dict(ckpt["model"], strict=False)

    optimizer, scheduler = build_optimizer(model_3d, cfg["optimizer"])

    # Count trainable params
    count_trainable_params(model_3d)

    # Training loop
    best_IOU = 0
    save_dir = os.path.join(train_cfg["save_dir"], train_cfg["name"])
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        logger.debug(f"Epoch {epoch} start → lr={optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_one_epoch(model_3d, model_2d, train_loader, optimizer, device, criterion_hm, logger, epoch, train_cfg)
        IOU, mae = evaluate(model_3d, test_loader, device, criterion_hm, logger)
        scheduler.step()

        if IOU > best_IOU:
            best_IOU = IOU
            model_path = os.path.join(save_dir, f"best_model_{sign}.pt")
            torch.save({
                "model": model_3d.state_dict(),
                "optimizer": optimizer.state_dict(),
                "Epoch": epoch
            }, model_path)
            logger.debug(f"New best model saved → IOU={best_IOU:.4f} | {model_path}")

    logger.debug(f"Training complete. Best IOU: {best_IOU:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_stage2.yaml")
    args = parser.parse_args()
    main(args.config)
