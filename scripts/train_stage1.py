"""
Stage 1 Training Script (2D Affordance Prediction Branch)

"""
import torch
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader
import sys
sys.path.append(".")

from utils.utils import seed_torch, read_yaml
from utils.logger import setup_logger
from utils.metrics import evaluating_2d
from renderer.gaussian_render import Gaussian_Renderer
from model.branch_2d import Branch2D
from dataset.laso import LasoDataset
from dataset.piad import PiadDataset

def count_trainable_params(model):
    """
    Print the number of trainable parameters in each submodule.
    Helps to identify which parts of the network are fine-tuned.
    """
    for name, module in model.named_children():
        if any(p.requires_grad for p in module.parameters()):
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"Module: {name} | Trainable params: {param_count / 1e6:.2f}M")


def build_model(cfg, render_cfg):
    """Build the main Branch2D model according to YAML configuration."""
    return Branch2D(cfg, render_cfg)


def build_dataloader(cfg):
    """
    Create PyTorch DataLoader for training and testing.

    Args:
        cfg: dataset configuration (batch size, splits, etc.)
    Returns:
        train_loader, test_loader
    """
    if cfg["category"] == "piad":
        train_dataset = PiadDataset(cfg["train_split"], cfg["setting"], data_root=cfg["data_root"])
        test_dataset = PiadDataset(cfg["test_split"], data_root=cfg["data_root"])
    elif cfg["category"] == "laso":
        train_dataset = LasoDataset(cfg["train_split"], cfg["setting"], data_root=cfg["data_root"])
        test_dataset = LasoDataset(cfg["test_split"], data_root=cfg["data_root"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=cfg["shuffle"],
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False
    )
    return train_loader, test_loader


def build_optimizer(model, opt_cfg):
    """
    Build optimizer and learning rate scheduler.
    Two parameter groups are defined:
      (1) regular model parameters
      (2) text encoder parameters (usually lower learning rate)
    """
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad],
         "lr": opt_cfg["tlr"]}
    ]

    optimizer = torch.optim.Adam(
        params=param_dicts,
        lr=opt_cfg["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt_cfg["step_size"],
        gamma=opt_cfg["gamma"]
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, device, renderer, logger, epoch):
    """
    Run one training epoch.

    For each batch:
      - Render GT grayscale maps from 3D points and labels
      - Forward pass through the 2D model
      - Compute binary cross-entropy loss (pixel-wise)
      - Backpropagate and update weights
    """
    model.train()
    loss_sum = 0

    for i, (point, _, _, question, _, label) in enumerate(loader):
        optimizer.zero_grad()
        point, label = point.to(device), label.to(device)

        # Render ground truth grayscale maps using differentiable renderer
        with torch.no_grad():
            gt_images = torch.stack([renderer(p, l)[0] for p, l in zip(point, label)])
            render_dim = gt_images.shape[-1]
            gray_images = gt_images.mean(dim=2, keepdim=True).reshape(-1, 1, render_dim, render_dim)

        # Forward pass
        pred = model(question, point)

        # Binary classification loss (affordance heatmap)
        loss = nn.BCELoss()(pred, gray_images)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        if i % 10 == 0:
            logger.debug(f"[Epoch {epoch}] Iter {i}/{len(loader)} | Loss: {loss.item():.4f}")

    return loss_sum / len(loader)


def evaluate(model, loader, device, renderer, logger):
    """
    Evaluate model performance on validation set using two metrics:
      - SIM (similarity)
      - MAE (mean absolute error)
    """
    model.eval()
    SIM_list, MAE_list = [], []

    with torch.no_grad():
        for i, (point, _, _, question, _, label) in enumerate(loader):
            point, label = point.to(device), label.to(device)

            # Render GT grayscale maps
            gt_images = torch.stack([renderer(p, l)[0] for p, l in zip(point, label)])
            render_dim = gt_images.shape[-1]
            gray_images = gt_images.mean(dim=2, keepdim=True).reshape(-1, 1, render_dim, render_dim)

            # Prediction
            pred = model(question, point)

            # Convert to numpy for evaluation
            sim, mae = evaluating_2d(pred.cpu().numpy(), gray_images.cpu().numpy())
            SIM_list.append(sim)
            MAE_list.append(mae)

    mSIM = sum(SIM_list) / len(SIM_list)
    mMAE = sum(MAE_list) / len(MAE_list)
    logger.debug(f"Validation → mSIM: {mSIM:.4f}, mMAE: {mMAE:.4f}")
    return mMAE


def main(cfg_path="config/train_stage1.yaml"):
    """
    Stage 1 Training Script (2D Affordance Branch).

    Pipeline:
        1. Read configuration and initialize random seed
        2. Build dataloaders, model, optimizer, and renderer
        3. Run multiple epochs with validation after each
        4. Save the model achieving the lowest validation MAE
    """
    # Load YAML configuration
    cfg = read_yaml(cfg_path)
    train_cfg = cfg["train"]

    # Select device
    gpu_id = str(train_cfg.get("gpu", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"[INFO] Using GPU {gpu_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize random seed & logger
    seed_torch(train_cfg["seed"])
    logger, sign = setup_logger(cfg["train"])

    # Data pipeline
    train_loader, test_loader = build_dataloader({
        **cfg["dataset"],
        "batch_size": train_cfg["batch_size"]
    })

    # Model, optimizer, renderer
    model = build_model(cfg["model_2d"], cfg["renderer"]).to(device)
    optimizer, scheduler = build_optimizer(model, cfg["optimizer"])
    renderer = Gaussian_Renderer(**cfg["renderer"], device=device)

    # Display number of trainable parameters
    count_trainable_params(model)

    # Training loop
    best_MAE = float("inf")
    save_dir = os.path.join(train_cfg["save_dir"], train_cfg["name"])
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        logger.debug(f"Epoch {epoch} start → learning rate {optimizer.param_groups[0]['lr']:.6f}")

        # Train and validate
        train_loss = train_one_epoch(model, train_loader, optimizer, device, renderer, logger, epoch)
        val_mae = evaluate(model, test_loader, device, renderer, logger)
        scheduler.step()

        # Save model if improved
        if val_mae < best_MAE:
            best_MAE = val_mae
            model_path = os.path.join(save_dir, f"best_model_{sign}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "Epoch": epoch
            }, model_path)
            logger.debug(f"Best model saved → MAE={best_MAE:.4f} | {model_path}")

    logger.debug(f"Training complete. Best MAE: {best_MAE:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_stage1.yaml")
    opt = parser.parse_args()
    main(opt.config)
