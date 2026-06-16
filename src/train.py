"""
Training loop for PatchTST on GNSS PWV data.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models import build_model_from_config
from src.utils.metrics import compute_metrics


def get_device(config):
    """Auto-detect or use configured device."""
    device_cfg = config.training.get("device", "auto") if hasattr(config, "training") else "auto"
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_model(config):
    """Instantiate model from config."""
    return build_model_from_config(config)


def build_optimizer(model, config, steps_per_epoch=None):
    """Build optimizer and scheduler."""
    tcfg = config.training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
    )

    scheduler = None
    step_every_batch = False

    if tcfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg.T_max, eta_min=tcfg.eta_min
        )
    elif tcfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif tcfg.scheduler == "cosine_warmup":
        if steps_per_epoch is None:
            raise ValueError("cosine_warmup requires steps_per_epoch")
        warmup_epochs = getattr(tcfg, "warmup_epochs", 5)
        total_steps = tcfg.epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            min_lr_ratio = tcfg.eta_min / tcfg.lr
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        step_every_batch = True

    return optimizer, scheduler, step_every_batch


def train_epoch(model, dataloader, optimizer, criterion, device, use_amp=False, epoch=1, scheduler=None, step_every_batch=False, max_grad_norm=None):
    """Single training epoch with progress bar."""
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda") if use_amp and torch.cuda.is_available() else None

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch:03d}", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            # Gradient clipping
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            # Gradient clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None and step_every_batch:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    """Validation pass with progress bar."""
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validate", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)

            all_preds.append(out.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    # Compute metrics
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    metrics = compute_metrics(preds, trues, ["mse", "mae", "rmse"])

    return avg_loss, metrics


def train(config, train_loader, val_loader, exp_dir):
    """Full training pipeline."""
    log_path = os.path.join(exp_dir, "train.log")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    # Init log file
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.flush()

    device = get_device(config)
    model = build_model(config).to(device)
    criterion = nn.MSELoss()
    optimizer, scheduler, step_every_batch = build_optimizer(model, config, len(train_loader))

    tcfg = config.training
    use_amp = tcfg.use_amp and device.type == "cuda"
    val_interval = getattr(tcfg, "val_interval", 1)

    # Logging
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tb"))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[Train] Device: {device}")
    log(f"[Train] Model params: {total_params:,} (trainable: {trainable_params:,})")
    log(f"[Train] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    log(f"[Train] AMP: {use_amp} | Val interval: {val_interval}")
    log(f"[Train] Scheduler: {tcfg.scheduler} | step_every_batch: {step_every_batch}")
    log("-" * 60)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, tcfg.epochs + 1):
        start = time.time()

        max_grad_norm = getattr(tcfg, "max_grad_norm", None)
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_amp, epoch=epoch, scheduler=scheduler, step_every_batch=step_every_batch,
            max_grad_norm=max_grad_norm
        )

        if epoch % val_interval == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, device)

            if scheduler is not None and not step_every_batch:
                scheduler.step()

            epoch_time = time.time() - start
            current_lr = optimizer.param_groups[0]["lr"]

            # TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/val_mae", val_metrics["mae"], epoch)
            writer.add_scalar("Metrics/val_rmse", val_metrics["rmse"], epoch)
            writer.add_scalar("lr", current_lr, epoch)

            log(
                f"Epoch {epoch:03d}/{tcfg.epochs} | "
                f"lr: {current_lr:.6f} | "
                f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | "
                f"val_mae: {val_metrics['mae']:.4f} | val_rmse: {val_metrics['rmse']:.4f} | "
                f"time: {epoch_time:.1f}s"
            )

            # Checkpoint
            if epoch % tcfg.save_interval == 0:
                ckpt_path = os.path.join(exp_dir, f"ckpt_epoch_{epoch}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, ckpt_path)
                log(f"[Train] Saved checkpoint: {ckpt_path}")

            # Early stopping
            if val_loss < best_val_loss - tcfg.delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_path = os.path.join(exp_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "metrics": val_metrics,
                }, best_path)
                log(f"[Train] New best model saved (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            if patience_counter >= tcfg.patience:
                log(f"[Train] Early stopping at epoch {epoch}")
                break
        else:
            if scheduler is not None and not step_every_batch:
                scheduler.step()

            epoch_time = time.time() - start
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("lr", current_lr, epoch)
            log(
                f"Epoch {epoch:03d}/{tcfg.epochs} | "
                f"lr: {current_lr:.6f} | "
                f"train_loss: {train_loss:.6f} | val: skipped | "
                f"time: {epoch_time:.1f}s"
            )

    writer.close()
    log(f"[Train] Best val_loss: {best_val_loss:.6f}")
    log(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return model
