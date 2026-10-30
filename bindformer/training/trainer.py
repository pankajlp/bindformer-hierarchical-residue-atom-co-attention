from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bindformer.training.losses import BindFormerLoss
from bindformer.training.metrics import regression_metrics


def warmup_cosine_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_fn: BindFormerLoss | None = None,
        amp: bool = True,
        grad_clip_norm: float = 5.0,
        warmup_epochs: float = 1.0,
        early_stopping_patience: int | None = 10,
        selection_metric: str = "rmse",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn or BindFormerLoss()
        self.amp = amp and device.type == "cuda"
        self.grad_clip_norm = grad_clip_norm
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.selection_metric = selection_metric
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.scheduler = None

    def _is_better(self, value: float, best: float) -> bool:
        if math.isnan(value):
            return False
        return value < best if self.selection_metric in ("rmse", "mae", "sd") else value > best

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
        checkpoint_dir: str | Path,
        extra_state: dict | None = None,
    ) -> dict[str, float]:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        steps_per_epoch = max(1, len(train_loader))
        self.scheduler = warmup_cosine_schedule(
            self.optimizer, int(self.warmup_epochs * steps_per_epoch), epochs * steps_per_epoch
        )
        lower_is_better = self.selection_metric in ("rmse", "mae", "sd")
        best = float("inf") if lower_is_better else float("-inf")
        best_metrics: dict[str, float] = {}
        stale = 0
        log_path = checkpoint_dir / "log.jsonl"

        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader) if val_loader is not None else {}
            record = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"], **train_stats}
            record.update({f"val_{k}": v for k, v in metrics.items()})
            print(json.dumps(record))
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            state = {"model": self.model.state_dict(), "epoch": epoch, "metrics": metrics, **(extra_state or {})}
            torch.save(state, checkpoint_dir / "last.pt")
            if not metrics:
                continue
            if self._is_better(metrics[self.selection_metric], best):
                best, best_metrics, stale = metrics[self.selection_metric], metrics, 0
                torch.save(state, checkpoint_dir / "best.pt")
            else:
                stale += 1
                if self.early_stopping_patience and stale >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}: no {self.selection_metric} improvement for {stale} epochs")
                    break
        return best_metrics

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        for batch in tqdm(loader, desc="train", leave=False):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(batch)
                losses = self.loss_fn(outputs, batch)
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            for key, value in losses.items():
                totals[f"train_{key}"] = totals.get(f"train_{key}", 0.0) + float(value.detach().cpu())
        return {key: value / max(1, len(loader)) for key, value in totals.items()}

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        self.model.eval()
        ids, preds, targets = [], [], []
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = self._to_device(batch)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(batch)
            ids.extend(batch["complex_id"])
            preds.append(outputs["affinity"].float().cpu())
            targets.append(batch["affinity"].cpu())
        return ids, torch.cat(preds), torch.cat(targets)

    def evaluate(self, loader: DataLoader | None, predictions_path: str | Path | None = None) -> dict[str, float]:
        if loader is None:
            return {}
        ids, pred, target = self.predict(loader)
        if predictions_path is not None:
            with open(predictions_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["complex_id", "pred_pk", "true_pk"])
                writer.writerows(zip(ids, pred.tolist(), target.tolist()))
        return {"n": len(ids), **regression_metrics(pred, target)}

    def _to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {key: self._to_device(value) for key, value in batch.items()}
        return batch
