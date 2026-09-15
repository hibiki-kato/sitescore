"""Span MLM masking, pretraining, and plots for V8S2."""

import contextlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

_CACHE_DIR = Path(tempfile.gettempdir()) / "dmel-v8s2-mpl-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from . import common
from .checkpoints import FORMAT_VERSION, MODEL_KIND
from .model import DNA_VOCAB_SIZE, MASK_TOKEN


def mask_span_mlm(
    sequence,
    mask_fraction=0.22,
    min_span=3,
    max_span=30,
    mask_token=MASK_TOKEN,
    generator=None,
):
    """Create span-masked inputs and labels, excluding N bases from the loss."""
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be in (0, 1)")
    if min_span <= 0 or max_span < min_span:
        raise ValueError("invalid span length bounds")
    device = sequence.device
    inputs = sequence.clone()
    targets = torch.full_like(sequence, -100)
    batch, length = sequence.shape
    for row in range(batch):
        eligible = torch.nonzero(sequence[row] != common.N, as_tuple=False).flatten()
        if eligible.numel() == 0:
            continue
        target_count = max(1, int(round(mask_fraction * eligible.numel())))
        selected = torch.zeros(length, dtype=torch.bool, device=device)
        attempts = 0
        while int(selected.sum()) < target_count and attempts < target_count * 20:
            attempts += 1
            start_index = torch.randint(
                0, eligible.numel(), (1,), device=device, generator=generator
            ).item()
            span_len = torch.randint(
                min_span,
                max_span + 1,
                (1,),
                device=device,
                generator=generator,
            ).item()
            start = int(eligible[start_index].item())
            end = min(length, start + span_len)
            span = torch.arange(start, end, device=device)
            span = span[sequence[row, span] != common.N]
            selected[span] = True
        if int(selected.sum()) > target_count:
            chosen = torch.nonzero(selected, as_tuple=False).flatten()
            order = torch.randperm(chosen.numel(), device=device, generator=generator)
            selected[chosen[order[target_count:]]] = False
        targets[row, selected] = sequence[row, selected]
        replace = torch.rand(length, device=device, generator=generator)
        mask_positions = selected & (replace < 0.8)
        random_positions = selected & (replace >= 0.8) & (replace < 0.9)
        inputs[row, mask_positions] = mask_token
        if random_positions.any():
            inputs[row, random_positions] = torch.randint(
                0,
                DNA_VOCAB_SIZE - 1,
                (int(random_positions.sum()),),
                device=device,
                generator=generator,
            )
    return inputs, targets


def autocast_context(device, amp):
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def mlm_metrics(logits, targets):
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
    mask = targets != -100
    if mask.any():
        pred = logits.argmax(dim=-1)
        accuracy = (pred[mask] == targets[mask]).float().mean()
        n_masked = int(mask.sum().item())
    else:
        accuracy = logits.sum() * 0.0
        n_masked = 0
    return loss, {"loss": loss, "accuracy": accuracy, "n_masked": n_masked}


def build_optimizer(model, learning_rate, weight_decay):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in lowered:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


def make_lr_schedule(base_lr, total_steps, warmup_steps, min_lr_ratio=0.1):
    warmup_steps = min(max(1, warmup_steps), max(1, total_steps))
    min_lr = base_lr * min_lr_ratio

    def lr_at(step):
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    return lr_at


@torch.no_grad()
def evaluate_mlm(model, loader, device, amp, mask_fraction, seed):
    model.eval()
    generator = torch.Generator()
    generator.manual_seed(seed)
    loss_sum = accuracy_sum = masked_sum = 0.0
    batches = 0
    for sequence in loader:
        masked, targets = mask_span_mlm(
            sequence,
            mask_fraction=mask_fraction,
            generator=generator,
        )
        masked = masked.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(masked, task="mlm")
            loss, values = mlm_metrics(logits, targets)
        loss_sum += float(loss.item())
        accuracy_sum += float(values["accuracy"].item())
        masked_sum += values["n_masked"]
        batches += 1
    val_loss = loss_sum / max(batches, 1)
    return {
        "loss": val_loss,
        "accuracy": accuracy_sum / max(batches, 1),
        "perplexity": float(math.exp(min(val_loss, 20.0))),
        "n_masked": int(masked_sum),
    }


def save_mlm_curves(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], "o-", label="train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history], "o-", label="val")
    axes[0, 0].set_title("MLM loss")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [row["train_accuracy"] for row in history], "o-", label="train")
    axes[0, 1].plot(epochs, [row["val_accuracy"] for row in history], "o-", label="val")
    axes[0, 1].set_title("MLM masked-base accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [row["train_perplexity"] for row in history], "o-", label="train")
    axes[1, 0].plot(epochs, [row["val_perplexity"] for row in history], "o-", label="val")
    axes[1, 0].set_title("MLM perplexity")
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, [row["learning_rate"] for row in history], "o-")
    axes[1, 1].set_title("Learning rate")
    for axis in axes.ravel():
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_dir / "mlm_curves.png", dpi=150)
    plt.close(figure)


def train_mlm(
    model,
    train_loader,
    val_loader,
    device,
    out_dir,
    profile_name,
    profile_stats,
    model_config,
    epochs,
    learning_rate,
    weight_decay,
    grad_accum,
    warmup_steps,
    max_grad_norm,
    patience,
    amp,
    mask_fraction,
    run_args,
    checkpoint_name="mlm_best.pt",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max(1, epochs * optimizer_steps_per_epoch)
    lr_at = make_lr_schedule(learning_rate, total_steps, warmup_steps)
    metrics_path = out_dir / "metrics.jsonl"
    history = []
    best_loss = float("inf")
    no_improvement = 0
    optimizer_step = 0
    generator = torch.Generator()
    generator.manual_seed(int(run_args.get("seed", 42)))
    log_every = max(1, int(run_args.get("log_every", 25)))

    with open(metrics_path, "w") as metrics_file:
        for epoch in range(1, epochs + 1):
            model.train()
            started = time.time()
            loss_sum = accuracy_sum = masked_sum = 0.0
            batches = accumulated = 0
            optimizer.zero_grad(set_to_none=True)
            print(
                f"MLM epoch {epoch}: {len(train_loader):,} batches; "
                f"logging every {log_every} batches",
                flush=True,
            )
            for batch_index, sequence in enumerate(train_loader, start=1):
                if batch_index == 1:
                    print("starting first MLM batch", flush=True)
                masked, targets = mask_span_mlm(
                    sequence,
                    mask_fraction=mask_fraction,
                    generator=generator,
                )
                masked = masked.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(optimizer_step)
                with autocast_context(device, amp):
                    logits = model(masked, task="mlm")
                    loss, values = mlm_metrics(logits, targets)
                    scaled = loss / grad_accum
                if not torch.isfinite(scaled):
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    print(
                        f"WARN: skipped non-finite MLM loss epoch {epoch} batch {batch_index}",
                        flush=True,
                    )
                    continue
                scaled.backward()
                accumulated += 1
                if accumulated == grad_accum:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    accumulated = 0
                loss_sum += float(loss.item())
                accuracy_sum += float(values["accuracy"].item())
                masked_sum += values["n_masked"]
                batches += 1
                if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(train_loader):
                    elapsed = time.time() - started
                    rate = batch_index * train_loader.batch_size / max(elapsed, 1e-9)
                    print(
                        f"[MLM epoch {epoch} {batch_index}/{len(train_loader)}] "
                        f"loss={loss_sum / max(batches, 1):.4f} "
                        f"acc={accuracy_sum / max(batches, 1):.3f} "
                        f"lr={lr_at(optimizer_step):.2e} {rate:.1f} samples/s",
                        flush=True,
                    )
            if accumulated:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            train_loss = loss_sum / max(batches, 1)
            validation = evaluate_mlm(
                model,
                val_loader,
                device,
                amp,
                mask_fraction=mask_fraction,
                seed=int(run_args.get("seed", 42)) + epoch,
            )
            row = {
                "epoch": epoch,
                "elapsed_seconds": time.time() - started,
                "learning_rate": lr_at(optimizer_step),
                "train_loss": train_loss,
                "train_accuracy": accuracy_sum / max(batches, 1),
                "train_perplexity": float(math.exp(min(train_loss, 20.0))),
                "train_n_masked": int(masked_sum),
                "val_loss": validation["loss"],
                "val_accuracy": validation["accuracy"],
                "val_perplexity": validation["perplexity"],
                "val_n_masked": validation["n_masked"],
            }
            history.append(row)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            save_mlm_curves(history, out_dir / "plots")
            print(
                f"\nMLM epoch {epoch} ({row['elapsed_seconds']:.0f}s) "
                f"train={row['train_loss']:.4f} val={row['val_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.3f}",
                flush=True,
            )
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                no_improvement = 0
                torch.save(
                    {
                        "format_version": FORMAT_VERSION,
                        "checkpoint_type": "mlm",
                        "model_kind": MODEL_KIND,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "model_config": model_config,
                        "data_profile": profile_name,
                        "profile_stats": profile_stats,
                        "parameter_count": model.count_parameters(),
                        "best_val_loss": best_loss,
                        "validation": validation,
                        "history": history,
                        "run_args": run_args,
                    },
                    out_dir / checkpoint_name,
                )
                print(f"saved {checkpoint_name}: val MLM loss {best_loss:.4f}", flush=True)
            else:
                no_improvement += 1
                if patience and no_improvement >= patience:
                    print(f"early stopping after {patience} MLM epochs without improvement")
                    break
    return best_loss, history
