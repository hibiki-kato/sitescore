"""Fine-tuning loop, candidate-masked loss with asymmetric label smoothing,
validation metrics and checkpointing for the convmamba site model."""

import contextlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

_CACHE_DIR = Path(tempfile.gettempdir()) / "sitescore-mpl-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from . import common
from .checkpoints import FORMAT_VERSION, MODEL_KIND


SITE_NAMES = ("donor", "acceptor", "start", "stop")


def candidate_masks(sequence):
    batch, length = sequence.shape
    masks = {
        name: torch.zeros(
            (batch, length), dtype=torch.bool, device=sequence.device
        )
        for name in SITE_NAMES
    }
    masks["donor"][:, :-1] = (
        (sequence[:, :-1] == common.G) & (sequence[:, 1:] == common.T)
    )
    masks["acceptor"][:, :-1] = (
        (sequence[:, :-1] == common.A) & (sequence[:, 1:] == common.G)
    )
    first, second, third = (
        sequence[:, :-2],
        sequence[:, 1:-1],
        sequence[:, 2:],
    )
    masks["start"][:, :-2] = (
        (first == common.A)
        & (second == common.T)
        & (third == common.G)
    )
    masks["stop"][:, :-2] = (
        ((first == common.T) & (second == common.A) & (third == common.A))
        | ((first == common.T) & (second == common.A) & (third == common.G))
        | ((first == common.T) & (second == common.G) & (third == common.A))
    )
    return masks


def site_targets(splice_labels, start_stop_labels):
    return {
        "donor": splice_labels == 1,
        "acceptor": splice_labels == 2,
        "start": start_stop_labels == 1,
        "stop": start_stop_labels == 2,
    }


def v7_candidate_loss(logits, sequence, splice_labels, start_stop_labels):
    if logits.ndim != 4 or logits.shape[2:] != (4, 2):
        raise ValueError(f"expected site logits (B,T,4,2), got {tuple(logits.shape)}")
    motifs = candidate_masks(sequence)
    targets = site_targets(splice_labels, start_stop_labels)
    losses = {}
    valid_losses = []
    for index, name in enumerate(SITE_NAMES):
        mask = motifs[name] | targets[name]
        if mask.any():
            loss = F.cross_entropy(
                logits[:, :, index, :][mask], targets[name][mask].long()
            )
            valid_losses.append(loss)
        else:
            loss = logits[:, :, index, :].sum() * 0.0
        losses[name] = loss
    total = torch.stack(valid_losses).mean() if valid_losses else logits.sum() * 0.0
    losses["total"] = total
    return losses



DEFAULT_ALPHA = {name: 0.0 for name in SITE_NAMES}


def binary_entropy(rate):
    """``H(rate)`` in nats -- the loss a perfectly calibrated model still pays.

    A soft target of ``alpha`` cannot be fit to zero loss: the best achievable
    per-candidate loss is ``H(alpha)``. Reporting that floor alongside the loss
    keeps the numbers comparable with a hard-label (alpha = 0) run.
    """
    if rate <= 0.0 or rate >= 1.0:
        return 0.0
    return -(rate * math.log(rate) + (1.0 - rate) * math.log1p(-rate))


def smoothed_candidate_loss(
    logits,
    sequence,
    splice_labels,
    start_stop_labels,
    trusted=None,
    alpha=None,
    soft_weight=1.0,
):
    """Asymmetric soft-target cross-entropy over candidate positions.

    Per candidate ``j`` of site type ``t`` the target is

        y = 1        EviAnn labelled this position as a site
        y = 0        the position lies inside an EviAnn CDS locus on its own
                     strand, so the annotation is trustworthy and an
                     unlabelled candidate really is a negative
        y = alpha_t  otherwise -- the annotation is silent here, so the
                     candidate is unlabelled rather than negative

    and the loss is the binary cross-entropy against that target,
    ``-[y log p + (1 - y) log(1 - p)]``, averaged over the candidates of a site
    type and then over the four site types: the same reduction as the hard-label loss,
    with only the target changed.

    Why this loss:

    * It is a **proper scoring rule** -- uniquely minimised at ``p = y`` -- so
      the third rule tells the model "the base rate here is alpha_t" instead of
      "this is definitely not a site". That puts a floor of roughly alpha_t
      under the background probabilities rather than letting them collapse,
      which is the specific failure mode s3 hits downstream (UniAnn takes logs
      of these probabilities and has a finite log floor).
    * It is the **exact expected hard-label loss** under the generative story
      that each unannotated candidate is a real site with probability alpha_t,
      independently. So it is the Bayes-consistent objective for a known class
      prior on the unlabelled set, not an ad-hoc regulariser.
    * With ``trusted=None``, ``alpha=None`` or an all-zero ``alpha`` it reduces
      to ``v7_candidate_loss`` exactly, which makes ``--alpha-mode zero`` a
      clean ablation against the hard-label loss (alpha = 0).

    ``soft_weight`` scales the contribution of the smoothed candidates only
    (1.0 = untouched); it is a separate knob from ``alpha`` because it changes
    how loudly the smoothed set speaks, not what it says.

    Returns the usual per-site and ``total`` entries plus two diagnostics:
    ``floor`` (the mean ``H(alpha_t)`` the data makes unavoidable) and
    ``excess`` (``total - floor``). Both are detached. ``floor`` depends only
    on the data, never on the model, so on a fixed validation set it is
    constant across epochs -- selecting a checkpoint on ``val_total_loss`` and
    on excess loss pick the same epoch.
    """
    if logits.ndim != 4 or logits.shape[2:] != (4, 2):
        raise ValueError(f"expected site logits (B,T,4,2), got {tuple(logits.shape)}")
    motifs = candidate_masks(sequence)
    targets = site_targets(splice_labels, start_stop_labels)
    untrusted = None if trusted is None else ~trusted.bool()
    losses = {}
    valid_losses = []
    valid_floors = []
    for index, name in enumerate(SITE_NAMES):
        positive = targets[name]
        mask = motifs[name] | positive
        if not bool(mask.any()):
            losses[name] = logits[:, :, index, :].sum() * 0.0
            continue
        rate = float(alpha.get(name, 0.0)) if alpha else 0.0
        # float32 regardless of autocast: alpha is O(1e-5) here and bf16 has
        # roughly three significant decimal digits, so the (1 - y) term would
        # be quantised away.
        log_probability = F.log_softmax(
            logits[:, :, index, :][mask].float(), dim=-1
        )
        is_positive = positive[mask]
        if untrusted is None or rate <= 0.0:
            is_soft = torch.zeros_like(is_positive)
        else:
            is_soft = untrusted[mask] & ~is_positive
        one = torch.ones((), dtype=log_probability.dtype, device=logits.device)
        zero = torch.zeros((), dtype=log_probability.dtype, device=logits.device)
        smoothed = torch.full(
            (), rate, dtype=log_probability.dtype, device=logits.device
        )
        y = torch.where(is_positive, one, torch.where(is_soft, smoothed, zero))
        per_candidate = -(
            y * log_probability[:, 1] + (1.0 - y) * log_probability[:, 0]
        )
        if soft_weight != 1.0:
            weight = torch.where(
                is_soft,
                torch.full(
                    (), float(soft_weight),
                    dtype=log_probability.dtype, device=logits.device,
                ),
                one,
            )
            denominator = weight.sum().clamp_min(1e-12)
            loss = (per_candidate * weight).sum() / denominator
            soft_fraction = (
                is_soft.to(log_probability.dtype) * weight
            ).sum() / denominator
        else:
            loss = per_candidate.mean()
            soft_fraction = is_soft.to(log_probability.dtype).mean()
        losses[name] = loss
        valid_losses.append(loss)
        valid_floors.append(soft_fraction.detach() * binary_entropy(rate))
    total = torch.stack(valid_losses).mean() if valid_losses else logits.sum() * 0.0
    losses["total"] = total
    floor = (
        torch.stack(valid_floors).mean()
        if valid_floors
        else torch.zeros((), dtype=total.dtype, device=logits.device)
    )
    losses["floor"] = floor.detach()
    losses["excess"] = (total.detach() - floor).detach()
    return losses


candidate_loss = smoothed_candidate_loss


def v6_candidate_loss(outputs, sequence, splice_labels, start_stop_labels):
    splice_logits, start_stop_logits = outputs
    motifs = candidate_masks(sequence)
    splice_mask = (
        motifs["donor"]
        | motifs["acceptor"]
        | (splice_labels == 1)
        | (splice_labels == 2)
    )
    start_stop_mask = (
        motifs["start"]
        | motifs["stop"]
        | (start_stop_labels == 1)
        | (start_stop_labels == 2)
    )
    splice = F.cross_entropy(
        splice_logits[splice_mask], splice_labels[splice_mask]
    )
    start_stop = F.cross_entropy(
        start_stop_logits[start_stop_mask], start_stop_labels[start_stop_mask]
    )
    return {
        "splice": splice,
        "start_stop": start_stop,
        "total": splice + start_stop,
    }


def output_site_probabilities(outputs, model_kind=None):
    """(B, L, 4, 2) logits -> per-type P(site) tensors."""
    probabilities = torch.softmax(outputs.float(), dim=-1)[..., 1]
    return {name: probabilities[:, :, index] for index, name in enumerate(SITE_NAMES)}


def threshold_metrics(scores, truth, thresholds):
    best = {
        "threshold": float(thresholds[0]),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    truth = truth.astype(bool, copy=False)
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.count_nonzero(predicted & truth))
        fp = int(np.count_nonzero(predicted & ~truth))
        fn = int(np.count_nonzero(~predicted & truth))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return best


def topk_accuracy(scores, truth):
    k = int(np.count_nonzero(truth))
    if k == 0 or len(scores) == 0:
        return 0.0
    indices = np.argpartition(scores, -k)[-k:]
    return float(np.count_nonzero(truth[indices]) / k)


def summarize_site_scores(scores, truth):
    coarse = threshold_metrics(scores, truth, np.arange(0.05, 1.0, 0.05))
    fine = threshold_metrics(scores, truth, np.linspace(0.0, 1.0, 201))
    ap = float(average_precision_score(truth, scores)) if truth.any() else 0.0
    return {
        "coarse_f1": coarse["f1"],
        "coarse_threshold": coarse["threshold"],
        "coarse_precision": coarse["precision"],
        "coarse_recall": coarse["recall"],
        "fine_f1": fine["f1"],
        "fine_threshold": fine["threshold"],
        "fine_precision": fine["precision"],
        "fine_recall": fine["recall"],
        "average_precision": ap,
        "topk": topk_accuracy(scores, truth),
        "n_candidates": int(len(scores)),
        "n_positive": int(np.count_nonzero(truth)),
    }


def autocast_context(device, amp):
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(
    model, loader, device, model_kind, amp="fp32", alpha=None, soft_weight=1.0
):
    model.eval()
    score_parts = {name: [] for name in SITE_NAMES}
    truth_parts = {name: [] for name in SITE_NAMES}
    loss_totals = {}
    batches = 0
    for sequence, splice_labels, start_stop_labels, trusted in loader:
        sequence = sequence.to(device, non_blocking=True)
        splice_labels = splice_labels.to(device, non_blocking=True)
        start_stop_labels = start_stop_labels.to(device, non_blocking=True)
        trusted = trusted.to(device, non_blocking=True)
        with autocast_context(device, amp):
            outputs = model(sequence)
            losses = (
                smoothed_candidate_loss(
                    outputs,
                    sequence,
                    splice_labels,
                    start_stop_labels,
                    trusted,
                    alpha,
                    soft_weight,
                )
                if True
                else v6_candidate_loss(
                    outputs, sequence, splice_labels, start_stop_labels
                )
            )
        for name, loss in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(loss.item())
        probabilities = output_site_probabilities(outputs, model_kind)
        motifs = candidate_masks(sequence)
        targets = site_targets(splice_labels, start_stop_labels)
        for name in SITE_NAMES:
            mask = motifs[name]
            if mask.any():
                score_parts[name].append(
                    probabilities[name][mask].detach().cpu().numpy()
                )
                truth_parts[name].append(
                    targets[name][mask].detach().cpu().numpy()
                )
        batches += 1

    metrics = {
        "losses": {
            name: value / max(batches, 1) for name, value in loss_totals.items()
        },
        "sites": {},
    }
    for name in SITE_NAMES:
        scores = (
            np.concatenate(score_parts[name])
            if score_parts[name]
            else np.empty(0, dtype=np.float32)
        )
        truth = (
            np.concatenate(truth_parts[name]).astype(bool)
            if truth_parts[name]
            else np.empty(0, dtype=bool)
        )
        metrics["sites"][name] = summarize_site_scores(scores, truth)
    metrics["selection_score"] = float(
        np.mean(
            [metrics["sites"][name]["coarse_f1"] for name in SITE_NAMES]
        )
    )
    metrics["fine_mean_f1"] = float(
        np.mean([metrics["sites"][name]["fine_f1"] for name in SITE_NAMES])
    )
    metrics["mean_average_precision"] = float(
        np.mean(
            [metrics["sites"][name]["average_precision"] for name in SITE_NAMES]
        )
    )
    metrics["mean_splice_f1"] = float(
        np.mean([metrics["sites"][name]["coarse_f1"] for name in ("donor", "acceptor")])
    )
    metrics["mean_start_stop_f1"] = float(
        np.mean([metrics["sites"][name]["coarse_f1"] for name in ("start", "stop")])
    )
    return metrics


def _split_decay(named_parameters):
    decay, no_decay = [], []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "norm" in lowered
            or "embedding" in lowered
            or "position" in lowered
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


def build_finetune_optimizer(model, encoder_lr, head_lr, weight_decay):
    encoder_params, head_params = [], []
    for name, parameter in model.named_parameters():
        if name.startswith("heads."):
            head_params.append((name, parameter))
        else:
            encoder_params.append((name, parameter))
    encoder_decay, encoder_no_decay = _split_decay(encoder_params)
    head_decay, head_no_decay = _split_decay(head_params)
    return torch.optim.AdamW(
        [
            {"params": encoder_decay, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )


def make_lr_schedule(base_lr, total_steps, warmup_steps, min_lr_ratio=0.1):
    warmup_steps = min(max(1, warmup_steps), max(1, total_steps))
    min_lr = base_lr * min_lr_ratio

    def factor_at(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr / base_lr + 0.5 * (1.0 - min_lr / base_lr) * (
            1 + math.cos(math.pi * progress)
        )

    return factor_at


def flatten_metrics(epoch, train_losses, validation, elapsed, lrs):
    row = {
        "epoch": epoch,
        "elapsed_seconds": elapsed,
        "encoder_learning_rate": lrs[0],
        "head_learning_rate": lrs[-1],
        "selection_score": validation["selection_score"],
        "fine_mean_f1": validation["fine_mean_f1"],
        "mean_average_precision": validation["mean_average_precision"],
        "mean_splice_f1": validation["mean_splice_f1"],
        "mean_start_stop_f1": validation["mean_start_stop_f1"],
    }
    for name, value in train_losses.items():
        row[f"train_{name}_loss"] = value
    for name, value in validation["losses"].items():
        row[f"val_{name}_loss"] = value
    for site, values in validation["sites"].items():
        for key, value in values.items():
            row[f"{site}_{key}"] = value
    return row


def save_training_curves(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    loss_panels = [
        ("total", "Total loss"),
        ("donor", "Donor loss"),
        ("acceptor", "Acceptor loss"),
        ("start", "Start loss"),
        ("stop", "Stop loss"),
    ]
    for axis, (suffix, title) in zip(axes.ravel(), loss_panels):
        axis.plot(
            epochs,
            [row[f"train_{suffix}_loss"] for row in history],
            "o-",
            label="train",
        )
        axis.plot(
            epochs,
            [row[f"val_{suffix}_loss"] for row in history],
            "o-",
            label="val",
        )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    axes.ravel()[-1].axis("off")
    figure.tight_layout()
    figure.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for metric, title, axis in (
        ("coarse_f1", "Candidate F1", axes[0]),
        ("average_precision", "PR-AUC", axes[1]),
        ("topk", "Top-k accuracy", axes[2]),
    ):
        for site in SITE_NAMES:
            axis.plot(
                epochs,
                [row[f"{site}_{metric}"] for row in history],
                "o-",
                label=site,
            )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "site_metric_curves.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        epochs,
        [row["mean_splice_f1"] for row in history],
        "o-",
        label="mean splice",
    )
    axis.plot(
        epochs,
        [row["mean_start_stop_f1"] for row in history],
        "o-",
        label="mean start/stop",
    )
    axis.plot(
        epochs,
        [row["selection_score"] for row in history],
        "o-",
        label="mean all",
    )
    axis.set(xlabel="epoch", ylabel="coarse candidate F1", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "mean_f1_curves.png", dpi=150)
    plt.close(figure)


SELECTION_METRICS = ("val_loss", "mean_f1")


def resolve_selection_metric(run_args):
    """Which validation quantity drives best-checkpoint choice and early stopping.

    Defaults to ``val_loss``. Mean candidate F1 is a *ranking* metric and is
    therefore invariant to logit scale, so it cannot see a model saturating its
    probabilities. With empty windows the vast majority of training windows are
    pure-negative, which pushes logits down relentlessly; selecting on F1 let
    training continue tens of epochs past the validation-loss minimum until the
    emitted probabilities underflowed the downstream decoder's log floor.
    Validation loss is a proper scoring rule and does penalise that drift.
    """
    metric = str(run_args.get("selection_metric", "val_loss"))
    if metric not in SELECTION_METRICS:
        raise ValueError(
            f"unknown selection_metric {metric!r}; choose from {SELECTION_METRICS}"
        )
    return metric


def selection_value(source, metric):
    """Score to maximise. Accepts a validation dict or a history row."""
    if metric == "mean_f1":
        return float(source["selection_score"])
    if "losses" in source:
        return -float(source["losses"]["total"])
    return -float(source["val_total_loss"])


def train_model(
    model,
    model_kind,
    run_kind,
    train_loader,
    val_loader,
    device,
    out_dir,
    checkpoint_name,
    profile_name,
    profile_stats,
    model_config,
    epochs,
    encoder_lr,
    head_lr,
    weight_decay,
    grad_accum,
    warmup_steps,
    max_grad_norm,
    patience,
    amp,
    run_args,
    resume_checkpoint=None,
    alpha=None,
    soft_weight=1.0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = build_finetune_optimizer(model, encoder_lr, head_lr, weight_decay)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max(1, epochs * optimizer_steps_per_epoch)
    factor_at = make_lr_schedule(1.0, total_steps, warmup_steps)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    metrics_path = out_dir / "metrics.jsonl"
    selection_metric = resolve_selection_metric(run_args)
    print(f"checkpoint selection / early stopping on: {selection_metric}", flush=True)
    alpha = dict(alpha) if alpha else dict(DEFAULT_ALPHA)
    floor_now = sum(binary_entropy(alpha.get(name, 0.0)) for name in SITE_NAMES) / len(
        SITE_NAMES
    )
    print(
        "label smoothing alpha: "
        + ", ".join(f"{name}={alpha.get(name, 0.0):.3e}" for name in SITE_NAMES)
        + f" (soft_weight={soft_weight:g}; per-candidate entropy floor "
        f"<= {floor_now:.5f})",
        flush=True,
    )
    history = []
    best_score = -float("inf")
    no_improvement = 0
    optimizer_step = 0
    log_every = max(1, int(run_args.get("log_every", 25)))
    start_epoch = 1
    metrics_mode = "w"
    write_resume_history = False
    if resume_checkpoint is not None:
        history = list(resume_checkpoint.get("history", []))
        start_epoch = int(resume_checkpoint.get("epoch", len(history))) + 1
        # Only reuse the stored best score when it was produced by the same
        # metric; otherwise recompute it from history so a changed criterion
        # cannot be compared against an incompatible baseline.
        if resume_checkpoint.get("selection_metric") == selection_metric:
            best_score = float(resume_checkpoint["best_selection_score"])
        else:
            best_score = max(
                (selection_value(row, selection_metric) for row in history),
                default=-float("inf"),
            )
        optimizer_state = resume_checkpoint.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            for base_lr, group in zip(base_lrs, optimizer.param_groups):
                group["lr"] = base_lr
        optimizer_step = int(
            resume_checkpoint.get(
                "optimizer_step",
                max(0, start_epoch - 1) * optimizer_steps_per_epoch,
            )
        )
        if metrics_path.exists():
            metrics_mode = "a"
        else:
            write_resume_history = True
        print(
            f"resuming fine-tune from epoch {start_epoch - 1}; "
            f"target epoch {epochs}; best mean F1 so far {best_score:.4f}",
            flush=True,
        )
        if start_epoch > epochs:
            print(
                f"resume checkpoint is already at epoch {start_epoch - 1}; "
                f"nothing to train for target epoch {epochs}",
                flush=True,
            )
            return best_score, history


    def record_epoch(epoch, train_losses, elapsed):
        """Validate, append to history/metrics.jsonl, refresh plots, checkpoint if best.
        Returns True when the selection metric improved. Epoch 0 = the initial
        weights before any update (a candidate for best like any other epoch)."""
        nonlocal best_score, no_improvement
        validation = evaluate(
            model, val_loader, device, model_kind=model_kind, amp=amp, alpha=alpha, soft_weight=soft_weight
        )
        row = flatten_metrics(
            epoch, train_losses, validation, elapsed, [group["lr"] for group in optimizer.param_groups]
        )
        history.append(row)
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        save_training_curves(history, out_dir / "plots")
        print_epoch_summary(row)
        score = selection_value(validation, selection_metric)
        if score <= best_score:
            return False
        best_score = score
        no_improvement = 0
        checkpoint_data = {
            "format_version": FORMAT_VERSION,
            "checkpoint_type": "fine_tune",
            "model_kind": MODEL_KIND,
            "run_kind": run_kind,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model_config,
            "data_profile": profile_name,
            "profile_stats": profile_stats,
            "parameter_count": model.count_parameters(),
            "selection_metric": selection_metric,
            "label_strategy": "asymmetric_smoothing",
            "alpha": dict(alpha),
            "soft_weight": float(soft_weight),
            "best_selection_score": best_score,
            "best_val_total_loss": float(validation["losses"]["total"]),
            "best_mean_coarse_f1": float(validation["selection_score"]),
            "optimizer_step": optimizer_step,
            "best_thresholds": {name: validation["sites"][name]["coarse_threshold"] for name in SITE_NAMES},
            "validation": validation,
            "history": history,
            "run_args": run_args,
        }
        torch.save(checkpoint_data, out_dir / checkpoint_name)
        print(
            f"saved {checkpoint_name}: {selection_metric}="
            f"{validation['losses']['total'] if selection_metric == 'val_loss' else validation['selection_score']:.5f} "
            f"(val loss {validation['losses']['total']:.5f}, mean coarse F1 {validation['selection_score']:.4f})",
            flush=True,
        )
        return True

    with open(metrics_path, metrics_mode) as metrics_file:
        if write_resume_history:
            for row in history:
                metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
        if start_epoch == 1:
            # epoch 0: the starting weights (pretrained init or random) on the validation split
            t0 = time.time()
            nan_losses = {name: float("nan") for name in ("donor", "acceptor", "start", "stop", "total", "floor", "excess")}
            record_epoch(0, nan_losses, time.time() - t0)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            start_time = time.time()
            loss_sums = {}
            batches = accumulated = 0
            optimizer.zero_grad(set_to_none=True)
            print(
                f"FT epoch {epoch}: {len(train_loader):,} batches; "
                f"logging every {log_every} batches",
                flush=True,
            )
            for batch_index, (
                sequence,
                splice_labels,
                start_stop_labels,
                trusted,
            ) in enumerate(train_loader, start=1):
                sequence = sequence.to(device, non_blocking=True)
                splice_labels = splice_labels.to(device, non_blocking=True)
                start_stop_labels = start_stop_labels.to(
                    device, non_blocking=True
                )
                trusted = trusted.to(device, non_blocking=True)
                factor = factor_at(optimizer_step)
                for base_lr, group in zip(base_lrs, optimizer.param_groups):
                    group["lr"] = base_lr * factor
                with autocast_context(device, amp):
                    outputs = model(sequence)
                    losses = (
                        smoothed_candidate_loss(
                            outputs,
                            sequence,
                            splice_labels,
                            start_stop_labels,
                            trusted,
                            alpha,
                            soft_weight,
                        )
                        if True
                        else v6_candidate_loss(
                            outputs,
                            sequence,
                            splice_labels,
                            start_stop_labels,
                        )
                    )
                scaled = losses["total"] / grad_accum
                if not torch.isfinite(scaled):
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    print(
                        f"WARN: skipped non-finite loss at epoch {epoch}, "
                        f"batch {batch_index}",
                        flush=True,
                    )
                    continue
                scaled.backward()
                accumulated += 1
                if accumulated == grad_accum:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    accumulated = 0
                for name, loss in losses.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(
                        loss.item()
                    )
                batches += 1
                if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(train_loader):
                    elapsed = time.time() - start_time
                    print(
                        f"[FT epoch {epoch} {batch_index}/{len(train_loader)}] "
                        f"loss={loss_sums['total'] / max(batches, 1):.5f} "
                        f"enc_lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"head_lr={optimizer.param_groups[-1]['lr']:.2e} "
                        f"{batch_index * train_loader.batch_size / max(elapsed, 1e-9):.1f} samples/s",
                        flush=True,
                    )
            if accumulated:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            train_losses = {
                name: value / max(batches, 1)
                for name, value in loss_sums.items()
            }
            improved = record_epoch(epoch, train_losses, time.time() - start_time)
            if not improved:
                no_improvement += 1
                if patience and no_improvement >= patience:
                    print(f"early stopping after {patience} epochs without improvement")
                    break
    return best_score, history

    with open(metrics_path, metrics_mode) as metrics_file:
        if write_resume_history:
            for row in history:
                metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            start_time = time.time()
            loss_sums = {}
            batches = accumulated = 0
            optimizer.zero_grad(set_to_none=True)
            print(
                f"FT epoch {epoch}: {len(train_loader):,} batches; "
                f"logging every {log_every} batches",
                flush=True,
            )
            for batch_index, (
                sequence,
                splice_labels,
                start_stop_labels,
                trusted,
            ) in enumerate(train_loader, start=1):
                sequence = sequence.to(device, non_blocking=True)
                splice_labels = splice_labels.to(device, non_blocking=True)
                start_stop_labels = start_stop_labels.to(
                    device, non_blocking=True
                )
                trusted = trusted.to(device, non_blocking=True)
                factor = factor_at(optimizer_step)
                for base_lr, group in zip(base_lrs, optimizer.param_groups):
                    group["lr"] = base_lr * factor
                with autocast_context(device, amp):
                    outputs = model(sequence)
                    losses = (
                        smoothed_candidate_loss(
                            outputs,
                            sequence,
                            splice_labels,
                            start_stop_labels,
                            trusted,
                            alpha,
                            soft_weight,
                        )
                        if True
                        else v6_candidate_loss(
                            outputs,
                            sequence,
                            splice_labels,
                            start_stop_labels,
                        )
                    )
                scaled = losses["total"] / grad_accum
                if not torch.isfinite(scaled):
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    print(
                        f"WARN: skipped non-finite loss at epoch {epoch}, "
                        f"batch {batch_index}",
                        flush=True,
                    )
                    continue
                scaled.backward()
                accumulated += 1
                if accumulated == grad_accum:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    accumulated = 0
                for name, loss in losses.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(
                        loss.item()
                    )
                batches += 1
                if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(train_loader):
                    elapsed = time.time() - start_time
                    print(
                        f"[FT epoch {epoch} {batch_index}/{len(train_loader)}] "
                        f"loss={loss_sums['total'] / max(batches, 1):.5f} "
                        f"enc_lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"head_lr={optimizer.param_groups[-1]['lr']:.2e} "
                        f"{batch_index * train_loader.batch_size / max(elapsed, 1e-9):.1f} samples/s",
                        flush=True,
                    )
            if accumulated:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            train_losses = {
                name: value / max(batches, 1)
                for name, value in loss_sums.items()
            }
            validation = evaluate(
                model,
                val_loader,
                device,
                model_kind=model_kind,
                amp=amp,
                alpha=alpha,
                soft_weight=soft_weight,
            )
            elapsed = time.time() - start_time
            row = flatten_metrics(
                epoch,
                train_losses,
                validation,
                elapsed,
                [group["lr"] for group in optimizer.param_groups],
            )
            history.append(row)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            save_training_curves(history, out_dir / "plots")
            print_epoch_summary(row)

            score = selection_value(validation, selection_metric)
            if score > best_score:
                best_score = score
                no_improvement = 0
                checkpoint_data = {
                    "format_version": FORMAT_VERSION,
                    "checkpoint_type": "fine_tune",
                    "model_kind": MODEL_KIND,
                    "run_kind": run_kind,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_config": model_config,
                    "data_profile": profile_name,
                    "profile_stats": profile_stats,
                    "parameter_count": model.count_parameters(),
                    "selection_metric": selection_metric,
                    "label_strategy": "asymmetric_smoothing",
                    "alpha": dict(alpha),
                    "soft_weight": float(soft_weight),
                    "best_selection_score": best_score,
                    "best_val_total_loss": float(validation["losses"]["total"]),
                    "best_mean_coarse_f1": float(validation["selection_score"]),
                    "optimizer_step": optimizer_step,
                    "best_thresholds": {
                        name: validation["sites"][name][
                            "coarse_threshold"
                        ]
                        for name in SITE_NAMES
                    },
                    "validation": validation,
                    "history": history,
                    "run_args": run_args,
                }
                torch.save(checkpoint_data, out_dir / checkpoint_name)
                print(
                    f"saved {checkpoint_name}: {selection_metric}="
                    f"{validation['losses']['total'] if selection_metric == 'val_loss' else validation['selection_score']:.5f} "
                    f"(val loss {validation['losses']['total']:.5f}, "
                    f"mean coarse F1 {validation['selection_score']:.4f})",
                    flush=True,
                )
            else:
                no_improvement += 1
                if patience and no_improvement >= patience:
                    print(
                        f"early stopping after {patience} epochs without improvement"
                    )
                    break
    return best_score, history


def print_epoch_summary(row):
    floor = row.get("val_floor_loss")
    excess = (
        f" (floor {floor:.5f}, excess {row['val_excess_loss']:.5f})"
        if floor is not None
        else ""
    )
    print(
        f"\nEpoch {row['epoch']} ({row['elapsed_seconds']:.0f}s) "
        f"train={row['train_total_loss']:.5f} "
        f"val={row['val_total_loss']:.5f}{excess}"
    )
    for name in SITE_NAMES:
        print(
            f"  {name:<8} coarse F1={row[f'{name}_coarse_f1']:.3f} "
            f"fine F1={row[f'{name}_fine_f1']:.3f} "
            f"AP={row[f'{name}_average_precision']:.3f} "
            f"topk={row[f'{name}_topk']:.3f}"
        )
    print(
        f"  selection mean F1={row['selection_score']:.4f}; "
        f"splice={row['mean_splice_f1']:.4f}; "
        f"start/stop={row['mean_start_stop_f1']:.4f}"
    )
