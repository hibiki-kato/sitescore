"""CUDA micro-batch probing for V8S2."""

import contextlib
import gc
import math

import numpy as np
import torch

from .mlm import mask_span_mlm, mlm_metrics
from .training import candidate_loss


def configure_cuda_attention():
    """Prefer memory-efficient SDPA kernels when the installed torch exposes them."""
    if not torch.cuda.is_available():
        return
    backend = getattr(torch.backends, "cuda", None)
    if backend is None:
        return
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        fn = getattr(backend, name, None)
        if fn is None:
            continue
        try:
            fn(True)
        except Exception:
            pass


def _autocast_context(device, amp):
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _oom_error(exc):
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message


def _cleanup_cuda(model):
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _candidate_batches(initial_batch, max_batch):
    values = set([1, max(1, int(initial_batch))])
    batch = 1
    while batch < max_batch:
        batch *= 2
        values.add(batch)
    values.add(max_batch)
    return sorted(value for value in values if 1 <= value <= max_batch)


def _take_rows(array, batch_size, dtype=torch.long):
    indices = np.arange(batch_size) % len(array)
    # `array` is either a numpy array / memmap, a SparseLabelStore, or a
    # TrustedRegionStore (the s3/s4/s5 supervised layout keeps both sparse).
    # All three expose .take(axis=0); np.take() would coerce a store into an
    # object array and fail.
    return torch.as_tensor(array.take(indices, axis=0), dtype=dtype)


def _memory_gib():
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return peak, total


def _target_ok(target_fraction):
    peak, total = _memory_gib()
    return peak <= total * target_fraction, peak, total


def _report(prefix, batch_size, status, peak=None, total=None):
    if peak is None:
        print(f"{prefix} batch={batch_size}: {status}", flush=True)
    else:
        print(
            f"{prefix} batch={batch_size}: {status}; "
            f"peak={peak:.1f}/{total:.1f} GiB",
            flush=True,
        )


def autotune_mlm_batch_size(
    model,
    dataset,
    device,
    amp,
    initial_batch,
    max_batch,
    target_fraction,
    mask_fraction,
):
    if device.type != "cuda" or not torch.cuda.is_available():
        return initial_batch
    configure_cuda_attention()
    max_batch = max(int(initial_batch), int(max_batch))
    selected = None
    selected_peak = None
    selected_total = None
    generator = torch.Generator()
    generator.manual_seed(12345)
    model.train()
    print(
        f"auto-batch MLM: probing up to batch {max_batch} "
        f"at target {target_fraction:.0%} GPU memory",
        flush=True,
    )
    for batch_size in _candidate_batches(initial_batch, max_batch):
        try:
            _cleanup_cuda(model)
            torch.cuda.reset_peak_memory_stats()
            sequence = _take_rows(dataset.sequence, batch_size)
            masked, targets = mask_span_mlm(
                sequence,
                mask_fraction=mask_fraction,
                generator=generator,
            )
            masked = masked.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast_context(device, amp):
                logits = model(masked, task="mlm")
                loss, _ = mlm_metrics(logits, targets)
            loss.backward()
            torch.cuda.synchronize()
            ok, peak, total = _target_ok(target_fraction)
            _report("auto-batch MLM", batch_size, "ok" if ok else "above target", peak, total)
            if ok or selected is None:
                selected = batch_size
                selected_peak = peak
                selected_total = total
            if not ok:
                break
        except RuntimeError as exc:
            if _oom_error(exc):
                _report("auto-batch MLM", batch_size, "OOM")
                break
            raise
        finally:
            _cleanup_cuda(model)
    if selected is None:
        selected = 1
    print(
        f"auto-batch MLM selected batch={selected}"
        + (
            f" peak={selected_peak:.1f}/{selected_total:.1f} GiB"
            if selected_peak is not None
            else ""
        ),
        flush=True,
    )
    return int(selected)


def autotune_finetune_batch_size(
    model,
    dataset,
    device,
    amp,
    initial_batch,
    max_batch,
    target_fraction,
):
    if device.type != "cuda" or not torch.cuda.is_available():
        return initial_batch
    configure_cuda_attention()
    max_batch = max(int(initial_batch), int(max_batch))
    selected = None
    selected_peak = None
    selected_total = None
    model.train()
    print(
        f"auto-batch fine-tune: probing up to batch {max_batch} "
        f"at target {target_fraction:.0%} GPU memory",
        flush=True,
    )
    for batch_size in _candidate_batches(initial_batch, max_batch):
        try:
            _cleanup_cuda(model)
            torch.cuda.reset_peak_memory_stats()
            sequence = _take_rows(dataset.sequence, batch_size).to(device, non_blocking=True)
            splice = _take_rows(dataset.splice, batch_size).to(device, non_blocking=True)
            start_stop = _take_rows(dataset.start_stop, batch_size).to(device, non_blocking=True)
            trusted = _take_rows(dataset.trusted, batch_size, dtype=torch.bool).to(
                device, non_blocking=True
            )
            with _autocast_context(device, amp):
                outputs = model(sequence)
                # The probe only measures memory, so the smoothing constants do
                # not matter -- but the trusted mask does, because it is a real
                # (B, T) tensor the training step will also have to hold.
                loss = candidate_loss(
                    outputs, sequence, splice, start_stop, trusted
                )["total"]
            loss.backward()
            torch.cuda.synchronize()
            ok, peak, total = _target_ok(target_fraction)
            _report(
                "auto-batch fine-tune",
                batch_size,
                "ok" if ok else "above target",
                peak,
                total,
            )
            if ok or selected is None:
                selected = batch_size
                selected_peak = peak
                selected_total = total
            if not ok:
                break
        except RuntimeError as exc:
            if _oom_error(exc):
                _report("auto-batch fine-tune", batch_size, "OOM")
                break
            raise
        finally:
            _cleanup_cuda(model)
    if selected is None:
        selected = 1
    print(
        f"auto-batch fine-tune selected batch={selected}"
        + (
            f" peak={selected_peak:.1f}/{selected_total:.1f} GiB"
            if selected_peak is not None
            else ""
        ),
        flush=True,
    )
    return int(selected)


def scaled_grad_accum(target_effective_batch, batch_size, current_grad_accum):
    if not target_effective_batch:
        return current_grad_accum
    return max(1, int(math.ceil(target_effective_batch / max(batch_size, 1))))
