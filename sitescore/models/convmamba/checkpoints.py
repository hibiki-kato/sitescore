"""Versioned checkpoint loading for the Conv+Mamba site model."""

import torch

from .model import ConvMambaNet, ConvMambaConfig


FORMAT_VERSION = 82
MODEL_KIND = "convmamba"
LEGACY_KINDS = {"v8s2"}   # checkpoints written before the rename


def _torch_load(path, device):
    return torch.load(path, map_location=device)


def load_checkpoint(path, device):
    checkpoint = _torch_load(path, device)
    if (
        checkpoint.get("format_version") != FORMAT_VERSION
        or checkpoint.get("checkpoint_type") != "fine_tune"
        or checkpoint.get("model_kind") not in {MODEL_KIND, *LEGACY_KINDS}
    ):
        raise ValueError(f"{path} is not a convmamba fine-tune checkpoint")
    model = ConvMambaNet(ConvMambaConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def load_mlm_weights(model, path, device):
    checkpoint = _torch_load(path, device)
    if (
        checkpoint.get("format_version") != FORMAT_VERSION
        or checkpoint.get("checkpoint_type") != "mlm"
        or checkpoint.get("model_kind") not in {MODEL_KIND, *LEGACY_KINDS}
    ):
        raise ValueError(f"{path} is not a convmamba MLM checkpoint")
    state = checkpoint["model_state"]
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current
        and current[key].shape == value.shape
        and not key.startswith("heads.")
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return checkpoint, {
        "loaded": sorted(compatible),
        "missing": missing,
        "unexpected": unexpected,
    }


def load_finetune_checkpoint(path, device):
    checkpoint = _torch_load(path, device)
    if (
        checkpoint.get("format_version") != FORMAT_VERSION
        or checkpoint.get("checkpoint_type") != "fine_tune"
        or checkpoint.get("model_kind") not in {MODEL_KIND, *LEGACY_KINDS}
    ):
        raise ValueError(f"{path} is not a convmamba fine-tune checkpoint")
    return checkpoint
