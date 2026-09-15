"""Versioned checkpoint loading for V8S2."""

import torch

from .model import GeneFinderV8S2, V8S2Config


FORMAT_VERSION = 82
MODEL_KIND = "v8s2"


def _torch_load(path, device):
    return torch.load(path, map_location=device)


def load_checkpoint(path, device):
    checkpoint = _torch_load(path, device)
    if (
        checkpoint.get("format_version") != FORMAT_VERSION
        or checkpoint.get("checkpoint_type") != "fine_tune"
        or checkpoint.get("model_kind") != MODEL_KIND
    ):
        raise ValueError(f"{path} is not a Dmel UniAnn V8S2 checkpoint")
    model = GeneFinderV8S2(V8S2Config.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def load_mlm_weights(model, path, device):
    checkpoint = _torch_load(path, device)
    if (
        checkpoint.get("format_version") != FORMAT_VERSION
        or checkpoint.get("checkpoint_type") != "mlm"
        or checkpoint.get("model_kind") != MODEL_KIND
    ):
        raise ValueError(f"{path} is not a V8S2 MLM checkpoint")
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
        or checkpoint.get("model_kind") != MODEL_KIND
    ):
        raise ValueError(f"{path} is not a V8S2 fine-tune checkpoint")
    return checkpoint
