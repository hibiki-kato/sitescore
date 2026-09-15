"""Model lookup by name.  Built-in models are registered via the
`siteval.models` entry-point group in pyproject.toml; third-party packages can
add their own without touching this repo."""
from __future__ import annotations

from importlib.metadata import entry_points

from .interface import SiteModel


def list_models() -> dict[str, type[SiteModel]]:
    return {ep.name: ep.load() for ep in entry_points(group="siteval.models")}


def get_model(name: str) -> type[SiteModel]:
    models = list_models()
    if name not in models:
        raise SystemExit(f"unknown model '{name}'; available: {sorted(models)}")
    return models[name]
