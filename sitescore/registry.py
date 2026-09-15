"""Model lookup by name.  Built-in models are registered via the
`sitescore.models` entry-point group in pyproject.toml; third-party packages can
add their own without touching this repo."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points

from .interface import SiteModel

# Fallback when the package is on PYTHONPATH but not pip-installed.
BUILTIN = {"convmamba": "sitescore.models.convmamba.adapter:ConvMambaSiteModel"}


def _load(spec: str) -> type[SiteModel]:
    mod, _, attr = spec.partition(":")
    return getattr(import_module(mod), attr)


def list_models() -> dict[str, type[SiteModel]]:
    models = {name: _load(spec) for name, spec in BUILTIN.items()}
    models.update({ep.name: ep.load() for ep in entry_points(group="sitescore.models")})
    return models


def get_model(name: str) -> type[SiteModel]:
    models = list_models()
    if name not in models:
        raise SystemExit(f"unknown model '{name}'; available: {sorted(models)}")
    return models[name]
