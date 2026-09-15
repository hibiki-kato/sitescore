"""Download a pretrained model_dir listed in models/registry.yaml and verify its sha256."""

from __future__ import annotations

import hashlib
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "models" / "registry.yaml"


def read_registry(path: Path = REGISTRY) -> dict[str, dict[str, str]]:
    """Tiny YAML subset reader: top-level keys with indented `key: value` pairs."""
    entries, cur = {}, None
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            cur = line.rstrip(":").strip()
            entries[cur] = {}
        elif cur:
            k, _, v = line.strip().partition(":")
            entries[cur][k.strip()] = v.strip()
    return entries


def fetch(name: str, dest: Path) -> Path:
    entry = read_registry().get(name)
    if not entry:
        raise SystemExit(f"unknown model '{name}'; registry has {sorted(read_registry())}")
    target = dest / name
    if (target / "sitescore.json").exists():
        print(f"{target} already present", file=sys.stderr)
        return target
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        print(f"downloading {entry['url']}", file=sys.stderr)
        with urllib.request.urlopen(entry["url"]) as r:
            h = hashlib.sha256()
            while chunk := r.read(1 << 20):
                tmp.write(chunk)
                h.update(chunk)
    if h.hexdigest() != entry["sha256"]:
        Path(tmp.name).unlink()
        raise SystemExit(f"sha256 mismatch for {name}: {h.hexdigest()} != {entry['sha256']}")
    with tarfile.open(tmp.name) as tar:
        tar.extractall(dest)
    Path(tmp.name).unlink()
    if not (target / "sitescore.json").exists():
        raise SystemExit(f"archive did not contain {name}/sitescore.json")
    print(f"verified and unpacked to {target}", file=sys.stderr)
    return target
