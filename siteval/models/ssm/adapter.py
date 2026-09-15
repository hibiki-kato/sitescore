"""SSM evaluator: GeneFinderV8S2 (Conv stem + bidirectional Mamba3) from the
v8s5_hsap experiment. `model.py`, `checkpoints.py`, `common.py` are copied
verbatim from that package; this file is the only glue.

STATUS
- load / score: implemented here from the package's own primitives
  (make_windows, find_candidates, frame_to_genomic_1based). The original
  `scripts/score_chrX.py` was not delivered, so verify against it once
  available (notably Platt calibration, which the original driver applied
  after raw scoring).
- train: blocked; needs the package's data.py / training.py / autobatch.py
  and scripts/{make_train_data,train}.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ...interface import SITE_TYPES, SiteModel, SiteScore
from . import common

MOTIF = {"donor": lambda s, p: s[p:p + 2], "acceptor": lambda s, p: s[p:p + 2],
         "start": lambda s, p: s[p:p + 3], "stop": lambda s, p: s[p:p + 3]}
MISSING_TRAIN = ("hsap_v8s2/data.py", "hsap_v8s2/training.py", "hsap_v8s2/autobatch.py",
                 "scripts/make_train_data.py", "scripts/train.py")


class SSMModel(SiteModel):
    name = "ssm"

    def __init__(self, model, device, batch_size: int = 8):
        self.model, self.device, self.batch_size = model, device, batch_size

    @classmethod
    def train(cls, genome: Path, annotation: Path, out_dir: Path,
              init_dir: Path | None = None, **hparams) -> "SSMModel":
        raise NotImplementedError(
            "SSM training needs files not yet received: " + ", ".join(MISSING_TRAIN))

    @classmethod
    def load(cls, model_dir: Path, device: str | None = None) -> "SSMModel":
        import torch
        from .checkpoints import load_checkpoint
        cfg = json.loads((model_dir / "siteval.json").read_text())
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, _ = load_checkpoint(model_dir / cfg.get("checkpoint", "v8s2_best.pt"), device)
        return cls(model, device, batch_size=cfg.get("hparams", {}).get("score_batch", 8))

    def _site_probs(self, encoded: np.ndarray) -> np.ndarray:
        """(L, 4) probability of each site type at every base, center-tiled."""
        import torch
        plan = common.make_windows(len(encoded))
        out = np.zeros((len(encoded), len(SITE_TYPES)), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(plan), self.batch_size):
                chunk = plan[i:i + self.batch_size]
                width = max(w["actual_len"] for w in chunk)
                ids = np.full((len(chunk), width), common.N, dtype=np.int64)
                for j, w in enumerate(chunk):
                    ids[j, :w["actual_len"]] = encoded[w["win_start"]:w["win_end"]]
                logits = self.model(torch.from_numpy(ids).to(self.device), task="sites")
                probs = torch.softmax(logits.float(), dim=-1)[..., 1].cpu().numpy()  # (B, L, 4)
                for j, w in enumerate(chunk):
                    lo, hi = w["pred_lo"], w["pred_hi"]
                    out[w["win_start"] + lo:w["win_start"] + hi] = probs[j, lo:hi]
        return out

    def score(self, chrom: str, seq: str, strands: Iterable[str] = ("+",)) -> Iterator[SiteScore]:
        L = len(seq)
        for strand in strands:
            s = seq if strand == "+" else common.reverse_complement(seq)
            enc = common.encode_sequence(s)
            probs = self._site_probs(enc)
            cands = common.find_candidates(enc)
            for k, kind in enumerate(SITE_TYPES):
                for p in cands[kind]:
                    yield SiteScore(chrom, common.frame_to_genomic_1based(strand, p, L),
                                    strand, kind, MOTIF[kind](s, p), float(probs[p, k]))
