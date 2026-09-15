"""SSM (Mamba) evaluator adapter.

Drop the received training/scoring code into this package (e.g. `data.py`,
`model.py`, `train.py`, `score.py`) and wire it through the three methods
below.  Nothing outside this package should import from it directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from ...interface import SiteModel, SiteScore


class SSMModel(SiteModel):
    name = "ssm"

    @classmethod
    def train(cls, genome: Path, annotation: Path, out_dir: Path,
              init_dir: Path | None = None, **hparams) -> "SSMModel":
        raise NotImplementedError("wire received training code here (fine-tune from init_dir if given)")

    @classmethod
    def load(cls, model_dir: Path) -> "SSMModel":
        raise NotImplementedError("load checkpoint from model_dir")

    def score(self, chrom: str, seq: str, strands: Iterable[str] = ("+",)) -> Iterator[SiteScore]:
        raise NotImplementedError("yield SiteScore per candidate motif")
