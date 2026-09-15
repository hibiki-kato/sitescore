"""The one contract every evaluator implements.

Input side:  genome FASTA + annotation GFF (EviAnn output) for training,
             one sequence for scoring.
Output side: a score table, one row per candidate motif, consumed by
             `uniann.sh -s`:  chrom  pos  strand  type  motif  prob
             (pos is 1-based on the + strand; prob in (0, 1]).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

SITE_TYPES = ("donor", "acceptor", "start", "stop")
SCORE_COLUMNS = ("chrom", "pos", "strand", "type", "motif", "prob")


@dataclass(frozen=True)
class SiteScore:
    chrom: str
    pos: int          # 1-based, + strand coordinate of the motif's first base
    strand: str       # "+" or "-"
    type: str         # one of SITE_TYPES
    motif: str        # e.g. GT, AG, ATG, TAA
    prob: float       # (0, 1]

    def row(self) -> str:
        return f"{self.chrom}\t{self.pos}\t{self.strand}\t{self.type}\t{self.motif}\t{self.prob:.6g}"


class SiteModel(ABC):
    """Implement these four methods to plug a new model (SSM, LLM, PWM, ...) in.

    A model lives in a directory (`model_dir`) holding whatever it needs:
    weights, config, tokenizer.  `train` may start from an existing directory
    (`init_dir`) to fine-tune pretrained weights.
    """

    name: str = "base"

    @classmethod
    @abstractmethod
    def train(cls, genome: Path, annotation: Path, out_dir: Path,
              init_dir: Path | None = None, **hparams) -> "SiteModel":
        """Train (or fine-tune from `init_dir`) and save into `out_dir`."""

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path) -> "SiteModel":
        """Restore a trained model from `model_dir`."""

    @abstractmethod
    def score(self, chrom: str, seq: str, strands: Iterable[str] = ("+",)) -> Iterator[SiteScore]:
        """Yield one SiteScore per candidate motif on the requested strands."""


def write_scores(scores: Iterable[SiteScore], out) -> int:
    """Write the header + rows to a text handle; returns row count."""
    out.write("\t".join(SCORE_COLUMNS) + "\n")
    n = 0
    for s in scores:
        out.write(s.row() + "\n")
        n += 1
    return n
