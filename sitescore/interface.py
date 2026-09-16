"""The one contract every evaluator implements.

Input side:  genome FASTA + annotation GFF (EviAnn output) for training,
             one sequence for scoring.
Output side: a score table, one row per candidate motif, consumed by
             `uniann.sh -s`:  chrom  pos  strand  type  motif  prob
             (pos is 1-based on the + strand; prob in (0, 1]).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SITE_TYPES = ("donor", "acceptor", "start", "stop")
SCORE_COLUMNS = ("chrom", "pos", "strand", "type", "motif", "prob")


@dataclass(frozen=True)
class SiteScore:
    chrom: str
    pos: int  # 1-based, + strand coordinate of the motif's first base
    strand: str  # "+" or "-"
    type: str  # one of SITE_TYPES
    motif: str  # e.g. GT, AG, ATG, TAA
    prob: float  # (0, 1]

    def row(self) -> str:
        return (
            f"{self.chrom}\t{self.pos}\t{self.strand}\t{self.type}\t{self.motif}\t{self.prob:.6g}"
        )


@dataclass
class SiteBlock:
    """Many candidates of one (chrom, strand, type) at once; same content as a
    list of SiteScore, but arrays. Models may yield these from `score_blocks`
    for speed; the pipeline treats both forms identically."""

    chrom: str
    strand: str
    type: str
    pos: np.ndarray  # int64, 1-based + strand coordinates
    motif: np.ndarray  # dtype S2/S3 (bytes) or str
    prob: np.ndarray  # float32/64 in (0, 1]

    def rows(self) -> Iterator[SiteScore]:
        for p, m, q in zip(self.pos.tolist(), self.motif.tolist(), self.prob.tolist(), strict=True):
            m = m.decode() if isinstance(m, bytes) else m
            yield SiteScore(self.chrom, int(p), self.strand, self.type, m, float(q))


class SiteModel(ABC):
    """Implement these four methods to plug a new model (Conv+Mamba, LLM, PWM, ...) in.

    A model lives in a directory (`model_dir`) holding whatever it needs:
    weights, config, tokenizer.  `train` may start from an existing directory
    (`init_dir`) to fine-tune pretrained weights.
    """

    name: str = "base"

    @classmethod
    @abstractmethod
    def train(
        cls, genome: Path, annotation: Path, out_dir: Path, init_dir: Path | None = None, **hparams
    ) -> SiteModel:
        """Train (or fine-tune from `init_dir`) and save into `out_dir`."""

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path) -> SiteModel:
        """Restore a trained model from `model_dir`."""

    @abstractmethod
    def score(self, chrom: str, seq: str, strands: Iterable[str] = ("+",)) -> Iterator[SiteScore]:
        """Yield one SiteScore per candidate motif on the requested strands."""

    def score_blocks(
        self, chrom: str, seq: str, strands: Iterable[str] = ("+",)
    ) -> Iterator[SiteBlock]:
        """Optional fast path: the same candidates as `score`, as SiteBlock arrays.
        Default wraps `score`; override it to avoid per-row Python objects."""
        for s in self.score(chrom, seq, strands):
            yield SiteBlock(
                s.chrom,
                s.strand,
                s.type,
                np.array([s.pos]),
                np.array([s.motif]),
                np.array([s.prob]),
            )


def read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (id, upper-case sequence) per record; no third-party dependency."""
    cid, chunks = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cid is not None:
                    yield cid, "".join(chunks).upper()
                cid, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
    if cid is not None:
        yield cid, "".join(chunks).upper()


def write_scores(scores: Iterable, out) -> int:
    """Write the header + rows for SiteScore and/or SiteBlock items; returns row count."""
    out.write("\t".join(SCORE_COLUMNS) + "\n")
    n = 0
    for s in scores:
        if isinstance(s, SiteBlock):
            if s.pos.size == 0:
                continue
            motif = s.motif.astype("U") if s.motif.dtype.kind == "S" else s.motif.astype("U")
            head = f"{s.chrom}\t"
            mid = f"\t{s.strand}\t{s.type}\t"
            lines = np.char.add(
                np.char.add(np.char.add(head, s.pos.astype("U")), mid),
                np.char.add(
                    np.char.add(motif, "\t"), np.char.mod("%.6g", s.prob.astype(np.float64))
                ),
            )
            out.write("\n".join(lines.tolist()) + "\n")
            n += int(s.pos.size)
        else:
            out.write(s.row() + "\n")
            n += 1
    return n
