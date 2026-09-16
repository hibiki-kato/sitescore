"""Per-type Platt scaling of evaluator probabilities against EviAnn truth.
Adapted from calibrate_multi.py by Chirag Adwani (https://github.com/divide-by-zer0).

Pooled fit over
validation sequences, truth = annotated sites matched by exact genomic position.
Model-agnostic: any `SiteModel` is scored through its `score()` and the fitted
(a, b) per site type land in `model_dir/calibration.json`
`sitescore score`
applies them unless `--raw` is given.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import platt
from .interface import SITE_TYPES, SiteBlock, SiteModel, read_fasta

CALIBRATION = "calibration.json"


def annotation_truth(gff: Path, chrom: str, seq: str, require_canonical: bool = True):
    """{(strand, type): set of 1-based + strand positions} from an EviAnn GFF."""
    from .models.convmamba import common  # pure GFF/sequence helpers, no model dependency

    fasta = {chrom: seq}
    transcripts, cds, _ = common.parse_gff3(str(gff), {chrom})
    ranges = common.compute_cds_ranges(cds)
    splice, _ = common.extract_cds_internal_splice_sites(
        transcripts, ranges, fasta, require_canonical
    )
    starts, stops, _ = common.extract_start_stop_sites(cds, fasta)
    return common.true_sites_genomic(splice, starts, stops, chrom, len(seq))


def collect(scores: Iterable, truth) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per type: (prob array, truth flag array) over every candidate (SiteScore or SiteBlock)."""
    pos, prob = defaultdict(list), defaultdict(list)
    for s in scores:
        if isinstance(s, SiteBlock):
            pos[(s.strand, s.type)].append(np.asarray(s.pos, dtype=np.int64))
            prob[(s.strand, s.type)].append(np.asarray(s.prob, dtype=np.float64))
        else:
            pos[(s.strand, s.type)].append(np.array([s.pos], dtype=np.int64))
            prob[(s.strand, s.type)].append(np.array([s.prob], dtype=np.float64))
    out = {}
    for t in SITE_TYPES:
        p_parts, y_parts = [], []
        for strand in "+-":
            k = (strand, t)
            if not pos.get(k):
                continue
            p = np.concatenate(pos[k])
            tp = truth.get(k, set())
            y = np.isin(p, np.fromiter(tp, dtype=np.int64)) if tp else np.zeros(p.size, bool)
            p_parts.append(np.concatenate(prob[k]))
            y_parts.append(y)
        out[t] = (
            np.concatenate(p_parts) if p_parts else np.empty(0),
            np.concatenate(y_parts) if y_parts else np.empty(0, bool),
        )
    return out


def fit(
    model: SiteModel, genome: Path, annotation: Path, chroms: list[str], n_bins: int = 15
) -> dict:
    pooled = {t: ([], []) for t in SITE_TYPES}
    for cid, seq in read_fasta(genome):
        if cid not in chroms:
            continue
        per_type = collect(
            model.score(cid, seq, ("+", "-")), annotation_truth(annotation, cid, seq)
        )
        for t, (p, y) in per_type.items():
            pooled[t][0].append(p)
            pooled[t][1].append(y)
    params = {}
    for t, (ps, ys) in pooled.items():
        p = np.concatenate(ps) if ps else np.empty(0)
        y = np.concatenate(ys) if ys else np.empty(0, bool)
        a, b, info = platt.fit_platt(platt.prob_to_logit(p), y)
        params[t] = {
            "a": a,
            "b": b,
            **info,
            "ece_before": platt.expected_calibration_error(p, y, n_bins),
            "ece_after": platt.expected_calibration_error(platt.apply_platt(p, a, b), y, n_bins),
            "reliability_after": platt.reliability_table(platt.apply_platt(p, a, b), y, n_bins),
        }
    return {
        "calibration": "per-type Platt scaling, Bayes/Laplace targets (Platt 1999)",
        "truth_source": str(annotation),
        "fit_chroms": chroms,
        "params": params,
    }


def load(model_dir: Path) -> dict[str, tuple[float, float]] | None:
    f = Path(model_dir) / CALIBRATION
    if not f.exists():
        return None
    return {t: (v["a"], v["b"]) for t, v in json.loads(f.read_text())["params"].items()}


def apply(scores: Iterable, ab: dict[str, tuple[float, float]]):
    """Apply per-type Platt maps to SiteScore or SiteBlock items."""
    from dataclasses import replace

    for s in scores:
        a, b = ab[s.type]
        if isinstance(s, SiteBlock):
            yield replace(s, prob=platt.apply_platt(np.asarray(s.prob, dtype=np.float64), a, b))
        else:
            yield replace(s, prob=float(platt.apply_platt(s.prob, a, b)))
