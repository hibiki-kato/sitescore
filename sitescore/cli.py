"""sitescore train | calibrate | score | models

  sitescore train     --model convmamba --genome g.fna --annotation eviann.gff --out model_dir [--init base_model_dir]
  sitescore calibrate --model-dir model_dir --genome g.fna --annotation eviann.gff [--chroms a,b]
  sitescore score     --model-dir model_dir --fasta seq.fa [--strands +-] [--raw] > sites.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import calibration
from .interface import read_fasta, write_scores
from .registry import get_model, list_models

CONFIG = "sitescore.json"   # written into every model_dir: {"model": name, "hparams": {...}}


def cmd_train(a):
    cls = get_model(a.model)
    hparams = json.loads(a.hparams) if a.hparams else {}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / CONFIG).write_text(json.dumps({"model": a.model, "hparams": hparams}, indent=2))
    cls.train(Path(a.genome), Path(a.annotation), out,
              init_dir=Path(a.init) if a.init else None, **hparams)
    if not a.no_calibrate:
        _calibrate(out, Path(a.genome), Path(a.annotation), None)


def _load_model(md: Path):
    cfg = json.loads((md / CONFIG).read_text())
    return get_model(cfg["model"]).load(md)


def _calibrate(md: Path, genome: Path, annotation: Path, chroms: list[str] | None):
    """Platt-fit on `chroms` (default: the plug-in's validation sequences)."""
    if chroms is None:
        info = md / "train_info.json"
        chroms = json.loads(info.read_text()).get("val_chroms") if info.exists() else None
    if not chroms:
        print("calibrate: no validation sequences known; pass --chroms", file=sys.stderr)
        return
    report = calibration.fit(_load_model(md), genome, annotation, list(chroms))
    (md / calibration.CALIBRATION).write_text(json.dumps(report, indent=2))
    for t, v in report["params"].items():
        print(f"calibrate {t}: a={v['a']:.3f} b={v['b']:.3f} n+={v['n_pos']} "
              f"ECE {v['ece_before']:.4f} -> {v['ece_after']:.4f}", file=sys.stderr)


def cmd_calibrate(a):
    _calibrate(Path(a.model_dir), Path(a.genome), Path(a.annotation),
               a.chroms.split(",") if a.chroms else None)


def cmd_score(a):
    md = Path(a.model_dir)
    model = _load_model(md)
    strands = tuple(c for c in a.strands if c in "+-") or ("+",)
    scores = (s for cid, seq in read_fasta(a.fasta) for s in model.score(cid, seq, strands))
    ab = None if a.raw else calibration.load(md)
    write_scores(calibration.apply(scores, ab) if ab else scores, sys.stdout)


def cmd_models(a):
    for name, cls in sorted(list_models().items()):
        print(f"{name}\t{cls.__module__}.{cls.__name__}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sitescore", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--model", required=True)
    t.add_argument("--genome", required=True)
    t.add_argument("--annotation", required=True, help="EviAnn GFF (training labels)")
    t.add_argument("--out", required=True, help="model_dir to write")
    t.add_argument("--init", help="model_dir to fine-tune from")
    t.add_argument("--hparams", help='JSON dict passed to the model, e.g. \'{"epochs": 5}\'')
    t.add_argument("--no-calibrate", action="store_true", help="skip the Platt fit on the validation sequences")

    c = sp.add_parser("calibrate"); c.set_defaults(fn=cmd_calibrate)
    c.add_argument("--model-dir", required=True)
    c.add_argument("--genome", required=True)
    c.add_argument("--annotation", required=True, help="EviAnn GFF (truth)")
    c.add_argument("--chroms", help="comma list of sequences to fit on (default: train_info.json val_chroms)")

    s = sp.add_parser("score"); s.set_defaults(fn=cmd_score)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--fasta", required=True)
    s.add_argument("--strands", default="+-", help='"+", "-" or "+-" (default)')
    s.add_argument("--raw", action="store_true", help="emit raw probabilities even if calibration.json exists")

    m = sp.add_parser("models"); m.set_defaults(fn=cmd_models)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
