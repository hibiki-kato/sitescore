"""siteval train | score | models

  siteval train --model ssm --genome g.fna --annotation eviann.gff --out model_dir [--init base_model_dir]
  siteval score --model-dir model_dir --fasta seq.fa [--strands +,-] > sites.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .interface import read_fasta, write_scores
from .registry import get_model, list_models

CONFIG = "siteval.json"   # written into every model_dir: {"model": name, "hparams": {...}}


def cmd_train(a):
    cls = get_model(a.model)
    hparams = json.loads(a.hparams) if a.hparams else {}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / CONFIG).write_text(json.dumps({"model": a.model, "hparams": hparams}, indent=2))
    cls.train(Path(a.genome), Path(a.annotation), out,
              init_dir=Path(a.init) if a.init else None, **hparams)


def cmd_score(a):
    md = Path(a.model_dir)
    cfg = json.loads((md / CONFIG).read_text())
    model = get_model(cfg["model"]).load(md)
    strands = tuple(c for c in a.strands if c in "+-") or ("+",)
    write_scores((s for cid, seq in read_fasta(a.fasta) for s in model.score(cid, seq, strands)),
                 sys.stdout)


def cmd_models(a):
    for name, cls in sorted(list_models().items()):
        print(f"{name}\t{cls.__module__}.{cls.__name__}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="siteval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--model", required=True)
    t.add_argument("--genome", required=True)
    t.add_argument("--annotation", required=True, help="EviAnn GFF (training labels)")
    t.add_argument("--out", required=True, help="model_dir to write")
    t.add_argument("--init", help="model_dir to fine-tune from")
    t.add_argument("--hparams", help='JSON dict passed to the model, e.g. \'{"epochs": 5}\'')

    s = sp.add_parser("score"); s.set_defaults(fn=cmd_score)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--fasta", required=True)
    s.add_argument("--strands", default="+-", help='"+", "-" or "+-" (default)')

    m = sp.add_parser("models"); m.set_defaults(fn=cmd_models)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
