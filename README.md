# siteval

Pluggable evaluator for **donor / acceptor / start / stop** sites.
Input: genome FASTA + EviAnn GFF (labels). Output: the score table UniAnn's
Viterbi consumes. The model behind it (SSM today, LLM or anything else
tomorrow) is swappable behind one interface.

```
genome.fna + eviann.gff ──train──> model_dir ──score(seq.fa)──> sites.tsv
```

## Install
```bash
pip install -e .
```

## Use
```bash
siteval models                                                   # list plug-ins
siteval train --model ssm --genome g.fna --annotation eviann.gff --out run/ssm [--init models/ssm-dmel-v1]
siteval score --model-dir run/ssm --fasta chrX.fa > sites.tsv
```

`sites.tsv` columns: `chrom pos strand type motif prob` (1-based, + strand,
prob in (0,1]). Pretrained weights: see `models/README.md`.

## Add a model
1. Create `siteval/models/<name>/` and subclass `SiteModel`
   (`siteval/interface.py`): implement `train`, `load`, `score`.
2. Register it in `pyproject.toml` under `[project.entry-points."siteval.models"]`.
3. `pytest` — `tests/test_interface.py` is the contract every model must meet.

## Roadmap
- ChimAnn runs `siteval` as a separate Nextflow stage.
- Final product: folded into UniAnn (evaluator + PSAURON + Viterbi/k-Viterbi)
  as the module that takes EviAnn GFF + genome and emits missing CDS as GFF.
