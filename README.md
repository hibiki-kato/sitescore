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
siteval score --model-dir run/ssm --fasta chrX.fa [--strands +] > sites.tsv
```

`sites.tsv` columns: `chrom pos strand type motif prob` (pos = 1-based + strand
coordinate of the motif's first base; both strands by default). Pretrained weights: see `models/README.md`.

## Add a model
1. Create `siteval/models/<name>/` and subclass `SiteModel`
   (`siteval/interface.py`): implement `train`, `load`, `score`.
2. Register it in `pyproject.toml` under `[project.entry-points."siteval.models"]`.
3. `pytest` — `tests/test_interface.py` is the contract every model must meet.

## Models
| plug-in | code | pretrained (`models/`, not in git) |
| --- | --- | --- |
| `ssm` | `siteval/models/ssm/` — GeneFinderV8S2, Conv + bidirectional Mamba3 (from the v8s5_hsap experiment) | `ssm-hsap-v8s5-with_chr1/` (114 MB, GRCh38 fine-tuned) |

`ssm` plug-in: `train` builds V8S5 windows from the EviAnn GFF (validation =
smallest sequences up to `val_fraction`, or `val_chroms`), fine-tunes from
`--init` (or trains from scratch), and writes `v8s2_best.pt` + `train_info.json`;
`score` tiles 10 kb windows (stride 5 kb) and emits every candidate motif on
the requested strands. Hyper-parameters: `DEFAULTS` in
`siteval/models/ssm/adapter.py`, overridable with `--hparams`. Needs a CUDA GPU
with `mamba-ssm`; `{"mamba_backend": "reference"}` runs on CPU for smoke tests.

## Roadmap
- ChimAnn runs `siteval` as a separate Nextflow stage.
- Final product: folded into UniAnn (evaluator + PSAURON + Viterbi/k-Viterbi)
  as the module that takes EviAnn GFF + genome and emits missing CDS as GFF.
