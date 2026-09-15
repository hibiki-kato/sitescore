# sitescore

Pluggable evaluator for **donor / acceptor / start / stop** sites.
Input: genome FASTA + EviAnn GFF (labels). Output: the score table UniAnn's
Viterbi consumes. The model behind it (a Conv+Mamba network today, anything else
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
sitescore models                                                   # list plug-ins
sitescore train --model convmamba --genome g.fna --annotation eviann.gff --out run/convmamba [--init models/convmamba-human-grch38]
sitescore calibrate --model-dir run/convmamba --genome g.fna --annotation eviann.gff   # Platt fit (train does this by default)
sitescore score --model-dir run/convmamba --fasta chrX.fa [--strands +] [--raw] > sites.tsv
```

`sites.tsv` columns: `chrom pos strand type motif prob` (pos = 1-based + strand
coordinate of the motif's first base; both strands by default). Pretrained weights: see `models/README.md`.

## Calibration
`sitescore train` ends with a per-type Platt fit (`sitescore/platt.py`, Bayes/Laplace
targets) on the validation sequences against the EviAnn annotation, saved as
`model_dir/calibration.json`; `sitescore score` applies it unless `--raw`.
`sitescore calibrate` refits an existing `model_dir` on chosen sequences.

## Add a model
1. Create `sitescore/models/<name>/` and subclass `SiteModel`
   (`sitescore/interface.py`): implement `train`, `load`, `score`.
2. Register it in `pyproject.toml` under `[project.entry-points."sitescore.models"]`.
3. `pytest` — `tests/test_interface.py` is the contract every model must meet.

## Models
| plug-in | code | pretrained (`models/`, not in git) |
| --- | --- | --- |
| `convmamba` | `sitescore/models/convmamba/` — dilated Conv stem + bidirectional Mamba3 context stack, four candidate heads | `convmamba-human-grch38/` (114 MB, fine-tuned on GRCh38 with EviAnn labels) |

`convmamba` plug-in: `train` builds 10 kb windows from the EviAnn GFF (validation =
smallest sequences up to `val_fraction`, or `val_chroms`), fine-tunes from
`--init` (or trains from scratch), and writes `model.pt` + `train_info.json`;
`score` tiles 10 kb windows (stride 5 kb) and emits every candidate motif on
the requested strands. Hyper-parameters: `DEFAULTS` in
`sitescore/models/convmamba/adapter.py`, overridable with `--hparams`. Needs a CUDA GPU
with `mamba-ssm`; `{"mamba_backend": "reference"}` runs on CPU for smoke tests.

## Roadmap
- ChimAnn runs `sitescore` as a separate Nextflow stage.
- Final product: folded into UniAnn (evaluator + PSAURON + Viterbi/k-Viterbi)
  as the module that takes EviAnn GFF + genome and emits missing CDS as GFF.

## Credits
The `convmamba` model (`sitescore/models/convmamba/`: architecture, data
builder, training loop, checkpoints) and the Platt calibration
(`sitescore/platt.py`, the logic in `sitescore/calibration.py`) were written by
[Chirag Adwani (@divide-by-zer0)](https://github.com/divide-by-zer0) for the
UniAnn site-scoring experiments; sitescore packages that code behind the
`SiteModel` interface. Remaining code by Hibiki Kato.
