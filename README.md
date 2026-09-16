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
pip install -e ".[convmamba]"     # torch + mamba-ssm (CUDA GPU); see environment.yml for a conda env
sitescore fetch convmamba-human-grch38          # pretrained weights (GitHub Release, sha256-verified)
```
`mamba-ssm` builds CUDA kernels at install time; a CPU-only machine can still run
the tests and smoke trainings with `{"mamba_backend": "reference"}`.

## Use
```bash
sitescore models                                                   # list plug-ins
sitescore train --model convmamba --genome g.fna --annotation eviann.gff --out run/convmamba [--init models/convmamba-human-grch38]
sitescore calibrate --model-dir run/convmamba --genome g.fna --annotation eviann.gff   # Platt fit (train does this by default)
sitescore score --model-dir run/convmamba --fasta chrX.fa [--strands +] [--raw] > sites.tsv
```

`sites.tsv` columns: `chrom pos strand type motif prob` (pos = 1-based + strand
coordinate of the motif's first base; both strands by default). Pretrained weights: `models/README.md`.

Training holds out the smallest sequences until `val_fraction` (15%) of the genome
is covered (or the explicit `val_chroms` list) for early stopping (validation loss),
checkpoint selection and the Platt fit; the rest is trained on. `model_dir/metrics.jsonl`
and `model_dir/plots/` record train/validation loss and per-type F1 / PR-AUC per epoch,
including **epoch 0** = the starting weights (`--init` or random) evaluated on the
validation split before any update; it competes for the best checkpoint like any
other epoch, so fine-tuning that only hurts keeps the pretrained weights.

## Calibration
`sitescore train` ends with a per-type Platt fit (`sitescore/platt.py`, Bayes/Laplace
targets) on the validation sequences against the EviAnn annotation, saved as
`model_dir/calibration.json`; `sitescore score` applies it unless `--raw`.
`sitescore calibrate` refits an existing `model_dir` on chosen sequences.

## Add a model
1. Create `sitescore/models/<name>/` and subclass `SiteModel`
   (`sitescore/interface.py`): implement `train`, `load`, `score` (one `SiteScore`
   per candidate). Optionally override `score_blocks` to yield `SiteBlock` arrays
   per (strand, type) instead; the CLI uses it when present and it avoids
   per-row Python objects (convmamba: 3x faster output).
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
`sitescore/models/convmamba/adapter.py`, overridable with `--hparams`. By default
the batch size is probed against free VRAM (`auto_batch`, up to `auto_batch_max`)
and `grad_accum` rescaled so the effective batch stays `batch_size * grad_accum`;
validation runs `evals_per_epoch` times per epoch (default 4; fractional epochs in
`metrics.jsonl`) and early stopping is on validation loss after `patience`
evaluations without improvement (fine-tuning from a pretrained model usually peaks
within the first epoch or two, so this stops long before the epoch cap). Needs a CUDA GPU
with `mamba-ssm`; `{"mamba_backend": "reference"}` runs on CPU for smoke tests.

## Results
Site-level precision-recall of the convmamba scores against RefSeq (all candidate
motifs, both strands), after fine-tuning from `convmamba-human-grch38` on EviAnn labels:

| genome | donor AP | acceptor AP | start AP | stop AP |
| --- | --- | --- | --- | --- |
| D. melanogaster, 7 main sequences | 0.970 | 0.964 | 0.847 | 0.859 |
| D. melanogaster, held-out 2L only | 0.968 | 0.962 | 0.827 | 0.855 |

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
