# Pretrained weights

Weights are not committed. Each pretrained `model_dir` (`sitescore.json` + checkpoint)
is a GitHub Release asset listed in `registry.yaml` with its URL and sha256.

```bash
sitescore fetch convmamba-human-grch38            # -> models/convmamba-human-grch38/ (verified)
sitescore score --model-dir models/convmamba-human-grch38 --fasta chrX.fa > sites.tsv
sitescore train --model convmamba --init models/convmamba-human-grch38 \
    --genome g.fna --annotation eviann.gff --out run/convmamba   # fine-tune on a new genome
```

| name | model | trained on | size |
| --- | --- | --- | --- |
| `convmamba-human-grch38` | convmamba | human GRCh38.p14, EviAnn labels, all chromosomes | 104 MB |

Adding a model: `tar czf <name>.tar.gz <name>/` (the directory must contain
`sitescore.json`), attach it to a release, add the entry with `sha256sum` here.
