# Model weights

Weights are **not** committed. Each pretrained model is a `model_dir`
(`siteval.json` + checkpoint files) published as an archive and listed in
`registry.yaml` with its URL and sha256:

```bash
siteval-fetch ssm-hsap-v8s5-with_chr1 models/      # TODO: helper; until then download + verify by hand
siteval score --model-dir models/ssm-hsap-v8s5-with_chr1 --fasta chrX.fa > sites.tsv
siteval train --model ssm --init models/ssm-hsap-v8s5-with_chr1 ...   # fine-tune for a new genome
```

Storage: GitHub Releases of this repo (one asset per model version). Move to
Hugging Face Hub if archives exceed the 2 GB asset limit.
