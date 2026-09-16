"""convmamba evaluator: ConvMambaNet (dilated Conv stem + bidirectional Mamba3
context stack + four candidate heads). model/checkpoints/common/data/training
are the model's own code; this file is the only glue to the SiteModel
interface. `_install_profile` tells the data builder which sequences of the
input FASTA are validation.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ...interface import SITE_TYPES, SiteBlock, SiteModel, SiteScore, read_fasta
from . import common

CHECKPOINT = "model.pt"
PROFILE = "sitescore"
MOTIF_LEN = {"donor": 2, "acceptor": 2, "start": 3, "stop": 3}

# Training defaults; any key can be overridden via
# `sitescore train --hparams '{...}'`.
DEFAULTS = dict(
    epochs=64, patience=8, batch_size=2, grad_accum=8, num_workers=4, amp="bf16",
    auto_batch=True,             # probe the largest batch that fits the GPU (autobatch.py);
    auto_batch_max=64,           #   grad_accum is rescaled so batch_size*grad_accum is kept
    auto_batch_target_mem=0.85,  #   fraction of free VRAM to target
    encoder_lr=3.0e-5, head_lr=1.0e-4, weight_decay=1.0e-4, warmup_steps=300,
    max_grad_norm=1.0, selection_metric="val_loss", soft_weight=1.0, seed=42,
    mamba_backend="mamba3",      # "reference" = CPU smoke tests only
    val_chroms=None,             # explicit list, else chosen by val_fraction
    val_fraction=0.15,           # smallest sequences held out until this share
    trust_region="locus",        # or "exon"
    refseq_gff=None,             # reference GFF -> conditional alpha priors
    alpha=None,                  # float | {type: float}; default: init ckpt's, else 0
    keep_data=False,             # keep out_dir/data (windows can be GBs)
    score_batch=32,
)


def choose_val_chroms(lengths: dict[str, int], fraction: float) -> list[str]:
    """Hold out the smallest sequences until `fraction` of the genome is covered."""
    total, held, acc = sum(lengths.values()), [], 0
    for cid in sorted(lengths, key=lengths.get):
        if len(held) + 1 >= len(lengths):      # keep at least one training sequence
            break
        held.append(cid)
        acc += lengths[cid]
        if acc >= fraction * total:
            break
    return held


def _install_profile(chroms: list[str], val_chroms: list[str]) -> None:
    from . import adapter_profile as profiles
    common.NC_TO_NAME = {c: c for c in chroms}          # data builder iterates this
    profiles.PROFILES[PROFILE] = profiles.DataProfile(
        name=PROFILE, test_chrom="", val_chroms=tuple(val_chroms),
        expected_train=None, expected_val=None, test_status="sitescore")


def _resolve_alpha(hp, init, stats) -> dict[str, float]:
    a = hp["alpha"]
    if a is None and hp["refseq_gff"] and stats.get("alpha_priors"):
        a = stats["alpha_priors"]["conditional"]
    if a is None:
        a = init["alpha"] if init else 0.0
    if isinstance(a, (int, float)):
        a = {t: float(a) for t in SITE_TYPES}
    return {t: float(a.get(t, 0.0)) for t in SITE_TYPES}


class ConvMambaSiteModel(SiteModel):
    name = "convmamba"

    def __init__(self, model, device, batch_size: int = 32, amp: str = "bf16"):
        self.model, self.device, self.batch_size, self.amp = model, device, batch_size, amp

    # ------------------------------------------------------------------ train
    @classmethod
    def train(cls, genome: Path, annotation: Path, out_dir: Path,
              init_dir: Path | None = None, **hparams) -> "ConvMambaSiteModel":
        import torch
        from torch.utils.data import DataLoader
        from .checkpoints import MODEL_KIND, load_finetune_checkpoint
        from .data import SupervisedWindowDataset, build_supervised_profile
        from .model import ConvMambaNet, ConvMambaConfig
        from .training import train_model

        hp = {**DEFAULTS, **hparams}
        torch.manual_seed(hp["seed"]); np.random.seed(hp["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out_dir = Path(out_dir)

        lengths = {cid: len(seq) for cid, seq in read_fasta(genome)}
        if len(lengths) < 2:
            raise ValueError("training needs >= 2 sequences: one is held out for validation")
        val = list(hp["val_chroms"] or choose_val_chroms(lengths, hp["val_fraction"]))
        _install_profile(list(lengths), val)
        data_dir = out_dir / "data"
        stats = build_supervised_profile(PROFILE, str(genome), str(annotation), data_dir,
                                         trust_region=hp["trust_region"],
                                         refseq_gff_path=hp["refseq_gff"])

        init = load_finetune_checkpoint(Path(init_dir) / CHECKPOINT, device) if init_dir else None
        cfg = (ConvMambaConfig.from_dict(init["model_config"]) if init
               else ConvMambaConfig(mamba_backend=hp["mamba_backend"]))
        model = ConvMambaNet(cfg).to(device)
        if init:
            model.load_state_dict(init["model_state"])
        alpha = _resolve_alpha(hp, init, stats)

        train_set, val_set = SupervisedWindowDataset(data_dir, "train"), SupervisedWindowDataset(data_dir, "val")
        batch, accum = hp["batch_size"], hp["grad_accum"]
        if hp["auto_batch"] and device.type == "cuda":
            from .autobatch import autotune_finetune_batch_size, scaled_grad_accum
            batch = autotune_finetune_batch_size(model, train_set, device, hp["amp"], initial_batch=batch,
                                                 max_batch=hp["auto_batch_max"],
                                                 target_fraction=hp["auto_batch_target_mem"])
            accum = scaled_grad_accum(hp["batch_size"] * hp["grad_accum"], batch, accum)
            print(f"auto_batch: batch_size={batch} grad_accum={accum} (effective {batch * accum})", flush=True)
        kw = dict(num_workers=hp["num_workers"], pin_memory=device.type == "cuda")
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, **kw)
        val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, **kw)
        run_args = {**hp, "init_dir": str(init_dir) if init_dir else None, "val_chroms": val,
                    "alpha_resolved": alpha, "batch_size_used": batch, "grad_accum_used": accum}
        train_model(model=model, model_kind=MODEL_KIND, run_kind="pretrained" if init else "scratch",
                    train_loader=train_loader, val_loader=val_loader, device=device,
                    out_dir=out_dir, checkpoint_name=CHECKPOINT, profile_name=PROFILE,
                    profile_stats=stats, model_config=cfg.to_dict(),
                    epochs=hp["epochs"], encoder_lr=hp["encoder_lr"], head_lr=hp["head_lr"],
                    weight_decay=hp["weight_decay"], grad_accum=accum,
                    warmup_steps=hp["warmup_steps"], max_grad_norm=hp["max_grad_norm"],
                    patience=hp["patience"], amp=hp["amp"], run_args=run_args,
                    alpha=alpha, soft_weight=hp["soft_weight"])
        (out_dir / "train_info.json").write_text(json.dumps(
            {"checkpoint": CHECKPOINT, "val_chroms": val, "alpha": alpha,
             "init_dir": run_args["init_dir"], "batch_size_used": batch, "grad_accum_used": accum,
             "hparams": hp}, indent=2, default=str))
        if not hp["keep_data"]:
            shutil.rmtree(data_dir, ignore_errors=True)
        return cls.load(out_dir)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, model_dir: Path, device: str | None = None) -> "ConvMambaSiteModel":
        import torch
        from .checkpoints import load_checkpoint
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "sitescore.json").read_text())
        hp = {**DEFAULTS, **cfg.get("hparams", {})}
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, _ = load_checkpoint(model_dir / cfg.get("checkpoint", CHECKPOINT), device)
        return cls(model, device, batch_size=hp["score_batch"], amp=hp["amp"])

    # ------------------------------------------------------------------ score
    def _site_probs(self, encoded: np.ndarray) -> np.ndarray:
        """(L, 4) P(site) per base; each base predicted by exactly one window."""
        import contextlib
        import torch
        plan = common.make_windows(len(encoded))
        out = np.zeros((len(encoded), len(SITE_TYPES)), dtype=np.float32)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if self.amp == "bf16" and self.device.type == "cuda" else contextlib.nullcontext())
        with torch.inference_mode(), amp:
            for i in range(0, len(plan), self.batch_size):
                chunk = plan[i:i + self.batch_size]
                ids = np.full((len(chunk), common.WINDOW_SIZE), common.N, dtype=np.int64)
                for j, w in enumerate(chunk):
                    ids[j, :w["actual_len"]] = encoded[w["win_start"]:w["win_end"]]
                logits = self.model(torch.from_numpy(ids).to(self.device), task="sites")
                probs = torch.softmax(logits.float(), dim=-1)[..., 1].cpu().numpy()  # (B, L, 4)
                for j, w in enumerate(chunk):
                    lo, hi = w["pred_lo"], w["pred_hi"]
                    out[w["win_start"] + lo:w["win_start"] + hi] = probs[j, lo:hi]
        return out

    def score_blocks(self, chrom: str, seq: str, strands: Iterable[str] = ("+",)) -> Iterator[SiteBlock]:
        L = len(seq)
        for strand in strands:
            s = seq if strand == "+" else common.reverse_complement(seq)
            enc = common.encode_sequence(s)
            probs = self._site_probs(enc)
            cands = common.find_candidates(enc)
            raw = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
            for k, kind in enumerate(SITE_TYPES):
                p = np.asarray(cands[kind], dtype=np.int64)
                n = MOTIF_LEN[kind]
                pos = p + 1 if strand == "+" else L - p  # frame_to_genomic_1based, vectorized
                motif = raw[p[:, None] + np.arange(n)].view(f"S{n}").ravel()
                yield SiteBlock(chrom, strand, kind, pos, motif, probs[p, k])

    def score(self, chrom: str, seq: str, strands: Iterable[str] = ("+",)) -> Iterator[SiteScore]:
        for block in self.score_blocks(chrom, seq, strands):
            yield from block.rows()
