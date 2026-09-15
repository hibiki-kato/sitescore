"""Nucleotide-resolution Conv + Mamba site model with optional MLM pretraining.

Dilated depthwise Conv stem -> phase packing -> bidirectional Mamba3 + SwiGLU
context stack -> unpack/fuse -> Conv refinement -> four candidate heads
(donor, acceptor, start, stop). Attention-free.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


DNA_VOCAB_SIZE = 5
MASK_TOKEN = 5
MODEL_VOCAB_SIZE = 6


@dataclass
class ConvMambaConfig:
    vocab_size: int = MODEL_VOCAB_SIZE
    dna_vocab_size: int = DNA_VOCAB_SIZE
    mask_token: int = MASK_TOKEN
    local_dim: int = 192
    context_dim: int = 256
    pack_size: int = 4
    conv_kernel: int = 15
    conv_dilations: tuple = (1, 2, 4, 8, 16, 32)
    refine_dilations: tuple = (1, 2)
    # 5 bidirectional Mamba3 blocks in the context stack
    n_ssm_layers: int = 5
    d_state: int = 64
    headdim: int = 64
    ff_mult: int = 4
    dropout: float = 0.1
    mamba_backend: str = "mamba3"
    grad_checkpoint: bool = True

    def to_dict(self):
        result = asdict(self)
        result["conv_dilations"] = list(self.conv_dilations)
        result["refine_dilations"] = list(self.refine_dilations)
        return result

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "conv_dilations" in values:
            values["conv_dilations"] = tuple(values["conv_dilations"])
        if "refine_dilations" in values:
            values["refine_dilations"] = tuple(values["refine_dilations"])
        return cls(**values)


class ResidualDepthwiseConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, dilation, dropout):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conv kernel must be odd")
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
        )
        self.pointwise = nn.Conv1d(dim, 2 * dim, 1)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm.apply(_init_weights)
        self.output.apply(_init_weights)

    def forward(self, x):
        residual = x
        hidden = self.norm(x).transpose(1, 2)
        hidden = self.depthwise(hidden)
        value, gate = self.pointwise(hidden).chunk(2, dim=1)
        hidden = (value * F.silu(gate)).transpose(1, 2)
        return residual + self.dropout(self.output(hidden))


def pack_phases(x, pack_size):
    """Concatenate fixed within-pack phases without losing their ordering."""
    batch, length, dim = x.shape
    pad = (-length) % pack_size
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    return x.reshape(batch, (length + pad) // pack_size, pack_size * dim), pad


def unpack_phases(x, pack_size, phase_dim, pad=0):
    batch, packed_length, width = x.shape
    expected = pack_size * phase_dim
    if width != expected:
        raise ValueError(f"packed width {width} does not match expected {expected}")
    unpacked = x.reshape(batch, packed_length * pack_size, phase_dim)
    return unpacked[:, :-pad] if pad else unpacked


class PhasePacker(nn.Module):
    def __init__(self, local_dim, context_dim, pack_size):
        super().__init__()
        self.pack_size = pack_size
        self.proj = nn.Linear(pack_size * local_dim, context_dim)
        self.norm = nn.LayerNorm(context_dim)
        self.apply(_init_weights)

    def forward(self, x):
        packed, pad = pack_phases(x, self.pack_size)
        return self.norm(self.proj(packed)), pad


class PhaseUnpacker(nn.Module):
    def __init__(self, context_dim, local_dim, pack_size):
        super().__init__()
        self.pack_size = pack_size
        self.local_dim = local_dim
        self.proj = nn.Linear(context_dim, pack_size * local_dim)
        self.apply(_init_weights)

    def forward(self, x, pad):
        return unpack_phases(
            self.proj(x), self.pack_size, self.local_dim, pad=pad
        )


class ReferenceSequenceMixer(nn.Module):
    """Small CPU-safe mixer used only by tests and smoke runs."""

    def __init__(self, d_model, **_):
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model)
        self.gate = nn.Linear(d_model, 2 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.gate.apply(_init_weights)
        self.output.apply(_init_weights)

    def forward(self, x):
        hidden = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        value, gate = self.gate(hidden).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


def resolve_mamba_factory(backend):
    if backend == "reference":
        return ReferenceSequenceMixer
    if backend != "mamba3":
        raise ValueError(f"unsupported Mamba backend: {backend}")
    try:
        from mamba_ssm import Mamba3
    except ImportError as exc:
        raise ImportError(
            "convmamba training/scoring requires mamba_ssm.Mamba3 (see environment.yml). "
            "The 'reference' backend "
            "is reserved for CPU smoke tests."
        ) from exc
    return Mamba3


class BiMamba3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        headdim,
        backend="mamba3",
        mixer_factory=None,
    ):
        super().__init__()
        factory = mixer_factory or resolve_mamba_factory(backend)
        self.forward_mixer = factory(
            d_model=d_model, d_state=d_state, headdim=headdim
        )
        self.backward_mixer = factory(
            d_model=d_model, d_state=d_state, headdim=headdim
        )

    def forward(self, x):
        forward = self.forward_mixer(x)
        backward = self.backward_mixer(x.flip(1)).flip(1)
        return forward + backward


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.input = nn.Linear(dim, 2 * hidden_dim)
        self.output = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.apply(_init_weights)

    def forward(self, x):
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(self.dropout(value * F.silu(gate)))


class SSMBlock(nn.Module):
    """Attention-free context block: bidirectional Mamba3 + SwiGLU FFN."""

    def __init__(self, config, mixer_factory=None):
        super().__init__()
        dim = config.context_dim
        self.mamba_norm = nn.LayerNorm(dim)
        self.mamba = BiMamba3(
            dim,
            config.d_state,
            config.headdim,
            backend=config.mamba_backend,
            mixer_factory=mixer_factory,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, config.ff_mult * dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.mamba_norm.apply(_init_weights)
        self.ffn_norm.apply(_init_weights)

    def forward(self, x):
        x = x + self.dropout(self.mamba(self.mamba_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class CandidateHead(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )
        self.apply(_init_weights)

    def forward(self, x):
        return self.layers(x)


class ConvMambaNet(nn.Module):
    SITE_NAMES = ("donor", "acceptor", "start", "stop")

    def __init__(self, config=None, mixer_factory=None):
        super().__init__()
        self.config = config or ConvMambaConfig()
        cfg = self.config
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.local_dim)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.local_stem = nn.ModuleList(
            [
                ResidualDepthwiseConvBlock(
                    cfg.local_dim, cfg.conv_kernel, dilation, cfg.dropout
                )
                for dilation in cfg.conv_dilations
            ]
        )
        self.packer = PhasePacker(cfg.local_dim, cfg.context_dim, cfg.pack_size)
        self.ssm_blocks = nn.ModuleList(
            [SSMBlock(cfg, mixer_factory=mixer_factory) for _ in range(cfg.n_ssm_layers)]
        )
        self.context_norm = nn.LayerNorm(cfg.context_dim)
        self.unpacker = PhaseUnpacker(cfg.context_dim, cfg.local_dim, cfg.pack_size)
        self.fusion_gate = nn.Linear(2 * cfg.local_dim, cfg.local_dim)
        self.fusion_norm = nn.LayerNorm(cfg.local_dim)
        self.refinement = nn.ModuleList(
            [
                ResidualDepthwiseConvBlock(
                    cfg.local_dim, cfg.conv_kernel, dilation, cfg.dropout
                )
                for dilation in cfg.refine_dilations
            ]
        )
        self.heads = nn.ModuleList(
            [CandidateHead(cfg.local_dim, cfg.dropout) for _ in self.SITE_NAMES]
        )
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.dna_vocab_size))
        self.embedding.apply(_init_weights)
        self.context_norm.apply(_init_weights)
        self.fusion_gate.apply(_init_weights)
        self.fusion_norm.apply(_init_weights)

    def encode(self, input_ids):
        local = self.embedding_dropout(self.embedding(input_ids))
        for block in self.local_stem:
            local = block(local)
        context, pad = self.packer(local)
        for block in self.ssm_blocks:
            if (
                self.config.grad_checkpoint
                and self.training
                and self.config.mamba_backend != "mamba3"
            ):
                context = checkpoint(block, context, use_reentrant=False)
            else:
                context = block(context)
        context = self.context_norm(context)
        expanded = self.unpacker(context, pad)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((local, expanded), dim=-1)))
        hidden = self.fusion_norm(local + gate * expanded)
        for block in self.refinement:
            hidden = block(hidden)
        return hidden

    def forward_sites(self, input_ids):
        hidden = self.encode(input_ids)
        return torch.stack([head(hidden) for head in self.heads], dim=2)

    def forward_mlm(self, input_ids):
        hidden = self.encode(input_ids)
        return F.linear(
            hidden,
            self.embedding.weight[: self.config.dna_vocab_size],
            self.mlm_bias,
        )

    def forward(self, input_ids, task="sites"):
        if task == "sites":
            return self.forward_sites(input_ids)
        if task == "mlm":
            return self.forward_mlm(input_ids)
        if task == "both":
            hidden = self.encode(input_ids)
            sites = torch.stack([head(hidden) for head in self.heads], dim=2)
            mlm = F.linear(
                hidden,
                self.embedding.weight[: self.config.dna_vocab_size],
                self.mlm_bias,
            )
            return mlm, sites
        raise ValueError(f"unknown task: {task}")

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_from_checkpoint(checkpoint, device=None):
    config = ConvMambaConfig.from_dict(checkpoint["model_config"])
    model = ConvMambaNet(config)
    model.load_state_dict(checkpoint["model_state"])
    if device is not None:
        model = model.to(device)
    return model
