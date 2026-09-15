"""Data profile used by data.py: which sequences are validation / held out.
The adapter registers one profile per training run from the input FASTA."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DataProfile:
    name: str
    test_chrom: str          # "" = nothing held out
    val_chroms: tuple
    expected_train: int | None
    expected_val: int | None
    test_status: str


PROFILES: dict[str, DataProfile] = {}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; registered: {sorted(PROFILES)}") from exc
