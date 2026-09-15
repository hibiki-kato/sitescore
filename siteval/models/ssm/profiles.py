"""Data profiles and project paths for the Hsap V8S2 run.

GRCh38.p14 primary assembly (no ALTs, no patches). Chromosome-level split:

  train: chr2, 3, 4, 6, 7, 8, 11-21, X, Y
  val:   chr5, 9, 10, 22
  test:  chr1

Validation is ~17.8% of train+val by sequence (16.5% by labeled CDS features),
matching the ~20% the Dmel runs used. Four validation chromosomes keep the
structural diversity (gene density, GC/isochore, repeat content) that a
chromosome-level split turns into systematic rather than sampling variation.

chrMT (NC_012920.1) and unplaced NT_ scaffolds are excluded from every split
(different genetic code / unreliable context). MLM pretraining uses the train
chromosomes only, so chr1 and the val chromosomes are unseen by any training
signal.

Two profiles, mirroring the Dmel ``heldout_x`` / ``with_x`` pair:

``chr1_test``
    The real split described above. chr1 is held out of MLM pretraining and of
    supervised fine-tuning, so every number measured on it is honest.

``with_chr1``
    The training-exposed diagnostic. ``test_chrom`` is empty, and that single
    switch is what moves chr1: ``build_supervised_profile`` stops excluding it
    from the window set and ``build_mlm_profile`` stops excluding it from the
    MLM corpus, so both stages see it. Validation stays chr5/9/10/22, so early
    stopping and the Platt fit are unchanged. chr1 is still scored and still
    passed to UniAnn -- those numbers are an upper bound on what this
    architecture can extract from chr1, not a generalization estimate, and must
    never be read against a ``chr1_test`` run as if the two measured the same
    thing.

    This is the analogue of Dmel's ``with_x`` (``test_chrom=""`` plus
    ``pretrain_includes_chr_x=True``). The Hsap MLM builder already reads its
    corpus off ``test_chrom``, so no second flag is needed here.
"""

from dataclasses import dataclass
from pathlib import Path


TEST_ACCESSION = "NC_000001.11"  # chr1

VAL_CHROMS = (
    "NC_000005.10",  # chr5
    "NC_000009.12",  # chr9
    "NC_000010.11",  # chr10
    "NC_000022.11",  # chr22
)


@dataclass(frozen=True)
class DataProfile:
    name: str
    # Empty means "hold nothing out" -- chr1 goes into training. Both builders
    # and both validators branch on the truthiness of this field.
    test_chrom: str
    val_chroms: tuple
    # Window counts are asserted when set; None skips the count check (used for
    # Hsap where the counts are established by the first build).
    expected_train: int | None
    expected_val: int | None
    test_status: str


PROFILES = {
    "chr1_test": DataProfile(
        name="chr1_test",
        test_chrom=TEST_ACCESSION,
        val_chroms=VAL_CHROMS,
        expected_train=None,
        expected_val=None,
        test_status="held_out_test",
    ),
    "with_chr1": DataProfile(
        name="with_chr1",
        test_chrom="",
        val_chroms=VAL_CHROMS,
        expected_train=None,
        expected_val=None,
        test_status="training_exposed_diagnostic",
    ),
}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown profile {name!r}; choose from {', '.join(sorted(PROFILES))}"
        ) from exc


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Genetics_v3 layout: the experiment lives in <repo>/experiments/<name>/,
# so the repo root is two levels above the package root.
REPO_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_FASTA = (
    REPO_ROOT
    / "data"
    / "raw"
    / "Hsap"
    / "GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.fna"
)
DEFAULT_REFSEQ_GFF = (
    REPO_ROOT
    / "data"
    / "raw"
    / "Hsap"
    / "GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.filtered.gff"
)
DEFAULT_EVIANN_GFF = (
    REPO_ROOT
    / "data"
    / "eviann"
    / "Hsap"
    / "GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.fna.pseudo_label.gff"
)


def supervised_data_dir(name):
    return PACKAGE_ROOT / "data" / get_profile(name).name / "supervised"


def pretrain_data_dir(name):
    return PACKAGE_ROOT / "data" / get_profile(name).name / "mlm"


def checkpoint_dir(profile, run):
    return PACKAGE_ROOT / "checkpoints" / get_profile(profile).name / run
