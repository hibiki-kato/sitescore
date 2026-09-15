"""Contract test: any SiteModel must survive this with a toy sequence."""
import io
import re
from pathlib import Path

from sitescore.interface import SITE_TYPES, SiteModel, SiteScore, write_scores

MOTIFS = {"donor": ("GT",), "acceptor": ("AG",), "start": ("ATG",), "stop": ("TAA", "TAG", "TGA")}


class DummyModel(SiteModel):
    """Scores every candidate motif 0.5; the shape reference for real models."""
    name = "dummy"

    @classmethod
    def train(cls, genome, annotation, out_dir, init_dir=None, **hp):
        return cls()

    @classmethod
    def load(cls, model_dir):
        return cls()

    def score(self, chrom, seq, strands=("+",)):
        for t in SITE_TYPES:
            for m in MOTIFS[t]:
                for hit in re.finditer(f"(?={m})", seq):
                    yield SiteScore(chrom, hit.start() + 1, "+", t, m, 0.5)


def test_dummy_roundtrip(tmp_path: Path):
    model = DummyModel.train(tmp_path, tmp_path, tmp_path)
    buf = io.StringIO()
    n = write_scores(model.score("chr1", "ATGGTAAGTTAA"), buf)
    lines = buf.getvalue().splitlines()
    assert lines[0] == "chrom\tpos\tstrand\ttype\tmotif\tprob"
    assert n == len(lines) - 1 == 6
    assert "chr1\t1\t+\tstart\tATG\t0.5" in lines
