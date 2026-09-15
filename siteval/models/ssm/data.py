"""Supervised and MLM data builders/loaders for the Hsap V8S5 run.

V8S5 is the V8S2 architecture (Conv stem + pure bidirectional-Mamba3 context
stack) on the **s3 window set** -- every non-gap window is kept -- with an
**asymmetric label-smoothing target** replacing the hard 0/1 supervision.

    window selection   identical to V8S3 (gap rule only)
    targets            y = 1      EviAnn positive site
                       y = 0      candidate inside an EviAnn CDS locus
                       y = alpha  candidate outside every EviAnn CDS locus

The middle row is the s4 insight applied per *candidate* instead of per
*window*: inside an annotated coding locus EviAnn has already resolved the
exon/intron structure, so an unannotated GT there really is not a donor. The
third row is the admission that outside those loci the annotation is silent --
a GT in bulk intergenic sequence may well be a donor of a gene EviAnn missed,
so calling it a hard negative is a lie the model is forced to fit. s3 tells
that lie ~1.5 billion times per epoch, which is what drives its background
probabilities to underflow UniAnn's log floor.

Everything the *builder* has to add for that is one extra per-window array:
which bases of the window lie inside an EviAnn CDS locus on the window's own
strand. It is stored as clipped **intervals** (CSR ``indptr``/``start``/``end``)
rather than a bitmap, because CDS loci are long runs -- typically 0-3 per 10 kb
window, so the whole train split costs ~20 MB instead of 1.3 GB.

The builder also estimates the smoothing constants ``alpha`` themselves, from
the **RefSeq reference annotation** on the training chromosomes only, and
records them in ``stats.json``. Two estimates are stored:

    global        P(RefSeq site | candidate)                       per site type
    conditional   P(RefSeq site | candidate, outside EviAnn CDS)   per site type

``conditional`` is the default the trainer uses: it is the expected value of
the unknown label on exactly the population being smoothed, so it is the
Bayes-optimal constant target for that population. Using RefSeq at all is
deliberate leakage ("cheating"), confined to four scalars and confined to the
training chromosomes -- chr1 and the four validation chromosomes contribute
nothing to it. See ``README.md`` for how to replace it with a non-leaking
estimate later.

The builder **streams**, exactly as in V8S3/V8S4:

1. Pass 1 decides which windows survive (gap rule only) and how many land in
   each split, without touching labels.
2. The sequences are written straight into an on-disk ``.npy`` opened with
   ``np.lib.format.open_memmap`` and filled row by row.
3. Labels are stored **sparsely** (CSR-style ``indptr``/``col``/``val``); the
   trusted regions are stored as CSR intervals.

Layout written per split (``train`` / ``val``):

    <split>_sequence.npy   int8 (n_windows, window) -- memory-mapped at train time
    <split>_meta.npz       sparse labels + trusted intervals + chrom / strand /
                           window_start

``build_mlm_profile`` is unchanged from V8S2 and still writes ``{split}.npz``;
V8S5 reuses the V8S2 MLM checkpoint, so it normally never runs.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import common
from .profiles import TEST_ACCESSION, get_profile


SUPERVISED_FORMAT = "streaming_sparse_trusted_v1"
TRUST_REGIONS = ("locus", "exon")
SITE_NAMES = ("donor", "acceptor", "start", "stop")
_STATS_BLOCK = 4096
# Elements per chunk when block-summing a chromosome-length mask (see
# _region_counts). 16 M keeps the int32 working copy around 64 MB.
_BLOCK_SUM_CHUNK = 16_000_000


class SparseLabelStore:
    """CSR-style store for the mostly-zero per-base label vectors."""

    def __init__(self, indptr, col, val, width):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.col = np.asarray(col, dtype=np.int32)
        self.val = np.asarray(val, dtype=np.int8)
        self.width = int(width)

    def __len__(self):
        return int(len(self.indptr) - 1)

    def dense(self, index):
        out = np.zeros(self.width, dtype=np.int8)
        lo = int(self.indptr[index])
        hi = int(self.indptr[index + 1])
        if hi > lo:
            out[self.col[lo:hi]] = self.val[lo:hi]
        return out

    def dense_block(self, start, stop):
        out = np.zeros((stop - start, self.width), dtype=np.int8)
        lo = int(self.indptr[start])
        hi = int(self.indptr[stop])
        if hi > lo:
            rows = (
                np.repeat(
                    np.arange(start, stop, dtype=np.int64),
                    np.diff(self.indptr[start : stop + 1]),
                )
                - start
            )
            out[rows, self.col[lo:hi]] = self.val[lo:hi]
        return out

    def take(self, indices, axis=0):
        """Dense rows for ``indices``, mirroring ``ndarray.take``.

        Callers that accept either a plain array or this store -- ``autobatch``
        probes ``dataset.splice`` / ``dataset.start_stop`` directly -- can then
        use one code path for both.
        """
        if axis != 0:
            raise ValueError("SparseLabelStore.take supports axis=0 only")
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        out = np.zeros((indices.size, self.width), dtype=np.int8)
        for row, index in enumerate(indices):
            lo = int(self.indptr[index])
            hi = int(self.indptr[index + 1])
            if hi > lo:
                out[row, self.col[lo:hi]] = self.val[lo:hi]
        return out

    def nnz_per_row(self):
        return np.diff(self.indptr)


class TrustedRegionStore:
    """CSR-style store of per-window trusted **intervals**.

    Row ``i`` holds the window-relative, half-open intervals
    ``[start[j], end[j])`` covered by an EviAnn CDS locus on that window's own
    strand, already expressed in the window's own orientation (so a ``-``
    strand row is indexed exactly like its reverse-complemented sequence row).

    Intervals, not a bitmap: CDS loci are long runs, so a 10 kb window carries
    0-3 of them. The train split costs tens of MB this way against 1.3 GB for a
    packed bitmap, and reconstruction is a couple of slice assignments.
    """

    def __init__(self, indptr, start, end, width):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.start = np.asarray(start, dtype=np.int32)
        self.end = np.asarray(end, dtype=np.int32)
        self.width = int(width)

    def __len__(self):
        return int(len(self.indptr) - 1)

    def dense(self, index):
        out = np.zeros(self.width, dtype=bool)
        lo = int(self.indptr[index])
        hi = int(self.indptr[index + 1])
        for j in range(lo, hi):
            out[int(self.start[j]) : int(self.end[j])] = True
        return out

    def dense_block(self, start, stop):
        out = np.zeros((stop - start, self.width), dtype=bool)
        for row, index in enumerate(range(start, stop)):
            lo = int(self.indptr[index])
            hi = int(self.indptr[index + 1])
            for j in range(lo, hi):
                out[row, int(self.start[j]) : int(self.end[j])] = True
        return out

    def take(self, indices, axis=0):
        if axis != 0:
            raise ValueError("TrustedRegionStore.take supports axis=0 only")
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        out = np.zeros((indices.size, self.width), dtype=bool)
        for row, index in enumerate(indices):
            lo = int(self.indptr[index])
            hi = int(self.indptr[index + 1])
            for j in range(lo, hi):
                out[row, int(self.start[j]) : int(self.end[j])] = True
        return out

    def covered_per_row(self):
        """Trusted base count for every row, without densifying."""
        widths = (self.end.astype(np.int64) - self.start.astype(np.int64))
        if widths.size == 0:
            return np.zeros(len(self), dtype=np.int64)
        return np.add.reduceat(
            np.concatenate([widths, [0]]), self.indptr[:-1]
        ) * (np.diff(self.indptr) > 0)


class _SparseLabelBuilder:
    def __init__(self):
        self.indptr = [0]
        self.col = []
        self.val = []

    def append(self, dense):
        nz = np.flatnonzero(dense)
        if nz.size:
            self.col.append(nz.astype(np.int32))
            self.val.append(dense[nz].astype(np.int8))
        self.indptr.append(self.indptr[-1] + int(nz.size))

    def finish(self):
        col = (
            np.concatenate(self.col)
            if self.col
            else np.zeros(0, dtype=np.int32)
        )
        val = (
            np.concatenate(self.val)
            if self.val
            else np.zeros(0, dtype=np.int8)
        )
        return np.asarray(self.indptr, dtype=np.int64), col, val


class _IntervalBuilder:
    def __init__(self):
        self.indptr = [0]
        self.start = []
        self.end = []

    def append(self, starts, ends):
        n = int(np.size(starts))
        if n:
            self.start.append(np.asarray(starts, dtype=np.int32))
            self.end.append(np.asarray(ends, dtype=np.int32))
        self.indptr.append(self.indptr[-1] + n)

    def finish(self):
        start = (
            np.concatenate(self.start)
            if self.start
            else np.zeros(0, dtype=np.int32)
        )
        end = (
            np.concatenate(self.end)
            if self.end
            else np.zeros(0, dtype=np.int32)
        )
        return np.asarray(self.indptr, dtype=np.int64), start, end


def _load_split(data_dir, split, window=common.WINDOW_SIZE):
    data_dir = Path(data_dir)
    seq_path = data_dir / f"{split}_sequence.npy"
    meta_path = data_dir / f"{split}_meta.npz"
    if not seq_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"missing supervised split {split!r} in {data_dir}: expected "
            f"{seq_path.name} and {meta_path.name}. Rebuild with "
            "scripts/make_train_data.py."
        )
    sequence = np.load(seq_path, mmap_mode="r")
    meta = np.load(meta_path)
    width = int(sequence.shape[1]) if sequence.ndim == 2 else window
    splice = SparseLabelStore(
        meta["splice_indptr"], meta["splice_col"], meta["splice_val"], width
    )
    start_stop = SparseLabelStore(
        meta["start_stop_indptr"],
        meta["start_stop_col"],
        meta["start_stop_val"],
        width,
    )
    if "trusted_indptr" not in meta:
        raise ValueError(
            f"{meta_path} has no trusted-region block. This is a V8S3/V8S4 "
            "build; V8S5 needs its own supervised data. Rebuild with "
            "scripts/make_train_data.py."
        )
    trusted = TrustedRegionStore(
        meta["trusted_indptr"], meta["trusted_start"], meta["trusted_end"], width
    )
    return sequence, splice, start_stop, trusted, meta


class SupervisedWindowDataset(Dataset):
    """Windows for V8S5: sequence, both label vectors, and the trusted mask.

    The fourth tensor is what separates V8S5 from V8S3. ``True`` means "this
    base lies inside an EviAnn CDS locus on this window's strand", i.e. the
    annotation is trustworthy here and an unlabelled candidate is a real
    negative. ``False`` means the annotation is silent and the candidate's
    target is smoothed to ``alpha`` instead of 0.
    """

    def __init__(self, data_dir, split):
        (
            self.sequence,
            self.splice,
            self.start_stop,
            self.trusted,
            _,
        ) = _load_split(data_dir, split)

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        return (
            torch.from_numpy(np.asarray(self.sequence[index]).astype(np.int64)),
            torch.from_numpy(self.splice.dense(index).astype(np.int64)),
            torch.from_numpy(self.start_stop.dense(index).astype(np.int64)),
            torch.from_numpy(self.trusted.dense(index)),
        )


class MLMDataset(Dataset):
    def __init__(self, data_dir, split):
        path = Path(data_dir) / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing MLM split: {path}")
        data = np.load(path)
        self.sequence = data["sequence"]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        return torch.from_numpy(self.sequence[index].astype(np.int64))


def candidate_counts(sequence, splice, start_stop):
    encoded = sequence
    counts = {}
    candidates = {
        "donor": (encoded[:, :-1] == common.G) & (encoded[:, 1:] == common.T),
        "acceptor": (encoded[:, :-1] == common.A) & (encoded[:, 1:] == common.G),
        "start": (
            (encoded[:, :-2] == common.A)
            & (encoded[:, 1:-1] == common.T)
            & (encoded[:, 2:] == common.G)
        ),
    }
    s0, s1, s2 = encoded[:, :-2], encoded[:, 1:-1], encoded[:, 2:]
    candidates["stop"] = (
        ((s0 == common.T) & (s1 == common.A) & (s2 == common.A))
        | ((s0 == common.T) & (s1 == common.A) & (s2 == common.G))
        | ((s0 == common.T) & (s1 == common.G) & (s2 == common.A))
    )
    positives = {
        "donor": splice == 1,
        "acceptor": splice == 2,
        "start": start_stop == 1,
        "stop": start_stop == 2,
    }
    for name, mask in candidates.items():
        full = np.zeros_like(encoded, dtype=bool)
        full[:, : mask.shape[1]] = mask
        counts[name] = {
            "candidates": int((full | positives[name]).sum()),
            "positives": int(positives[name].sum()),
        }
    return counts


def _blocked_candidate_counts(sequence, splice_store, start_stop_store):
    """candidate_counts over a memmapped split, in blocks, so RAM stays flat."""
    total = {
        name: {"candidates": 0, "positives": 0} for name in SITE_NAMES
    }
    n = len(sequence)
    for start in range(0, n, _STATS_BLOCK):
        stop = min(start + _STATS_BLOCK, n)
        block = np.asarray(sequence[start:stop])
        counts = candidate_counts(
            block,
            splice_store.dense_block(start, stop),
            start_stop_store.dense_block(start, stop),
        )
        for name, values in counts.items():
            total[name]["candidates"] += values["candidates"]
            total[name]["positives"] += values["positives"]
    return total


def _encode_chromosome(sequence):
    """Encoded chromosome, its complement, and a literal-N mask.

    Windows become slices of these arrays: the ``-`` strand window is
    ``complement[ws : ws + window][::-1]``, which is exactly
    ``encode_sequence(reverse_complement(chunk))`` but without building a
    10 kb Python string per window.
    """
    encoded = common.encode_sequence(sequence)
    raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    is_n = raw == ord("N")
    del raw
    complement = np.where(
        encoded == common.N, np.int8(common.N), (np.int8(3) - encoded)
    )
    return encoded, complement.astype(np.int8, copy=False), is_n


def _region_counts(mask, n_windows, stride, offset, span):
    """Per-window count of True in ``[k*stride + offset, ... + span)``.

    Evaluated from block sums on the coarsest grid that divides ``stride``,
    ``offset`` and ``span`` (2500 bp for the 10 kb / 5 kb / 2500 bp geometry
    used here), so a whole chromosome costs one pass over a bool array and no
    250 M-element prefix sum. The block sums are taken in chunks, because
    ``reduceat`` casts its whole input to the accumulator dtype otherwise --
    on a 250 Mb chromosome that alone would be a 2 GB temporary.
    """
    if n_windows <= 0:
        return np.zeros(0, dtype=np.int64)
    grid = math.gcd(math.gcd(int(stride), int(offset) or int(stride)), int(span))
    if grid > 0 and stride % grid == 0 and offset % grid == 0 and span % grid == 0:
        flat = np.asarray(mask).view(np.uint8)
        n_blocks = int(np.ceil(len(flat) / grid))
        block = np.zeros(n_blocks, dtype=np.int64)
        chunk_blocks = max(1, _BLOCK_SUM_CHUNK // grid)
        for first in range(0, n_blocks, chunk_blocks):
            last = min(first + chunk_blocks, n_blocks)
            lo, hi = first * grid, min(last * grid, len(flat))
            if hi <= lo:
                break
            local = np.arange(last - first, dtype=np.int64) * grid
            block[first:last] = np.add.reduceat(
                flat[lo:hi].astype(np.int32), local[local < (hi - lo)]
            )
        idx = np.arange(n_windows, dtype=np.int64) * (stride // grid) + offset // grid
        counts = np.zeros(n_windows, dtype=np.int64)
        for step in range(span // grid):
            counts += block[idx + step]
        return counts
    prefix = np.concatenate(([0], np.cumsum(np.asarray(mask), dtype=np.int64)))
    idx = np.arange(n_windows, dtype=np.int64) * stride + offset
    return prefix[idx + span] - prefix[idx]


# ---------------------------------------------------------------------------
# Trusted regions: where the EviAnn annotation is believable
# ---------------------------------------------------------------------------


def cds_locus_mask(length, ranges_by_tid):
    """Boolean per-base mask of positions covered by a CDS locus extent.

    ``ranges_by_tid`` is a ``{transcript_id: (start, end)}`` map as produced by
    ``common.compute_cds_ranges`` (0-based, half-open) for ONE strand. Coverage
    is the CDS *span* of each transcript, so introns inside a coding gene count
    as trusted: EviAnn resolved that gene's exon/intron structure, so a GT in
    its intron that was not annotated as a donor really is not one.

    The two strands are always handled independently: this is only ever called
    with a single strand's ranges, so annotation on one strand can never affect
    the trust mask on the other.
    """
    mask = np.zeros(length, dtype=bool)
    for start, end in ranges_by_tid.values():
        begin = max(0, min(int(start), length))
        stop = max(0, min(int(end), length))
        if stop > begin:
            mask[begin:stop] = True
    return mask


def cds_exon_mask(length, segments_by_tid):
    """Boolean per-base mask of CDS *segments* only (introns excluded).

    ``segments_by_tid`` is a ``{transcript_id: [(start, end), ...]}`` map as
    produced by ``common.parse_gff3`` (1-based, inclusive, GFF convention) for
    ONE strand. This is the strict reading of "intersects an EviAnn CDS" and is
    available via ``--trust-region exon``; it trusts less ground than the locus
    rule, so more candidates get the smoothed target.
    """
    mask = np.zeros(length, dtype=bool)
    for segments in segments_by_tid.values():
        for start, end in segments:
            begin = max(0, min(int(start) - 1, length))
            stop = max(0, min(int(end), length))
            if stop > begin:
                mask[begin:stop] = True
    return mask


def _strand_trust_mask(length, chrom, strand, ranges, cds, trust_region):
    if trust_region == "locus":
        return cds_locus_mask(length, ranges.get((chrom, strand), {}))
    if trust_region == "exon":
        return cds_exon_mask(length, cds.get((chrom, strand), {}))
    raise ValueError(
        f"unknown trust_region {trust_region!r}; choose from {TRUST_REGIONS}"
    )


def _mask_runs(mask):
    """Half-open ``[start, end)`` runs of True in a 1-D boolean mask."""
    flat = np.asarray(mask, dtype=bool)
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    padded = np.zeros(flat.size + 2, dtype=np.int8)
    padded[1:-1] = flat
    diff = np.diff(padded)
    del padded
    starts = np.flatnonzero(diff == 1).astype(np.int64)
    ends = np.flatnonzero(diff == -1).astype(np.int64)
    return starts, ends


def _window_trusted_intervals(run_start, run_end, win_start, window, strand):
    """Trusted intervals for one window, in that window's own orientation.

    ``run_start``/``run_end`` are ascending, non-overlapping forward-strand
    runs. For a ``-`` strand window the sequence row is
    ``complement[ws:ws+W][::-1]``, so oriented offset ``i`` is forward position
    ``ws + W - 1 - i`` and a forward interval ``[a, b)`` clipped to the window
    becomes ``[W - b', W - a')``.
    """
    win_end = win_start + window
    lo = int(np.searchsorted(run_end, win_start, side="right"))
    hi = int(np.searchsorted(run_start, win_end, side="left"))
    if hi <= lo:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    starts = np.clip(run_start[lo:hi] - win_start, 0, window).astype(np.int32)
    ends = np.clip(run_end[lo:hi] - win_start, 0, window).astype(np.int32)
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    if strand == "-":
        starts, ends = (
            (window - ends)[::-1].copy(),
            (window - starts)[::-1].copy(),
        )
    return starts, ends


# ---------------------------------------------------------------------------
# alpha: how often is an unannotated candidate actually a site?
# ---------------------------------------------------------------------------


def _oriented_motif_indices(oriented, name):
    if name == "donor":
        return common.find_dinuc(oriented, common.G, common.T)
    if name == "acceptor":
        return common.find_dinuc(oriented, common.A, common.G)
    if name == "start":
        return common.find_trinuc(oriented, common.A, common.T, common.G)
    if name == "stop":
        return np.sort(
            np.concatenate(
                [
                    common.find_trinuc(oriented, common.T, common.A, common.A),
                    common.find_trinuc(oriented, common.T, common.A, common.G),
                    common.find_trinuc(oriented, common.T, common.G, common.A),
                ]
            )
        )
    raise ValueError(f"unknown site type {name!r}")


def compute_alpha_priors(
    chroms,
    fasta,
    trust_runs,
    refseq_gff_path,
    require_canonical=True,
):
    """Per-site-type P(site | candidate), from RefSeq, on the training split.

    Two estimates per site type:

    ``global``
        positives / candidates over every candidate on the supervised
        chromosomes.
    ``conditional``
        the same ratio restricted to candidates **outside** every EviAnn CDS
        locus -- exactly the population whose target gets smoothed, so it is
        the expected value of the label there, and therefore the constant that
        a proper scoring rule is minimised by.

    ``trust_runs`` maps ``(chrom, strand)`` to the forward-strand
    ``(run_start, run_end)`` arrays already computed for the window store, so
    the trust definition can never drift between the two uses.

    Positives are extracted from the RefSeq GFF with exactly the rules the
    EviAnn labels use (CDS-internal canonical splice pairs; codon-validated
    start/stop), so numerator and denominator are commensurable. Counting runs
    over whole chromosomes rather than the window grid: N-gaps contribute no
    candidates at all, and the 2x window overlap scales numerator and
    denominator alike, so the ratio is unaffected.
    """
    transcripts, cds, gff_stats = common.parse_gff3(refseq_gff_path, chroms)
    ranges = common.compute_cds_ranges(cds)
    splice_sites, splice_stats = common.extract_cds_internal_splice_sites(
        transcripts, ranges, fasta, require_canonical=require_canonical
    )
    start_sites, stop_sites, codon_stats = common.extract_start_stop_sites(
        cds, fasta
    )

    counts = {
        name: {
            "candidates": 0,
            "positives": 0,
            "untrusted_candidates": 0,
            "untrusted_positives": 0,
        }
        for name in SITE_NAMES
    }

    for chrom in chroms:
        sequence = fasta[chrom]
        length = len(sequence)
        encoded, complement, _ = _encode_chromosome(sequence)
        for strand in "+-":
            oriented = encoded if strand == "+" else complement[::-1]
            run_start, run_end = trust_runs[(chrom, strand)]
            trusted = np.zeros(length, dtype=bool)
            for begin, stop in zip(run_start, run_end):
                trusted[int(begin) : int(stop)] = True
            if strand == "-":
                trusted = trusted[::-1]
            # Whole-chromosome label vectors in the window's own orientation.
            # Reusing make_labels_for_window with window_size=length is what
            # guarantees the coordinate convention matches the builder's.
            splice_label, start_stop_label = common.make_labels_for_window(
                0,
                splice_sites.get(chrom, {}).get(strand, set()),
                start_sites.get(chrom, {}).get("+", set()),
                stop_sites.get(chrom, {}).get("+", set()),
                start_sites.get(chrom, {}).get("-", set()),
                stop_sites.get(chrom, {}).get("-", set()),
                strand,
                length,
            )
            positive_index = {
                "donor": np.flatnonzero(splice_label == 1),
                "acceptor": np.flatnonzero(splice_label == 2),
                "start": np.flatnonzero(start_stop_label == 1),
                "stop": np.flatnonzero(start_stop_label == 2),
            }
            del splice_label, start_stop_label
            for name in SITE_NAMES:
                motif = _oriented_motif_indices(oriented, name)
                positives = positive_index[name]
                # The supervised mask is motif | positive, so a positive that is
                # not a motif match still contributes a candidate. With
                # require_canonical this set is empty -- every RefSeq positive
                # is an exact oriented motif match -- but it is counted exactly
                # rather than assumed away.
                if motif.size and positives.size:
                    hit = np.clip(
                        np.searchsorted(motif, positives), 0, motif.size - 1
                    )
                    extra = positives[motif[hit] != positives]
                else:
                    extra = positives
                counts[name]["candidates"] += int(motif.size) + int(extra.size)
                counts[name]["positives"] += int(positives.size)
                counts[name]["untrusted_candidates"] += int(
                    np.count_nonzero(~trusted[motif])
                ) + int(np.count_nonzero(~trusted[extra]))
                counts[name]["untrusted_positives"] += int(
                    np.count_nonzero(~trusted[positives])
                )
                del motif, extra
            del trusted, positive_index
        del encoded, complement

    def ratio(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    return {
        "source": "refseq",
        "gff": str(refseq_gff_path),
        "domain": "train_chromosomes_whole",
        "require_canonical": bool(require_canonical),
        "counts": counts,
        "global": {
            name: ratio(counts[name]["positives"], counts[name]["candidates"])
            for name in SITE_NAMES
        },
        "conditional": {
            name: ratio(
                counts[name]["untrusted_positives"],
                counts[name]["untrusted_candidates"],
            )
            for name in SITE_NAMES
        },
        "gff_stats": gff_stats,
        "splice_extraction": splice_stats,
        "start_stop_extraction": codon_stats,
    }


def build_supervised_profile(
    profile_name,
    fasta_path,
    gff_path,
    out_dir,
    window=common.WINDOW_SIZE,
    stride=common.STRIDE,
    require_canonical=True,
    trust_region="locus",
    refseq_gff_path=None,
):
    if trust_region not in TRUST_REGIONS:
        raise ValueError(
            f"unknown trust_region {trust_region!r}; choose from {TRUST_REGIONS}"
        )
    profile = get_profile(profile_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chroms = [chrom for chrom in common.NC_TO_NAME if chrom != profile.test_chrom]
    fasta = common.load_fasta(fasta_path, wanted=chroms)
    transcripts, cds, gff_stats = common.parse_gff3(gff_path, chroms)
    ranges = common.compute_cds_ranges(cds)
    splice_sites, splice_stats = common.extract_cds_internal_splice_sites(
        transcripts, ranges, fasta, require_canonical=require_canonical
    )
    start_sites, stop_sites, codon_stats = common.extract_start_stop_sites(cds, fasta)

    val_set = set(profile.val_chroms)

    # ---- pass 1: which windows survive, and how many per split ----------
    # v8s5 keeps every window that is not a sequence gap -- the s3 rule --
    # so the decision needs no labels at all; only the N-content rule applies.
    kept_starts = {}
    skipped_gap = 0
    counts = {"train": 0, "val": 0}
    for chrom in chroms:
        sequence = fasta[chrom]
        length = len(sequence)
        n_windows = (length - window) // stride + 1 if length >= window else 0
        raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        n_counts = _region_counts(raw == ord("N"), n_windows, stride, 0, window)
        del raw
        keep = n_counts <= window * 0.5
        starts = np.flatnonzero(keep).astype(np.int64) * stride
        split = "val" if chrom in val_set else "train"
        for strand in "+-":
            kept_starts[(chrom, strand)] = starts
            counts[split] += int(starts.size)
            skipped_gap += int(n_windows - starts.size)

    # ---- trusted-region runs, one pass per chromosome and strand --------
    # Computed once and reused for both the per-window store and the alpha
    # estimate, so the two can never disagree about what "trusted" means.
    trust_runs = {}
    trusted_bases = 0
    total_bases = 0
    for chrom in chroms:
        length = len(fasta[chrom])
        total_bases += 2 * length
        for strand in "+-":
            mask = _strand_trust_mask(
                length, chrom, strand, ranges, cds, trust_region
            )
            trusted_bases += int(np.count_nonzero(mask))
            trust_runs[(chrom, strand)] = _mask_runs(mask)
            del mask

    # ---- allocate the on-disk sequence arrays ---------------------------
    arrays = {}
    for split in ("train", "val"):
        arrays[split] = np.lib.format.open_memmap(
            out_dir / f"{split}_sequence.npy",
            mode="w+",
            dtype=np.int8,
            shape=(counts[split], window),
        )

    # ---- pass 2: fill sequences, accumulate sparse labels + intervals ---
    cursors = {"train": 0, "val": 0}
    splice_builders = {split: _SparseLabelBuilder() for split in ("train", "val")}
    start_stop_builders = {split: _SparseLabelBuilder() for split in ("train", "val")}
    trusted_builders = {split: _IntervalBuilder() for split in ("train", "val")}
    meta_chrom = {split: [] for split in ("train", "val")}
    meta_strand = {split: [] for split in ("train", "val")}
    meta_start = {split: [] for split in ("train", "val")}
    empty_windows = 0
    fully_untrusted_windows = 0

    for chrom in chroms:
        sequence = fasta[chrom]
        encoded, complement, _ = _encode_chromosome(sequence)
        split = "val" if chrom in val_set else "train"
        target = arrays[split]
        for strand in "+-":
            sites = splice_sites.get(chrom, {}).get(strand, set())
            plus_starts = start_sites.get(chrom, {}).get("+", set())
            plus_stops = stop_sites.get(chrom, {}).get("+", set())
            minus_starts = start_sites.get(chrom, {}).get("-", set())
            minus_stops = stop_sites.get(chrom, {}).get("-", set())
            run_start, run_end = trust_runs[(chrom, strand)]
            for win_start in kept_starts[(chrom, strand)]:
                win_start = int(win_start)
                splice, start_stop = common.make_labels_for_window(
                    win_start,
                    sites,
                    plus_starts,
                    plus_stops,
                    minus_starts,
                    minus_stops,
                    strand,
                    window,
                )
                # v8s5: empty windows are kept (the s3 rule) so the model sees
                # the true background base rate -- but their candidates are now
                # smoothed to alpha wherever the annotation is silent.
                if splice.sum() == 0 and start_stop.sum() == 0:
                    empty_windows += 1
                interval_start, interval_end = _window_trusted_intervals(
                    run_start, run_end, win_start, window, strand
                )
                if interval_start.size == 0:
                    fully_untrusted_windows += 1
                row = cursors[split]
                if strand == "+":
                    target[row] = encoded[win_start : win_start + window]
                else:
                    target[row] = complement[win_start : win_start + window][::-1]
                splice_builders[split].append(splice)
                start_stop_builders[split].append(start_stop)
                trusted_builders[split].append(interval_start, interval_end)
                meta_chrom[split].append(chrom)
                meta_strand[split].append(strand)
                meta_start[split].append(win_start)
                cursors[split] = row + 1
        del encoded, complement

    split_stats = {}
    for split in ("train", "val"):
        arrays[split].flush()
        if cursors[split] != counts[split]:
            raise RuntimeError(
                f"{split}: wrote {cursors[split]} windows, expected {counts[split]}"
            )
        splice_indptr, splice_col, splice_val = splice_builders[split].finish()
        ss_indptr, ss_col, ss_val = start_stop_builders[split].finish()
        tr_indptr, tr_start, tr_end = trusted_builders[split].finish()
        np.savez_compressed(
            out_dir / f"{split}_meta.npz",
            splice_indptr=splice_indptr,
            splice_col=splice_col,
            splice_val=splice_val,
            start_stop_indptr=ss_indptr,
            start_stop_col=ss_col,
            start_stop_val=ss_val,
            trusted_indptr=tr_indptr,
            trusted_start=tr_start,
            trusted_end=tr_end,
            chrom=np.asarray(meta_chrom[split]),
            strand=np.asarray(meta_strand[split], dtype="U1"),
            window_start=np.asarray(meta_start[split], dtype=np.int64),
            window=np.asarray(window),
        )
        del arrays[split]
        sequence, splice_store, ss_store, trusted_store, _ = _load_split(
            out_dir, split, window
        )
        covered = trusted_store.covered_per_row()
        split_stats[split] = {
            "windows": int(len(sequence)),
            "non_empty_windows": int(
                np.count_nonzero(
                    splice_store.nnz_per_row() + ss_store.nnz_per_row()
                )
            ),
            "windows_with_trusted_bases": int(np.count_nonzero(covered)),
            "trusted_base_fraction": float(
                covered.sum() / max(len(sequence) * window, 1)
            ),
            "candidate_counts": _blocked_candidate_counts(
                sequence, splice_store, ss_store
            ),
            "chroms": sorted(
                {common.NC_TO_NAME.get(chrom, chrom) for chrom in set(meta_chrom[split])}
            ),
        }
        del sequence, splice_store, ss_store, trusted_store

    alpha_priors = None
    if refseq_gff_path is not None:
        # Train chromosomes only: the four validation chromosomes drive
        # checkpoint selection, so keeping them out of alpha keeps the
        # deliberate RefSeq leakage confined to the training split.
        alpha_chroms = [chrom for chrom in chroms if chrom not in val_set]
        alpha_priors = compute_alpha_priors(
            alpha_chroms,
            fasta,
            trust_runs,
            refseq_gff_path,
            require_canonical=require_canonical,
        )
        alpha_priors["trust_region"] = trust_region
        alpha_priors["chroms"] = [
            common.NC_TO_NAME.get(chrom, chrom) for chrom in alpha_chroms
        ]

    stats = {
        "profile": profile.name,
        "n_train_windows": counts["train"],
        "n_val_windows": counts["val"],
        "expected_train_windows": profile.expected_train,
        "expected_val_windows": profile.expected_val,
        "window": window,
        "stride": stride,
        "skip_empty_windows": False,
        "supervised_format": SUPERVISED_FORMAT,
        "label_strategy": "asymmetric_smoothing",
        "trust_region": trust_region,
        "strand_handling": "independent",
        "require_canonical": require_canonical,
        "val_chroms": list(profile.val_chroms),
        "test_chrom": profile.test_chrom or None,
        "test_status": profile.test_status,
        "splits": split_stats,
        "empty_windows": empty_windows,
        "fully_untrusted_windows": fully_untrusted_windows,
        "trusted_genome_fraction": float(trusted_bases / max(total_bases, 1)),
        "skipped_gap_windows": skipped_gap,
        "alpha_priors": alpha_priors,
        "gff": gff_stats,
        "splice_extraction": splice_stats,
        "start_stop_extraction": codon_stats,
    }
    with open(out_dir / "stats.json", "w") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def build_mlm_profile(profile_name, fasta_path, out_dir, window=common.WINDOW_SIZE, stride=common.STRIDE):
    profile = get_profile(profile_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chroms = []
    for chrom in common.NC_TO_NAME:
        if chrom in profile.val_chroms:
            continue
        if chrom == profile.test_chrom:
            # chr1_test: the test chromosome is never seen by pretraining.
            # with_chr1: test_chrom is "" and matches nothing, so chr1 stays
            # in the MLM corpus -- the analogue of the Dmel with_x profile's
            # pretrain_includes_chr_x=True.
            continue
        chroms.append(chrom)
    val_chroms = list(profile.val_chroms)
    fasta = common.load_fasta(fasta_path, wanted=chroms + val_chroms)

    def collect(target_chroms):
        rows, meta = [], []
        skipped_gap = 0
        for chrom in target_chroms:
            sequence = fasta[chrom]
            for strand in "+-":
                for win_start in range(0, len(sequence) - window + 1, stride):
                    chunk = sequence[win_start : win_start + window]
                    if len(chunk) != window or chunk.count("N") > window * 0.5:
                        skipped_gap += 1
                        continue
                    oriented = chunk if strand == "+" else common.reverse_complement(chunk)
                    rows.append(common.encode_sequence(oriented))
                    meta.append((chrom, strand, win_start))
        return np.stack(rows).astype(np.int8), meta, skipped_gap

    train, train_meta, train_gap = collect(chroms)
    val, val_meta, val_gap = collect(val_chroms)
    for split, arr, meta in (("train", train, train_meta), ("val", val, val_meta)):
        np.savez_compressed(
            out_dir / f"{split}.npz",
            sequence=arr,
            chrom=np.asarray([row[0] for row in meta]),
            strand=np.asarray([row[1] for row in meta], dtype="U1"),
            window_start=np.asarray([row[2] for row in meta], dtype=np.int64),
        )
    stats = {
        "profile": profile.name,
        "window": window,
        "stride": stride,
        "n_train_windows": int(len(train)),
        "n_val_windows": int(len(val)),
        "train_chroms": [common.NC_TO_NAME.get(chrom, chrom) for chrom in chroms],
        "val_chroms": [common.NC_TO_NAME.get(chrom, chrom) for chrom in val_chroms],
        "test_chrom_excluded": bool(profile.test_chrom),
        "skipped_gap_windows": int(train_gap + val_gap),
    }
    with open(out_dir / "stats.json", "w") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def validate_supervised(data_dir, profile_name):
    profile = get_profile(profile_name)
    errors = []
    report = {"profile": profile.name, "format": SUPERVISED_FORMAT, "splits": {}}
    for split, expected in (("train", profile.expected_train), ("val", profile.expected_val)):
        sequence, splice_store, ss_store, trusted_store, meta = _load_split(
            data_dir, split
        )
        chroms = set(meta["chrom"].tolist())
        nnz = splice_store.nnz_per_row() + ss_store.nnz_per_row()
        n_non_empty = int(np.count_nonzero(nnz))
        covered = trusted_store.covered_per_row()
        width = int(sequence.shape[1])
        report["splits"][split] = {
            "windows": int(len(sequence)),
            "non_empty_windows": n_non_empty,
            "expected_non_empty_windows": expected,
            "empty_windows": int(len(sequence) - n_non_empty),
            "windows_with_trusted_bases": int(np.count_nonzero(covered)),
            "trusted_base_fraction": float(
                covered.sum() / max(len(sequence) * width, 1)
            ),
            "chroms": sorted(common.NC_TO_NAME.get(chrom, chrom) for chrom in chroms),
        }
        if len(sequence) != len(splice_store) or len(sequence) != len(ss_store):
            errors.append(f"{split}: sequence and label row counts disagree")
        if len(sequence) != len(trusted_store):
            errors.append(f"{split}: sequence and trusted-region row counts disagree")
        if trusted_store.start.size:
            if int(trusted_store.start.min()) < 0 or int(trusted_store.end.max()) > width:
                errors.append(f"{split}: trusted interval outside the window")
            if int((trusted_store.end <= trusted_store.start).sum()):
                errors.append(f"{split}: empty or inverted trusted interval")
        # v8s5 uses the s3 window rule: empty windows are intentionally kept,
        # and the number of non-empty (positive-containing) windows should match
        # the s2 labelled count exactly. expected is None for Hsap until a first
        # build establishes it.
        if expected is not None and n_non_empty != expected:
            errors.append(
                f"{split}: expected {expected} non-empty windows, found {n_non_empty}"
            )
        if split == "train" and profile.test_chrom and profile.test_chrom in chroms:
            errors.append("train includes the test chromosome")
        if (
            split == "train"
            and not profile.test_chrom
            and TEST_ACCESSION not in chroms
        ):
            # The training-exposed profile has to actually expose chr1;
            # otherwise it is a slower duplicate of chr1_test wearing the
            # wrong label, and every number it produces is mislabeled.
            errors.append(f"{profile.name}: train does not include chr1")
        if split == "val" and not chroms <= set(profile.val_chroms):
            errors.append("validation split contains non-validation chromosomes")
    stats_path = Path(data_dir) / "stats.json"
    if stats_path.exists():
        stats = json.load(open(stats_path))
        report["alpha_priors"] = stats.get("alpha_priors")
        if stats.get("alpha_priors") is None:
            errors.append(
                "stats.json has no alpha_priors block; rebuild with a "
                "--refseq-gff or pass --alpha-mode fixed at training time"
            )
    report["ok"] = not errors
    report["errors"] = errors
    return report


def validate_mlm(data_dir, profile_name):
    profile = get_profile(profile_name)
    errors = []
    report = {"profile": profile.name, "splits": {}}
    for split in ("train", "val"):
        data = np.load(Path(data_dir) / f"{split}.npz")
        chroms = set(data["chrom"].tolist())
        report["splits"][split] = {
            "windows": int(len(data["sequence"])),
            "chroms": sorted(common.NC_TO_NAME.get(chrom, chrom) for chrom in chroms),
        }
        if split == "train":
            if chroms & set(profile.val_chroms):
                errors.append("MLM train includes validation chromosomes")
            if profile.test_chrom and profile.test_chrom in chroms:
                errors.append("MLM train includes the test chromosome")
            if not profile.test_chrom and TEST_ACCESSION not in chroms:
                errors.append(
                    f"{profile.name}: MLM train does not include chr1"
                )
        elif not chroms <= set(profile.val_chroms):
            errors.append("MLM val contains non-validation chromosomes")
    report["ok"] = not errors
    report["errors"] = errors
    return report
