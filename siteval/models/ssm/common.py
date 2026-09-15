"""Sequence, annotation, coordinate, and window helpers.

The coordinate rules intentionally mirror the SSM and V6 pipelines. This copy
keeps V8S2 independent without importing or modifying either existing project.
"""

import bisect
import gzip
from collections import defaultdict

import numpy as np


NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
A, C, G, T, N = 0, 1, 2, 3, 4
STOP_CODONS_PLUS = {"TAA", "TAG", "TGA"}
STOP_CODONS_MINUS = {"TTA", "CTA", "TCA"}

WINDOW_SIZE = 10000
STRIDE = 5000
CENTER_HALF = 2500

# GRCh38.p14 primary-assembly nuclear chromosomes. chrMT (NC_012920.1) and
# unplaced NT_ scaffolds are deliberately absent: nothing outside this map is
# used for training, validation, or testing.
NC_TO_NAME = {
    "NC_000001.11": "1",
    "NC_000002.12": "2",
    "NC_000003.12": "3",
    "NC_000004.12": "4",
    "NC_000005.10": "5",
    "NC_000006.12": "6",
    "NC_000007.14": "7",
    "NC_000008.11": "8",
    "NC_000009.12": "9",
    "NC_000010.11": "10",
    "NC_000011.10": "11",
    "NC_000012.12": "12",
    "NC_000013.11": "13",
    "NC_000014.9": "14",
    "NC_000015.10": "15",
    "NC_000016.10": "16",
    "NC_000017.11": "17",
    "NC_000018.10": "18",
    "NC_000019.10": "19",
    "NC_000020.11": "20",
    "NC_000021.9": "21",
    "NC_000022.11": "22",
    "NC_000023.11": "X",
    "NC_000024.10": "Y",
}


def encode_sequence(seq):
    table = np.full(256, N, dtype=np.int8)
    for base, idx in NUC_TO_IDX.items():
        table[ord(base)] = idx
        table[ord(base.lower())] = idx
    raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return table[raw]


def reverse_complement(seq):
    return "".join(COMPLEMENT.get(c, "N") for c in reversed(seq.upper()))


def load_fasta(path, wanted=None):
    wanted = set(wanted) if wanted is not None else None
    seqs = {}
    cur_id = None
    keep = False
    chunks = []
    opener = gzip.open if str(path).endswith(".gz") else open
    kwargs = {"mode": "rt"} if opener is gzip.open else {}
    with opener(path, **kwargs) as handle:
        for line in handle:
            if line.startswith(">"):
                if cur_id is not None and keep:
                    seqs[cur_id] = "".join(chunks).upper()
                cur_id = line[1:].split()[0]
                keep = wanted is None or cur_id in wanted
                chunks = []
            elif keep:
                chunks.append(line.strip())
        if cur_id is not None and keep:
            seqs[cur_id] = "".join(chunks).upper()
    return seqs


def parse_attrs(attr_string):
    attrs = {}
    for item in attr_string.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            attrs[key.strip()] = value.strip()
    return attrs


def parse_gff3(gff_path, chrom_set):
    chrom_set = set(chrom_set)
    mrna_info = {}
    transcripts = defaultdict(lambda: defaultdict(list))
    cds = defaultdict(lambda: defaultdict(list))
    n_mrna = n_exons = n_cds = 0

    with open(gff_path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "mRNA" or fields[0] not in chrom_set:
                continue
            tid = parse_attrs(fields[8]).get("ID")
            if tid:
                mrna_info[tid] = (fields[0], fields[6])
                n_mrna += 1

    with open(gff_path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in chrom_set:
                continue
            kind = fields[2]
            if kind not in {"exon", "CDS"}:
                continue
            parents = parse_attrs(fields[8]).get("Parent", "")
            for tid in parents.split(","):
                tid = tid.strip()
                if tid not in mrna_info:
                    continue
                chrom, strand = mrna_info[tid]
                interval = (int(fields[3]), int(fields[4]))
                if kind == "exon":
                    transcripts[(chrom, strand)][tid].append(interval)
                    n_exons += 1
                else:
                    cds[(chrom, strand)][tid].append(interval)
                    n_cds += 1
    return transcripts, cds, {
        "n_mrna": n_mrna,
        "n_exons": n_exons,
        "n_cds": n_cds,
    }


def compute_cds_ranges(cds):
    ranges = defaultdict(dict)
    for key, tx_dict in cds.items():
        for tid, segments in tx_dict.items():
            if segments:
                ranges[key][tid] = (
                    min(start for start, _ in segments) - 1,
                    max(end for _, end in segments),
                )
    return ranges


def extract_cds_internal_splice_sites(
    transcripts, cds_ranges, fasta_seqs, require_canonical=True
):
    sites = defaultdict(lambda: {"+": set(), "-": set()})
    n_total = n_kept = n_noncanon = 0
    for (chrom, strand), tx_dict in transcripts.items():
        ranges = cds_ranges.get((chrom, strand), {})
        seq = fasta_seqs.get(chrom)
        chrom_len = len(seq) if seq is not None else 0
        for tid, exons in tx_dict.items():
            if tid not in ranges:
                continue
            cds_lo, cds_hi = ranges[tid]
            ordered = sorted(exons)
            for (_, left_end), (right_start, _) in zip(ordered, ordered[1:]):
                if right_start <= left_end:
                    continue
                intron_start = left_end
                intron_end = right_start - 2
                if intron_end - intron_start < 1:
                    continue
                n_total += 1
                donor = intron_start
                acceptor = intron_end - 1
                if not (cds_lo <= donor < cds_hi and cds_lo <= acceptor < cds_hi):
                    continue
                if require_canonical:
                    if seq is None or donor < 0 or acceptor + 2 > chrom_len:
                        n_noncanon += 1
                        continue
                    d2 = seq[donor : donor + 2]
                    a2 = seq[acceptor : acceptor + 2]
                    canonical = (
                        d2 == "GT" and a2 == "AG"
                        if strand == "+"
                        else d2 == "CT" and a2 == "AC"
                    )
                    if not canonical:
                        n_noncanon += 1
                        continue
                sites[chrom][strand].add((donor, acceptor))
                n_kept += 1
    return sites, {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_noncanon": n_noncanon,
    }


def extract_start_stop_sites(cds, fasta_seqs):
    starts = defaultdict(lambda: {"+": set(), "-": set()})
    stops = defaultdict(lambda: {"+": set(), "-": set()})
    stats = {
        "n_start": 0,
        "n_stop": 0,
        "n_start_skip": 0,
        "n_stop_skip": 0,
    }
    for (chrom, strand), tx_dict in cds.items():
        seq = fasta_seqs.get(chrom)
        if seq is None:
            continue
        for segments in tx_dict.values():
            if not segments:
                continue
            ordered = sorted(segments)
            start = ordered[0][0] - 1 if strand == "+" else ordered[-1][1] - 3
            stop = ordered[-1][1] - 3 if strand == "+" else ordered[0][0] - 1
            if 0 <= start <= len(seq) - 3:
                codon = seq[start : start + 3]
                valid = codon == ("ATG" if strand == "+" else "CAT")
                if valid:
                    starts[chrom][strand].add(start)
                    stats["n_start"] += 1
                else:
                    stats["n_start_skip"] += 1
            if 0 <= stop <= len(seq) - 3:
                codon = seq[stop : stop + 3]
                valid = codon in (
                    STOP_CODONS_PLUS if strand == "+" else STOP_CODONS_MINUS
                )
                if valid:
                    stops[chrom][strand].add(stop)
                    stats["n_stop"] += 1
                else:
                    stats["n_stop_skip"] += 1
    return starts, stops, stats


def make_labels_for_window(
    win_start,
    sites,
    plus_starts,
    plus_stops,
    minus_starts,
    minus_stops,
    strand,
    window_size=WINDOW_SIZE,
):
    splice = np.zeros(window_size, dtype=np.int8)
    start_stop = np.zeros(window_size, dtype=np.int8)
    for donor, acceptor in sites:
        if strand == "+":
            for pos, label in ((donor, 1), (acceptor, 2)):
                offset = pos - win_start
                if 0 <= offset < window_size:
                    splice[offset] = label
        else:
            acceptor_rc = window_size - 1 - ((donor + 1) - win_start)
            donor_rc = window_size - 1 - ((acceptor + 1) - win_start)
            if 0 <= acceptor_rc < window_size:
                splice[acceptor_rc] = 2
            if 0 <= donor_rc < window_size:
                splice[donor_rc] = 1

    if strand == "+":
        for positions, label in ((plus_starts, 1), (plus_stops, 2)):
            for pos in positions:
                offset = pos - win_start
                if 0 <= offset < window_size:
                    start_stop[offset] = label
    else:
        for positions, label in ((minus_starts, 1), (minus_stops, 2)):
            for pos in positions:
                offset = pos + 2 - win_start
                if 0 <= offset < window_size:
                    start_stop[offset] = label
        start_stop = start_stop[::-1].copy()
    return splice, start_stop


def make_windows(length, win=WINDOW_SIZE, stride=STRIDE, center_half=CENTER_HALF):
    if stride != 2 * center_half:
        raise ValueError(
            "stride must equal 2 * center_half for gap-free center tiling"
        )
    starts = list(range(0, max(1, length - win + 1), stride))
    last_start = max(0, length - win)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    centers = [
        start + (min(start + win, length) - start) / 2 for start in starts
    ]
    boundaries = [0]
    boundaries.extend(
        int(round((left + right) / 2))
        for left, right in zip(centers, centers[1:])
    )
    boundaries.append(length)
    plan = []
    for index, start in enumerate(starts):
        end = min(start + win, length)
        actual_len = end - start
        pred_lo = boundaries[index] - start
        pred_hi = boundaries[index + 1] - start
        if not (0 <= pred_lo <= pred_hi <= actual_len):
            raise ValueError(
                f"invalid prediction interval for window at {start}: "
                f"{pred_lo}:{pred_hi} of {actual_len}"
            )
        plan.append(
            {
                "k": index,
                "win_start": start,
                "win_end": end,
                "actual_len": actual_len,
                "pred_lo": pred_lo,
                "pred_hi": pred_hi,
            }
        )
    return plan


def window_coverage(length, plan):
    coverage = np.zeros(length, dtype=np.int16)
    for window in plan:
        lo = window["win_start"] + window["pred_lo"]
        hi = window["win_start"] + window["pred_hi"]
        coverage[lo:hi] += 1
    return coverage


def find_dinuc(encoded, first, second):
    return np.flatnonzero((encoded[:-1] == first) & (encoded[1:] == second))


def find_trinuc(encoded, first, second, third):
    return np.flatnonzero(
        (encoded[:-2] == first)
        & (encoded[1:-1] == second)
        & (encoded[2:] == third)
    )


def find_candidates(encoded):
    stop = np.sort(
        np.concatenate(
            [
                find_trinuc(encoded, T, A, A),
                find_trinuc(encoded, T, A, G),
                find_trinuc(encoded, T, G, A),
            ]
        )
    )
    return {
        "donor": find_dinuc(encoded, G, T),
        "acceptor": find_dinuc(encoded, A, G),
        "start": find_trinuc(encoded, A, T, G),
        "stop": stop,
    }


def frame_to_genomic_1based(strand, frame_pos, chrom_length):
    return int(frame_pos) + 1 if strand == "+" else int(chrom_length - frame_pos)


def genomic_1based_to_frame(strand, genomic_pos, chrom_length):
    return int(genomic_pos) - 1 if strand == "+" else int(chrom_length - genomic_pos)


def true_sites_genomic(splice_sites, start_sites, stop_sites, chrom, chrom_length):
    output = {
        (strand, kind): set()
        for strand in "+-"
        for kind in ("donor", "acceptor", "start", "stop")
    }
    for strand in "+-":
        for donor, acceptor in splice_sites.get(chrom, {}).get(strand, set()):
            if strand == "+":
                donor_frame, acceptor_frame = donor, acceptor
            else:
                acceptor_frame = chrom_length - 1 - (donor + 1)
                donor_frame = chrom_length - 1 - (acceptor + 1)
            output[(strand, "donor")].add(
                frame_to_genomic_1based(strand, donor_frame, chrom_length)
            )
            output[(strand, "acceptor")].add(
                frame_to_genomic_1based(strand, acceptor_frame, chrom_length)
            )
    for kind, table in (("start", start_sites), ("stop", stop_sites)):
        for strand in "+-":
            for pos in table.get(chrom, {}).get(strand, set()):
                frame_pos = pos if strand == "+" else chrom_length - 1 - (pos + 2)
                output[(strand, kind)].add(
                    frame_to_genomic_1based(strand, frame_pos, chrom_length)
                )
    return output


def positions_in_window(sorted_positions, start, end):
    lo = bisect.bisect_left(sorted_positions, start)
    hi = bisect.bisect_left(sorted_positions, end)
    return sorted_positions[lo:hi]
