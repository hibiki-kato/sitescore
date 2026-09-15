from sitescore.models.convmamba.adapter import _resolve_alpha, choose_val_chroms


def test_choose_val_holds_out_smallest_until_fraction():
    lengths = {"a": 100, "b": 10, "c": 20, "d": 70}
    assert choose_val_chroms(lengths, 0.15) == ["b", "c"]
    assert choose_val_chroms({"a": 5, "b": 5}, 0.9) == ["a"]     # never empties train


def test_resolve_alpha_precedence():
    hp = {"alpha": None, "refseq_gff": None}
    assert _resolve_alpha(hp, None, {})["donor"] == 0.0
    init = {"alpha": {"donor": 0.01, "acceptor": 0.02, "start": 0.03, "stop": 0.04}}
    assert _resolve_alpha(hp, init, {})["stop"] == 0.04
    assert _resolve_alpha({**hp, "alpha": 0.5}, init, {})["donor"] == 0.5
