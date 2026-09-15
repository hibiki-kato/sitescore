import numpy as np
from sitescore import platt
from sitescore.calibration import apply, collect
from sitescore.interface import SiteScore


def test_platt_recovers_shift():
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000)
    y = rng.random(20000) < platt.sigmoid(2.0 * z - 1.0)   # true map a=2, b=-1
    a, b, info = platt.fit_platt(z, y)
    assert abs(a - 2.0) < 0.15 and abs(b + 1.0) < 0.15 and info["converged"]


def test_collect_and_apply():
    scores = [SiteScore("c", 5, "+", "donor", "GT", 0.9), SiteScore("c", 9, "+", "donor", "GT", 0.2)]
    p, y = collect(scores, {("+", "donor"): {5}})["donor"]
    assert list(y) == [True, False] and list(p) == [0.9, 0.2]
    out = list(apply(scores, {"donor": (1.0, 0.0)}))
    assert abs(out[0].prob - 0.9) < 1e-9
