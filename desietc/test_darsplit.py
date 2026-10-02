"""Self-contained checks for darsplit.py.

Reference numbers are frozen from the validated telemetry_mining analysis
(analysis/observing_limits/dar_split_plan.py + drift_coeff_check.py, which back docs/DAR_SPLIT_PLAN.md);
darsplit.py reproduces them bit-for-bit when loading the same fiberpos snapshot. This test hardcodes them
so it needs nothing outside the desietc package. Run: `python -m pytest test_darsplit.py` or
`python test_darsplit.py`.
"""
import numpy as np
from desietc import darsplit as D


def _close(a, b, rtol=1e-4, atol=0.0):
    assert abs(a - b) <= atol + rtol * abs(b), f"{a} != {b} (rtol={rtol})"


def test_layout_loaded():
    assert D.NPOS == 5000
    assert D._PETAL.min() == 0 and D._PETAL.max() == 9
    assert len(D._XIP) == D.NPOS == len(D._DEVICE)


def test_airmass():
    _close(D.airmass(0.0, 0.0), 1.178708)
    _close(D.airmass(2.0, -30.0), 2.691058)
    assert D.airmass(0.0, 89.0) > 0                       # near-zenith finite
    assert not np.isfinite(D.airmass(9.0, -80.0))         # below horizon -> inf


def test_sigma_and_field_seeing():
    _close(D.sigma(1.1), 51.0, rtol=0)                    # reference field seeing -> SIG0 exactly
    _close(D.sigma(1.3), 60.272727)
    _close(D.field_seeing(1.1, 1.0), 1.1, rtol=0)         # X=1 -> zenith == field
    _close(D.field_seeing(1.0, 2.691058), 1.0 * 2.691058 ** 0.3)


def test_v_p99_and_binding_fiber():
    _close(D.v_p99(1.0, -30.0, 0.99), 0.040145)
    _close(D.v_p99(0.0, 0.0, 0.99), 0.004024)
    # drift grows off-transit and toward the south
    assert D.v_p99(2.0, -30.0) > D.v_p99(0.0, -30.0)
    assert D.v_p99(2.0, -30.0) > D.v_p99(2.0, 0.0)
    # binding fiber names a real hardware location with rate ~ v_p99
    rate, petal, device = D.binding_fiber(2.0, -30.0, 0.99)
    assert (petal, device) == (4, 487)
    assert 0 <= petal <= 9
    _close(rate, D.v_p99(2.0, -30.0, 0.99), rtol=0.05)


def test_drift_coeff_and_tau_star():
    # frozen from drift_coeff_check.py (p=0.95, fixed sigma=51um at 1.1")
    _close(D.drift_coeff(1.0, -30.0, 1.1, 0.95), 4.2291e-08, rtol=1e-3)
    _close(D.tau_star(1.0, -30.0, 1.1, 0.95), 1123.6616)
    _close(D.tau_star(0.0, 0.0, 1.1, 0.95), 4953.3194)
    # tau* longer when drift is smaller (near transit / worse seeing widens sigma)
    assert D.tau_star(0.0, 0.0, 1.1) > D.tau_star(2.0, -30.0, 1.1)
    assert D.tau_star(2.0, -30.0, 1.3) > D.tau_star(2.0, -30.0, 1.1)


def test_dar_split_time_capped():
    # near transit tau* is huge -> capped at T_CAP; far south it bites
    assert D.dar_split_time(0.0, 0.0, 1.1) == D.T_CAP
    st = D.dar_split_time(2.5, -30.0, 1.1)
    assert st == min(D.T_CAP, D.tau_star(2.5, -30.0, 1.1))


def test_flux_retention():
    _close(D.fret(0.0, 1.1), 1.0, rtol=0)
    _close(D.fret(50.0, 1.1), 0.665)                       # curve value at 50um, 1.1"
    assert D.fret(80.0, 1.1) < D.fret(40.0, 1.1)           # monotone in offset
    # f2mean: 1 for no drift, <=1, decreasing in dmax
    assert D.f2mean(0.0, 1.1) == 1.0
    _close(D.f2mean(60.0, 1.1), 0.702427)
    assert D.f2mean(80.0, 1.1) < D.f2mean(30.0, 1.1) <= 1.0


def test_maxdrift_mode():
    # split when binding fiber reaches max_drift; midpoint tolerates 2x
    v = D.v_p99(2.0, -30.0)
    _close(D.dar_split_time_maxdrift(2.0, -30.0, 30.0, midpoint=True), min(D.T_CAP, 2 * 30.0 / v))
    _close(D.dar_split_time_maxdrift(2.0, -30.0, 30.0, midpoint=False), min(D.T_CAP, 30.0 / v))


def test_set_and_reset_positions():
    v0 = D.v_p99(2.0, -30.0); n0 = D.NPOS
    x, y, pl, dv = D._STATIC
    try:
        D.set_positions(x[:100], y[:100], pl[:100], dv[:100])   # inject a subset
        assert D.NPOS == 100 and np.isfinite(D.v_p99(2.0, -30.0))
        D.set_positions(x, y, pl, dv)                            # inject the same static positions
        _close(D.v_p99(2.0, -30.0), v0, rtol=0)                  # -> reproduces static exactly
    finally:
        D.reset_positions()
    assert D.NPOS == n0
    _close(D.v_p99(2.0, -30.0), v0, rtol=0)                      # restored from the startup cache


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("all darsplit checks passed")
