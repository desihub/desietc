"""DAR-driven split cadence, deflated-t_eff acceptance, and the p-percentile binding fiber.

Dependency-free physics helper for the ETC (numpy + stdlib only, no DOSlib/Pyro), so it imports
cleanly in the offline replay harness. It is a faithful port of the validated analysis code in
telemetry_mining (analysis/observing_limits/dar_split_plan.py and drift_coeff_check.py) that backs
docs/DAR_SPLIT_PLAN.md -- the refraction geometry is inlined there (Steve Kent distort.py, kappa=45"),
not imported, so nothing external is needed here either. test_darsplit.py checks this module reproduces
the analysis reference numbers.

WHAT IT PROVIDES (for the current pointing + conditions):
  - v_p99(HA, Dec)            p-percentile of the per-fiber DAR drift RATE (um/s), exact astrometry.
  - binding_fiber(HA, Dec)    that fiber's (rate, petal_loc, device_loc) -- fixed focal-plane hardware
                              geometry, NOT a per-tile fiberassign, so a static positioner layout is used.
  - sigma(field_seeing)       fibre-acceptance width (um) from the point-source offset curve.
  - dar_split_time(...)       min(T_CAP, tau*), tau* = (Delta/2a)^(1/3), a = v_p99^2/(12 sigma^2).
  - f2mean(dmax, field_seeing)  time-integrated <f>^2 (deflation factor) over a segment; <=1, ->1 short.
  - fret(delta, field_seeing)   instantaneous flux retention at offset delta (um).

SEEING CONVENTION (load-bearing -- read this before wiring):
  Every `field_seeing` argument here is the DELIVERED FWHM (arcsec) at the CURRENT airmass, i.e. the
  seeing actually on the focal plane -- the same quantity the offset curve was measured against and that
  sigma scales with. If you instead hold the ZENITH seeing, convert with field_seeing(zenith, X) =
  zenith * X**0.3 first. Do NOT pass zenith seeing directly. When wiring etc.py, confirm whether
  `self.seeing` is delivered (pass through) or zenith (convert).

BINDING FIBER: v_p99 (the RATE used for tau*/f2) is np.quantile at P for continuity with the validated
numbers; binding_fiber() returns an ACTUAL fiber (rank ceil(P*(N-1))) for the (petal_loc, device_loc)
report. The two rates are numerically ~identical; they are computed slightly differently on purpose
(a value vs. a concrete hardware location).
"""
import os
import numpy as np

# ----- constants / geometry (from the validated analysis; do not change without re-validating) -----
R       = 45 / 206265.0          # DAR field-differential coefficient (kappa = 45", radians)
PHI     = np.radians(31.9634)    # KPNO latitude
OMEGA   = 15.041 / 3600.0        # sidereal HA rate, deg/sec
UM      = 71.3                   # focal-plane plate scale, um per arcsec
MM2DEG  = 1.6 / 410.0            # mm -> deg on the sky (field radius 1.6 deg at 410 mm)
SIG0    = 51.0                   # fibre-acceptance sigma (um) at the 1.1" reference field seeing
AEXP    = 0.3                    # field-seeing airmass exponent: s_field = s_zenith * X**0.3 (MEASURED)
P       = 0.99                   # default binding percentile (pUniformity default)
DELTA   = 120.0                  # split overhead (s)
T_CAP   = 1500.0                 # cosmic-ray split cap (s)
S_REF   = 1.1                    # reference field seeing (arcsec) that SIG0 corresponds to

_FIBERPOS = os.path.join(os.path.dirname(__file__), "data", "fiberpos-desi.dat")


# ----- static positioner layout (loaded once at import) -----
def _load_fiberpos(path=_FIBERPOS):
    """Read the static fiberpos snapshot; return x,y (mm) and petal_loc,device_loc for DEVICE_TYPE==POS.

    Columns: PETAL_LOC DEVICE_LOC FVC_ID FLAGS DEVICE_ID DEVICE_TYPE X Y.
    """
    x, y, petal, device = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if p[5] == "POS":
                petal.append(int(p[0])); device.append(int(p[1]))
                x.append(float(p[6])); y.append(float(p[7]))
    return (np.array(x), np.array(y), np.array(petal, dtype=int), np.array(device, dtype=int))

_STATIC = _load_fiberpos()   # (x_mm, y_mm, petal_loc, device_loc) read from disk ONCE at import


def _activate(x_mm, y_mm, petal_loc, device_loc):
    """Bind the active positioner geometry that the drift functions use."""
    global _SX, _SY, _XIP, _ETAP, _PETAL, _DEVICE, NPOS
    _SX = np.asarray(x_mm, float); _SY = np.asarray(y_mm, float)
    _XIP = _SX * MM2DEG; _ETAP = _SY * MM2DEG          # focal-plane positions in field degrees
    _PETAL = np.asarray(petal_loc, int); _DEVICE = np.asarray(device_loc, int); NPOS = len(_SX)

_activate(*_STATIC)          # start on the static layout


def set_positions(x_mm, y_mm, petal_loc, device_loc):
    """Inject ACTUAL fiber-tip positions (online) in place of the static mounting-hole layout.

    The vendored layout is the positioner CENTERS; a fiber tip can be anywhere in its ~6-10 mm patrol
    disk. The ICS ETC wrapper (which has DOS access) may call PETALMAN `get_positions` at start_exposure
    and pass its x/y (+ petal_loc/device_loc) here via etc.py, keeping darsplit itself DOS-free. If this
    is never called, the static layout is used. Measured effect of actual vs. static positions: v_p99 and
    tau* move only ~0.2% / ~0.1% (patrol offsets average out over 5000 fibers); the real value is a
    *specific* binding-fiber (petal_loc, device_loc) rather than a representative edge one. The offline
    harness cannot fetch positions, so it keeps the static default. Scope: one exposure; call once at
    start_exposure, `reset_positions()` when done. (Pending: confirm `get_positions` returns x,y — Klaus
    to sample its output; revisit ~2026-09-18.)
    """
    _activate(x_mm, y_mm, petal_loc, device_loc)


def reset_positions():
    """Restore the static positioner layout cached at startup (no disk re-read)."""
    _activate(*_STATIC)


# ----- astrometry + DAR distortion -----
def zpsi(HA_deg, dec):
    """Zenith distance z and parallactic-frame angle psi (radians) at hour angle HA_deg, dec (deg)."""
    H = np.radians(HA_deg); d = np.radians(dec)
    z = np.arccos(np.clip(np.sin(PHI) * np.sin(d) + np.cos(PHI) * np.cos(d) * np.cos(H), -1, 1))
    return z, np.arctan2(np.sin(H), np.tan(PHI) * np.cos(d) - np.sin(d) * np.cos(H))

def _distort(z, psi):
    """Per-fiber DAR displacement (um) across the focal plane at zenith distance z, angle psi."""
    tz = np.tan(z); sp, cp = np.sin(psi), np.cos(psi)
    f1 = R * (1 + (sp * tz) ** 2); f2 = sp * cp * R * tz ** 2; f3 = R * (1 + (cp * tz) ** 2)
    return (-_XIP * f1 - _ETAP * f2) * 3600 * UM, (-_ETAP * f3 - _XIP * f2) * 3600 * UM

def airmass(HA_h, dec):
    """Airmass at hour angle HA_h (hours) and declination dec (deg); inf below the horizon."""
    z, _ = zpsi(HA_h * 15.0, dec); c = np.cos(z)
    return 1.0 / c if c > 0 else np.inf


# ----- DAR drift rate + binding fiber -----
def _drift_rates(HA_h, dec, h=10.0):
    """Per-fiber DAR drift-rate magnitude (um/s) at HA_h via central finite difference (dt=h s)."""
    hd = HA_h * 15.0; dH = OMEGA * h
    dxp, dep = _distort(*zpsi(hd + dH, dec))
    dxm, dem = _distort(*zpsi(hd - dH, dec))
    return np.hypot((dxp - dxm) / (2 * h), (dep - dem) / (2 * h))

def v_p99(HA_h, dec, p=P, h=10.0):
    """p-percentile of the per-fiber DAR drift rate (um/s) -- the binding-fiber rate for tau*/f2."""
    return float(np.quantile(_drift_rates(HA_h, dec, h), p))

def binding_fiber(HA_h, dec, p=P, h=10.0):
    """Return (rate_um_s, petal_loc, device_loc) of the p-percentile binding fiber (an actual fiber).

    Rank-selected (ceil(p*(N-1))) so it names a real hardware location; its rate matches v_p99 closely.
    """
    rates = _drift_rates(HA_h, dec, h)
    order = np.argsort(rates)
    k = int(np.ceil(p * (len(rates) - 1)))
    idx = order[k]
    return float(rates[idx]), int(_PETAL[idx]), int(_DEVICE[idx])


# ----- field seeing + fibre-acceptance width -----
def field_seeing(zenith_seeing, X):
    """Delivered field seeing (arcsec) from zenith seeing and airmass: s_zenith * X**0.3."""
    return zenith_seeing * X ** AEXP

def sigma(field_seeing_asec):
    """Fibre-acceptance sigma (um) at the given DELIVERED field seeing (arcsec)."""
    return SIG0 * field_seeing_asec / S_REF


# ----- measured point-source flux retention vs positioning offset (digitized, seeing 0.9-1.3") -----
_OFF   = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.])   # offset (um)
_CURVE = {0.9: np.array([1.00, 0.985, 0.94, 0.85, 0.72, 0.585, 0.46, 0.32, 0.20, 0.13, 0.075]),
          1.1: np.array([1.00, 0.99, 0.95, 0.885, 0.78, 0.665, 0.525, 0.40, 0.295, 0.225, 0.165]),
          1.3: np.array([1.00, 0.99, 0.96, 0.905, 0.815, 0.71, 0.575, 0.445, 0.345, 0.265, 0.195])}
_SEE_K = np.array([0.9, 1.1, 1.3])
_STACK = np.vstack([_CURVE[0.9], _CURVE[1.1], _CURVE[1.3]])

def _f2curve(field_seeing_asec):
    """Flux-retention row at the given field seeing (clamped to the measured 0.9-1.3" range)."""
    sf = min(max(field_seeing_asec, 0.9), 1.3)
    return np.array([np.interp(sf, _SEE_K, _STACK[:, j]) for j in range(len(_OFF))])

def fret(delta_um, field_seeing_asec):
    """Instantaneous point-source flux retention at positioning offset delta_um (um)."""
    return float(np.interp(abs(delta_um), _OFF, _f2curve(field_seeing_asec)))

def f2mean(dmax_um, field_seeing_asec, ns=120):
    """Time-integrated <f>^2 over a segment (linear drift 0 -> dmax_um, uniform in offset); <=1, ->1 short.

    Point source is conservative (extended ELG loses less), matching the analysis. dmax_um is the maximum
    offset over the segment: for midpoint placement dmax = v_p99 * tau/2; for start placement v_p99 * tau.
    """
    if dmax_um <= 0:
        return 1.0
    row = _f2curve(field_seeing_asec)
    ds = np.linspace(0, dmax_um, ns)
    return float(np.mean(np.interp(ds, _OFF, row))) ** 2


# ----- split cadence -----
def drift_coeff(HA_h, dec, field_seeing_asec, p=P):
    """Small-drift survey-speed cost coefficient a = v_p^2 / (12 sigma^2) (1/s^2)."""
    return v_p99(HA_h, dec, p) ** 2 / (12.0 * sigma(field_seeing_asec) ** 2)

def tau_star(HA_h, dec, field_seeing_asec, p=P, delta=DELTA):
    """Unclamped efficiency-optimal segment length (s): (Delta / 2a)^(1/3)."""
    return (delta / (2.0 * drift_coeff(HA_h, dec, field_seeing_asec, p))) ** (1.0 / 3.0)

def dar_split_time(HA_h, dec, field_seeing_asec, p=P, delta=DELTA, t_cap=T_CAP):
    """DAR-driven per-segment cap (s) = min(t_cap, tau*). etc.py stashes this on accum for update()."""
    return min(t_cap, tau_star(HA_h, dec, field_seeing_asec, p, delta))

def dar_split_time_maxdrift(HA_h, dec, max_drift_um, p=P, t_cap=T_CAP, midpoint=True):
    """Opt-in hard max-drift mode: split when the binding fiber reaches max_drift_um.

    Midpoint placement tolerates +/- max_drift_um about center -> dar_split_time = 2*max_drift/v_p99;
    start placement -> max_drift/v_p99. Same min(t_cap, .) cap as the efficiency mode. Default off.
    """
    v = v_p99(HA_h, dec, p)
    if v <= 0:
        return t_cap
    tau = (2.0 if midpoint else 1.0) * max_drift_um / v
    return min(t_cap, tau)
