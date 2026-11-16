"""Pointing-limit check for the ETC split decision.

A split commits the telescope to another segment on the same field, which keeps
setting while it tracks. Before committing, check that no pointing limit is
violated at any time during that next segment:

* the Mayall lower dome shutter (when raised): an ELEVATION limit on the lower
  edge of the field;
* observer-entered NFS constraints (OCS_NFSCONSTRAINTS elrange/azrange),
  applied to the field edge over the whole segment, not only at the start. For
  azimuth, the field edge means the forbidden arc is widened by the field's
  azimuth half-width, asin(sin r / cos el).

If a limit would be violated the ETC stops (reason LIMIT) instead of splitting.
Segments are never shortened to fit the remaining window.

Ported from telemetry_mining support/shutter_check.py (stdlib math only).

Geometry notes: the limiting point is the lower EDGE of the field, so an edge
limit converts to a centre limit by adding the field radius. Elevation is
monotonic in |HA|, so its minimum over the segment is at an endpoint. Azimuth
is NOT monotonic in HA, so azimuth limits are checked on a sampled grid.
Elevations are geometric (unrefracted), which errs slightly towards stopping.
"""
import math

LAT_DEG = 31.0 + 57.0 / 60 + 50.5 / 3600   # Mayall, 31:57:50.5 N
HA_DEG_PER_SEC = 15.041 / 3600.             # sidereal HA rate, as in etc.py._refresh_dar

DESI_FIELD_RADIUS_DEG = 1.6

# PROVISIONAL -- replace with a measured value before relying on it.
# Elevation of the raised lower-shutter edge (lowest unvignetted field-point
# elevation). INFERRED from KPNO's Mosaic3 limit (centre at airmass 2.1 minus the
# Mosaic3 half-diagonal ~0.42 deg), not a measurement of this dome.
SHUTTER_EDGE_EL = 90.0 - math.degrees(math.acos(1.0 / 2.1)) - 0.42   # 28.02 deg

# Always-on floor on the field centre (airmass 3).
EL_LIMIT_HARD = math.degrees(math.asin(1.0 / 3.0))   # 19.47 deg

DEFAULT_MARGIN_DEG = 0.5


def elevation(ha_deg, dec_deg, lat_deg=LAT_DEG):
    """Geometric elevation (deg)."""
    h, d, p = map(math.radians, (ha_deg, dec_deg, lat_deg))
    s = math.sin(p) * math.sin(d) + math.cos(p) * math.cos(d) * math.cos(h)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def azimuth(ha_deg, dec_deg, lat_deg=LAT_DEG):
    """Azimuth (deg) measured North through East, in [0, 360)."""
    h, d, p = map(math.radians, (ha_deg, dec_deg, lat_deg))
    y = -math.cos(d) * math.sin(h)
    x = math.sin(d) * math.cos(p) - math.cos(d) * math.sin(p) * math.cos(h)
    return math.degrees(math.atan2(y, x)) % 360.0


def _in_block(az_deg, lo_deg, hi_deg, pad_deg=0.0):
    """True if az lies in the arc from lo to hi going East (handles wrap), widened by pad_deg."""
    span = (hi_deg - lo_deg) % 360.0 + 2.0 * pad_deg
    if span >= 360.0:
        return True
    return ((az_deg - lo_deg + pad_deg) % 360.0) <= span


def az_half_width(el_deg, field_radius_deg=DESI_FIELD_RADIUS_DEG):
    """Azimuth half-width (deg) of a circular field of the given radius centred at el_deg."""
    c = math.cos(math.radians(el_deg))
    s = math.sin(math.radians(field_radius_deg))
    if c <= s:
        return 180.0    # field reaches the zenith: it spans every azimuth
    return math.degrees(math.asin(s / c))


def shutter_up(shutter_lower):
    """Map the telemetry 'shutter_lower' value to True if the shutter is raised (blocking).

    'open' means the lower shutter is DOWN (no obstruction); anything else ('closed',
    'not open', ...) is treated as raised. A non-string value is treated as raised.
    """
    if isinstance(shutter_lower, str):
        return shutter_lower.strip().lower() != 'open'
    return True


def observer_limits(elrange, azrange):
    """Convert NFS elrange/azrange to (el_min_deg or None, tuple of forbidden az blocks).

    ICS Observer Console convention: North is AZ=0, East is AZ=90; azrange is the
    ALLOWED arc, taken in the positive direction from the first number to the
    second. A span that is a multiple of 360 is no restriction. Only the lower
    elevation bound is used. None or malformed ranges impose nothing.
    """
    el_min = None
    if elrange is not None and len(elrange) == 2 and elrange[0] is not None and elrange[0] > 0:
        el_min = float(elrange[0])
    blocks = ()
    if azrange is not None and len(azrange) == 2 and None not in azrange:
        lo, hi = float(azrange[0]), float(azrange[1])
        if (hi - lo) % 360.0 != 0.0:
            blocks = ((hi % 360.0, lo % 360.0),)
    return el_min, blocks


def check_segment(ha_deg, dec_deg, seg_secs, shutter_lower, elrange=None, azrange=None,
                  margin_deg=DEFAULT_MARGIN_DEG, field_radius_deg=DESI_FIELD_RADIUS_DEG,
                  sample_step_secs=60.0, lat_deg=LAT_DEG):
    """Check a segment that starts now at hour angle ha_deg and lasts seg_secs.

    Returns (ok, reason). reason describes the binding violation, or 'ok'.
    All elevation limits apply to the lower field edge, plus margin_deg.
    """
    limits = [(EL_LIMIT_HARD + margin_deg, 'airmass-3 floor')]
    if shutter_up(shutter_lower):
        limits.append((SHUTTER_EDGE_EL + field_radius_deg + margin_deg, 'lower dome shutter'))
    obs_el, az_blocks = observer_limits(elrange, azrange)
    if obs_el is not None:
        limits.append((obs_el + field_radius_deg + margin_deg, 'observer elevation limit'))
    el_min, el_reason = max(limits)

    ha0 = (ha_deg + 180.0) % 360.0 - 180.0
    ha1 = ha0 + max(seg_secs, 0.0) * HA_DEG_PER_SEC

    worst_el, worst_ha = min((elevation(h, dec_deg, lat_deg), h) for h in (ha0, ha1))
    if worst_el < el_min:
        return False, (f'{el_reason}: centre el {worst_el:.2f} < {el_min:.2f} deg '
                       f'at HA {worst_ha:.2f} deg')

    if az_blocks:
        nstep = max(2, int(math.ceil(max(seg_secs, 0.0) / sample_step_secs)) + 1)
        for i in range(nstep):
            h = ha0 + (ha1 - ha0) * i / (nstep - 1)
            az = azimuth(h, dec_deg, lat_deg)
            pad = az_half_width(elevation(h, dec_deg, lat_deg), field_radius_deg)
            for lo, hi in az_blocks:
                if _in_block(az, lo, hi, pad):
                    return False, (f'observer azimuth limit: az {az:.1f} in blocked '
                                   f'[{lo:.1f}, {hi:.1f}] at HA {h:.2f} deg')
    return True, 'ok'
