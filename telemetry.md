# ETC Telemetry

Documentation for the variables reported by the `OnlineETC.get_status` method that are normally archived to the [ICS telemetry database](https://replicator.desi.lbl.gov/TV3/app/T/index).

ETC telemetry updates occur when a visit (sequence of split exposures on the same tile) is active, as signaled by calls to the `start` and `stop` methods of `OnlineETC`. During a visit, updates are triggered when:
 - The spectrograph shutters open or close, as signaled by calls to `start_etc` and `stop_etc`.
 - A new SkyCam exposure is processed.
 - The initial GFA acquisition exposure is processed.
 - The spectrograph shutters are open and:
   - a new GFA exposure is processed, or
   - more than 2 seconds have elapsed since the last update.

## Flags that control the exposure state machine:

 - `img_proc` (bool): Image processing is active for observing a single tile, possibly with cosmic splits.
 - `etc_proc` (bool): The spectrograph shutters are open.
 - `etc_ready` (bool): The ETC algorithm is ready and operating normally.

## Timestamps of exposure state transitions:

These are all [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) formatted strings representing dates and times in UTC.  Subtract 7 hours to get local times at KPNO. Use [datetime.fromoisoformat](https://docs.python.org/3/library/datetime.html#datetime.datetime.fromisoformat) to convert to `datetime` objects.

 - `img_start_time` (str): When the `img_proc` flag was last set.
 - `img_stop_time` (str): When the `img_proc` flag was last cleared.
 - `etc_start_time` (str): When the `etc_proc` flag was last set.
 - `etc_stop_time` (str): When the `etc_proc` flag was last cleared.

## Exposure parameters set in `prepare_for_exposure`:

 - `expid` (int): The current exposure identifier.
 - `req_efftime` (float): The requested effective exposure time in seconds.
 - `sbprof` (str): The surface brightness profile to use for FFRAC calculations. Must be one of PSF,ELG,BGS,FLT.
 - `max_exptime` (float): The maximum total exposure time for this tile in seconds, summed over any cosmic splits.
 - `cosmics_split` (float): The maximum length of a single exposure in seconds.
 - `maxsplit` (int): The maximum number of cosmic split exposures for this tile.
 - `warning_time` (float): Warn when a stop or split is expected within this interval in seconds.

## Stop sources captured by `stop` and `stop_etc`:

 - `img_stop_src` (str): The source of the last `stop` transition.
 - `etc_stop_src` (str): The source of the last `stop_etc` transition.

## General information about the ETC algorithm:

 - `desietc` (str): A description of the exact version of the [desietc package](https://github.com/desihub/desietc) being used.
 - `gfa_count` (int): The total number of GFA frames processed since the algorithm started, including both acquisition and guide frames.
 - `sky_count` (int): The total number of SKYCAM frames processed since the algorithm started.
 - `desi_count` (int): The total number of DESI spectrograph exposures processed since the algorithm started.

## Observing conditions updated after each GFA or SKY frame:

FFRAC below refers to the relative fiber acceptance fraction, which is a dimensionless quantity that estimates the fraction of light entering a typical fiber.

TRANSP below refers to the atmospheric transparency, which is a dimensionless quantity equal to 1 for nominal atmospheric extinction. Values are estimated from the observed flux of guide stars with known magnitudes and corrected to airmass 1 (zenith).

The FFRAC and TRANSP averages below combine all measurements over the past two minutes, and require at least 4 measurements.

 - `seeing` (float): FWHM of the PSF measured in the most recent acquisition image, in arcseconds.
 - `ffrac_psf` (float): FFRAC for a PSF-like source (star, quasar).
 - `ffrac_elg` (float): FFRAC for a nominal ELG target with a round exponential profile of 0.45" half-light radius.
 - `ffrac_bgs` (float): FFRAC for a nominal BGS target with a round DeVaucouleur profile of 1.50" half-light radius.
 - `ffrac_avg` (float): Average of recent `ffrac_psf` measurements relative to nominal conditions (1.1").
 - `transp` (float): Zenith atmospheric transparency.
 - `transp_avg` (float): Average of recent TRANSP measurements.
 - `thru_avg` (float): Average of recent TRANSP * FFRAC measurements relative to nominal conditions (1.1").
 - `skylevel` (float): Current sky level relative to nominal zenith dark sky conditions.

## Survey speeds updated after each GFA frame:

Survey speed is calculated separately for ELG (dark), BGS (bright) and PSF (backup) profiles and reflects
conditions at airmass 1 and with no galactic dust extinction. Speeds do not account for read noise or
source shot noise.
See [here](https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed#SurveySpeed) for details.

The variables ending with `_nts` are averaged over 20 minutes for Next Tile Selector (NTS) decisions.
The other variables have the same defintion but are averaged over 2 minutes.

 - `speed_dark` (float): speed calculated with the ELG profile and 2-min averaging.
 - `speed_bright` (float): speed calculated with the BGS profile and 2-min averaging.
 - `speed_backup` (float): speed calculated with the PSF profile and 2-min averaging.
 - `speed_dark_nts` (float): speed calculated with the ELG profile and 20-min averaging.
 - `speed_bright_nts` (float): speed calculated with the BGS profile and 20-min averaging.
 - `speed_backup_nts` (float): speed calculated with the PSF profile and 20-min averaging.

## Accumulated exposure-time tracking:

These quanties are updated after each new GFA or SKYCAM frame is analyzed while the spectrograph shutters are open, then stay constant while the shutter is closed, until the next exposure sequence starts.

 - `last_updated` (str): The ISO format datetime string corresponding to `last_mjd`.
 - `last_mjd` (float): The MJD of the last update to any accumulated values.
 - `efftime` (float): The effective exposure time in seconds accumulated since the shutter was last opened.
 - `realtime` (float): The real exposure time in seconds accumulated since the shutter was last opened. Should never reach `cosmics_split` when splits are enabled.
 - `efftime_tot` (float): Same as `efftime` but also including any previous cosmic splits for this tile. The exposure stops once this reaches `req_efftime`.
 - `realtime_tot` (float): Same as `realtime` but also including any previous cosmic splits for this tile. The exposure stops if this reaches `max_exptime` before `req_efftime` is accumulated.
 - `remaining` (float): The estimated additional real time in seconds required to reach `req_efftime`, summed over any future cosmic splits.
 - `proj_efftime` (float): The projected final value of `efftime_tot` when the exposure stops. Will be less than `req_efftime` is the exposure is not expected to finish before `max_exptime`.
 - `next_split` (float): The projected real time until the next cosmic split in seconds.
 - `splittable` (bool): Is the ETC allowed to request a cosmic split for the current exposure?

## Post-exposure results:

 - `rel_rotrate` (float): The best-fit linear field rotation rate during the last exposure in arcsec/minute. Note that this estimate is relative to any hexapod field rotation corrections already being applied during the exposure. **Not implemented yet.**