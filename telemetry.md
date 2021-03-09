# ETC Telemetry

Documentation for the variables reported by the `OnlineETC.get_status` method that are normally archived to the [ICS telemetry database](https://replicator.desi.lbl.gov/TV3/app/T/index).

## Flags that control the exposure state machine:

 - `img_proc` (bool): Image processing is active for observing a single tile, possibly with cosmic splits.
 - `etc_proc` (bool): The spectrograph shutters are open.
 - `etc_ready` ((bool)): The ETC algorithm is ready and operating normally.

## Timestamps of exposure state transitions:

These are all [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) formatted strings representing dates and times in UTC.  Subtract 7 hours to get local times at KPNO.

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

## Stop sources captured by `stop` and `stop_etc`:

 - `img_stop_src` (str): The source of the last `stop` transition.
 - `etc_stop_src` (str): The source of the last `stop_etc` transition.

## General information about the ETC algorithm:

 - `desietc` (str): A description of the exact version of the [desietc package](https://github.com/desihub/desietc) being used.
 - `gfa_count` (int): The total number of GFA frames processed since the algorithm started, including both acquisition and guide frames.
 - `sky_count` (int): The total number of SKYCAM frames processed since the algorithm started.
 - `desi_count` (int): The total number of DESI spectrograph exposures processed since the algorithm started.

## Observing conditions updated after each GFA or SKY frame:

FFRAC below refers to the fiber acceptance fraction, which is a dimensionless quantity between 0 and 1 that estimates the fraction of light entering a typical fiber.

TRANSP below refers to the atmospheric transparency, which is a dimensionless quantity equal to 1 for nominal atmospheric extinction. Values are estimated from the observed flux of guide stars with known magnitudes and corrected to airmass 1 (zenith).

The FFRAC and TRANSP averages below combine all measurements over the past two minutes, and require at least 4 measurements.

 - `seeing` (float): FWHM of the PSF measured in the most recent acquisition image, in arcseconds.
 - `ffrac_psf` (float): FFRAC for a PSF-like source (star, quasar).
 - `ffrac_elg` (float): FFRAC for a nominal ELG target with a round exponential profile of 0.45" half-light radius. **Not implemented yet.**
 - `ffrac_bgs` (float): FFRAC for a nominal BGS target with a round DeVaucouleur profile of 1.50" half-light radius. **Not implemented yet.**
 - `ffrac` (float): FFRAC appropriate for the `sbprof` specified for this tile.
 - `ffrac_avg` (float): Average of recent FFRAC measurements.
 - `transp` (float): Zenith atmospheric transparency.
 - `transp_avg` (float): Average of recent TRANSP measurements.
 - `skylevel` (float): Current sky level relative to nominal zenith dark sky conditions.

## Accumulated exposure-time tracking:

These quanties are updated after each new GFA or SKYCAM frame is analyzed while the spectrograph shutters are open, then stay constant while the shutter is closed, until the next exposure sequence starts.

 - `last_mjd` (float): The MJD of the last update to any accumulated values.
 - `signal` (float): The accumulated relative signal (`ffrac * transp`) since the shutter was last opened. Does not included MW dust extinction but does include atmospheric extinction.
 - `background` (float): The accumulated relative background (`skylevel`) since the shutter was last opened.
 - `efftime` (float): The effective exposure time in seconds accumulated since the shutter was last opened.
 - `realtime` (float): The real exposure time in seconds accumulated since the shutter was last opened. Should never reach `cosmics_split` when splits are enabled.
 - `efftime_tot` (float): Same as `efftime` but also including any previous cosmic splits for this tile. The exposure stops once this reaches `req_efftime`.
 - `realtime_tot` (float): Same as `realtime` but also including any previous cosmic splits for this tile. The exposure stops if this reaches `max_exptime` before `req_efftime` is accumulated.
 - `remaining` (float): The estimated additional real time in seconds required to reach `req_efftime`, summed over any future cosmic splits.
 - `prof_efftime` (float): The projected final value of `efftime_tot` when the exposure stops. Will be less than `req_efftime` is the exposure is not expected to finish before `max_exptime`.
 - `next_split` (float): The projected real time until the next cosmic split in seconds.
 - `splittable` (bool): Is the ETC allowed to request a cosmic split for the current exposure?

## Post-exposure results:

 - `fieldrot` (float): The best-fit linear field rotation during the last exposure in units TBD. Note that this estimate is relative to any hexapod field rotation corrections already being applied during the exposure. **Not implemented yet.**