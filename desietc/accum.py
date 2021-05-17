import datetime

try:
    import DOSlib.logger as logging
except ImportError:
    # Fallback when we are not running as a DOS application.
    import logging

import numpy as np

import desietc.util


class Accumulator(object):

    SECS_PER_DAY = 86400

    SOURCES = dict(OPEN=0, GFA=1, SKY=2, TICK=3, CLOSE=4)

    def __init__(self, sig_buffer, bg_buffer, grid_resolution, min_exptime_secs=0, max_transcript=10000):
        """Initialize an effective exposure time accumulator.

        Parameters
        ----------
        sig_buffer : MeasurementBuffer
            Buffer containing recent signal throughput measurements
            normalized to one for nominal conditions.
        bg_buffer : MeasurementBuffer
            Buffer containing recent background sky-level measurements
            normalized to one for nominal conditions.
        min_exptime_secs : float
            Minimum allowed spectrograph exposure time in seconds. A stop or split
            request will never be issued until this interval has elapsed after
            the spectrograph shutters open.
        max_transcript : int
            Maximum number of transcript samples that can be recorded.
        """
        self.sig_buffer = sig_buffer
        self.bg_buffer = bg_buffer
        self.grid_resolution = grid_resolution
        self.min_exptime_secs = min_exptime_secs
        self.reset()
        # Initialize a per-exposure transcript of updates.
        self.max_transcript = max_transcript
        self.ntranscript = 0
        self.transcript = np.zeros(max_transcript, dtype=[
            ('mjd', np.float64),
            ('mjd_src', np.float64),
            ('src', np.int32),
            ('signal', np.float32),
            ('background', np.float32),
            ('efftime', np.float32),
            ('realtime', np.float32),
            ('remaining', np.float32),
            ('next_split', np.float32),
        ])

    def reset(self):
        """Reset accumulated quantities.
        """
        self.last_mjd = desietc.util.date_to_mjd(datetime.datetime.utcnow(), utc_offset=0)
        self.last_updated = desietc.util.mjd_to_date(self.last_mjd, utc_offset=0).isoformat()
        self.efftime = self.realtime = self.efftime_tot = self.realtime_tot = 0.
        self.signal = self.background = 0.
        self.remaining = self.next_split = self.proj_efftime = 0.
        self.nsplit_remaining = 1
        self.nopen = self.nclose = 0
        self.shutter_is_open = False
        self.shutter_open = []
        self.shutter_close = []
        self.shutter_teff = []
        self.shutter_treal = []
        self.splittable = False
        self.action = None
        self.mjd_grid = None

    def setup(self, req_efftime, max_exposure_time, cosmics_split_time, maxsplit, warning_time,
              rdnoise_1ks):
        """Setup a new sequence of cosmic splits.

        Parameters
        ----------
        req_efftime : float
            The requested target value of the effective exposure time in seconds.
        max_exposure_time : float
            The maximum cummulative exposure time in seconds to allow for this tile,
            summed over all cosmic splits.
        cosmics_split_time : float
            The maximum exposure time in seconds for a single exposure.
        maxsplit : int
            The maximum number of exposures reserved by ICS for this tile.
        warning_time : float
            Warn when a stop or split is expected within this interval in seconds.
        rdnoise_1ks : float
            Nominal read noise relative to 1000s of nominal sky background.
        """
        self.req_efftime = req_efftime
        self.max_exposure_time = max_exposure_time
        self.cosmics_split_time = cosmics_split_time
        self.maxsplit = maxsplit
        self.warning_time = warning_time
        self.rdnoise_1ks = rdnoise_1ks
        self.MW_transp = 1.
        self.reset()

    def open(self, timestamp, splittable, max_shutter_time, MW_transp):
        """Open the shutter.

        Parameters
        ----------
        timestamp : datetime.datetime
            The UTC timestamp for this shutter opening.
        splittable : bool
            True if this exposure can be split for cosmics.
        max_shutter_time : float
            The maximum time in seconds that the spectrograph shutters will remain open.
        MW_transp : float
            Transparency factor for Milky Way dust extinction to use for accumulating
            effective exposure time.

        Returns
        -------
        bool
            True if successful, or False if we do not know the state of the shutter.
        """
        self.nopen, self.nclose = len(self.shutter_open), len(self.shutter_close)
        if self.nopen != self.nclose:
            # We do not have a consistent shutter state.
            logging.error(f'open_shutter called after {self.nopen} opens, {self.nclose} closes: will ignore it.')
            return False
        self.splittable = splittable
        self.max_shutter_time = max_shutter_time
        self.MW_transp = MW_transp
        # Initialize a MJD grid to use for SNR calculations during this shutter.
        # Grid values are bin centers, with spacing fixed at grid_resolution.
        # The grid covers from now up to the maximum allowed exposure time.
        self.max_remaining = self.max_exposure_time - np.sum(self.shutter_treal)
        if self.max_remaining < self.min_exptime_secs:
            logging.warning(f'Increased max remaining from {self.max_remaining:.1f}s to min exptime {self.min_exptime_secs:.1f}s.')
            self.max_remaining = self.min_exptime_secs
        ngrid = int(np.ceil(self.max_remaining / self.grid_resolution))
        mjd = desietc.util.date_to_mjd(timestamp, utc_offset=0)
        self.mjd_grid = mjd + (np.arange(ngrid) + 0.5) * self.grid_resolution / self.SECS_PER_DAY
        # Initialize signal and background rate grids.
        self.sig_grid = np.zeros_like(self.mjd_grid)
        self.bg_grid = np.zeros_like(self.mjd_grid)
        # Record this shutter opening.
        self.shutter_open.append(mjd)
        self.nopen += 1
        self.shutter_is_open = True
        # Start a new transcript for this exposure.
        self.ntranscript = 0
        # Initial update.
        self.update('OPEN', mjd, mjd)
        logging.info(f'Initialized accumulation with max remaining time {self.max_remaining:.1f}s, ' +
            f'MW transp={self.MW_transp:.4f}, splittable={self.splittable}.')
        return True

    def close(self, timestamp):
        """Close the shutter.

        Parameters
        ----------
        timestamp : datetime.datetime
            The UTC timestamp for this shutter closing.

        Returns
        -------
        bool
            True if successful, or False if we do not know the state of the shutter.
        """
        self.nopen, self.nclose = len(self.shutter_open), len(self.shutter_close)
        if self.nopen != self.nclose + 1:
            # We do not have a consistent shutter state.
            logging.error(f'close_shutter called after {self.nopen} opens, {self.nclose} closes: will ignore it.')
            return False
        # Ignore any previously requested action now that the shutter is closed.
        self.action = None
        self.next_split = 0.
        # Final update.
        mjd = desietc.util.date_to_mjd(timestamp, utc_offset=0)
        self.update('CLOSE', mjd, mjd)
        # Record this shutter closing.
        self.shutter_close.append(mjd)
        self.shutter_teff.append(self.efftime)
        self.shutter_treal.append((mjd - self.shutter_open[-1]) * self.SECS_PER_DAY)
        self.nclose += 1
        self.shutter_is_open = False
        return True

    def update(self, src, mjd_src, mjd_now):
        """Update acumulated quantities.

        Parameters
        ----------
        src : str
            The source of this update, which should be one of the keys to self.SOURCE.
        mjd_src : float
            The MJD time that triggered this update.  For a GFA or SKYCAM frame, this would
            be the time when the shutter closed.
        mjd_now : float
            The MJD time of this update.

        Returns
        -------
        bool
            True if the update was successful.
        """
        # Check for a large delay between the src and now.
        delay = (mjd_now - mjd_src) * self.SECS_PER_DAY
        if delay > 20:
            logging.warning(f'Large update delay from {src}: {delay:.1f}s.')
        # Check the state of the shutter.
        if not self.shutter_is_open:
            logging.error(f'update: called with shutter closed [{self.nopen},{self.nclose}].')
            return False
        elif self.nopen != self.nclose + 1:
            logging.error(f'update: invalid shutter state [{self.nopen},{self.nclose}].')
            return False
        # Ignore updates before the shutter opened.
        mjd_open = self.shutter_open[-1]
        if mjd_now < mjd_open:
            before = (mjd_open - mjd_now) * self.SECS_PER_DAY
            logging.info(f'update: ignoring frame recorded {before:.1f}s before shutter opened.')
            return False
        # Lookup the real and effective time accumulated on all previous splits for this exposure.
        prev_teff = np.sum(self.shutter_teff)
        prev_treal = np.sum(self.shutter_treal)
        # Calculate the MJD when the min exposure time has been reached.
        mjd_min_exptime = mjd_open + self.min_exptime_secs / self.SECS_PER_DAY
        # --- The shutter is open and we have the NTS parameters to use -------------------------
        # Record the timestamp of this update.
        self.last_mjd = mjd_now
        self.last_updated = desietc.util.mjd_to_date(mjd_now, utc_offset=0).isoformat()
        # Get grid index coresponding to the current time.
        inow = np.searchsorted(self.mjd_grid, mjd_now)
        past, future = slice(0, inow + 1), slice(inow, None)
        # Tabulate the signal and background since the most recent shutter opening.
        self.sig_grid[past] = self.sig_buffer.sample_grid(self.mjd_grid[past])
        self.bg_grid[past] = self.bg_buffer.sample_grid(self.mjd_grid[past])
        # Calculate the mean signal and background during this shutter open period.
        self.signal = np.mean(self.sig_grid[past])
        self.background = np.mean(self.bg_grid[past])
        # Calculate means of auxiliary signal quantities during this shutter open period.
        self.aux_mean = {}
        for aux_name in ('transp_obs', 'transp_zenith', 'ffrac_psf', 'ffrac_elg', 'ffrac_bgs', 'thru_psf'):
           self.aux_mean[aux_name] = np.mean(
               self.sig_buffer.sample_grid(self.mjd_grid[past], field=aux_name))
        # Calculate the accumulated effective exposure time for this shutter opening in seconds.
        self.realtime = (mjd_now - mjd_open) * self.SECS_PER_DAY
        self.efftime = self.get_efftime(self.realtime, self.signal, self.background)
        logging.info(f'shutter[{self.nopen}] treal={self.realtime:.1f}s, teff={self.efftime:.1f}s' +
            f' [+{prev_teff:.1f}s] using bg={self.background:.3f}, sig={self.signal:.3f}, thru={self.aux_mean["thru_psf"]:.3f}.')
        self.realtime_tot = self.realtime + prev_treal
        self.efftime_tot = self.efftime + prev_teff
        # Have we reached the cutoff time?
        if self.realtime >= self.max_remaining or len(self.mjd_grid[future]) == 0:
            # We have already reached the maximum allowed exposure time.
            self.action = ('stop', 'reached max_exposure_time')
            logging.info(f'Reached max remaining time of {self.max_remaining:.1f}s.')
            self.proj_efftime = self.efftime + prev_teff
            self.remaining = 0.
            self.next_split = 0.
        else:
            # Forecast the future signal and background.
            self.sig_grid[future] = self.sig_buffer.forecast_grid(self.mjd_grid[future])
            self.bg_grid[future] = self.bg_buffer.forecast_grid(self.mjd_grid[future])
            # Calculate accumulated signal and background, assuming the shutter stays open until
            # max_exposure_time.
            n_grid = 1 + np.arange(len(self.mjd_grid))
            accum_sig = np.cumsum(self.sig_grid) / n_grid
            accum_bg = np.cumsum(self.bg_grid) / n_grid
            # Calculate the corresponding accumulated effective exposure time in seconds.
            accum_treal = (self.mjd_grid - mjd_open) * self.SECS_PER_DAY
            accum_teff = self.get_efftime(accum_treal, accum_sig, accum_bg)
            # When do we expect to close the shutter.
            self.action = None
            if self.efftime + prev_teff >= self.req_efftime:
                # We have already reached the target.
                istop = inow
                self.action = ('stop', 'reached req_efftime')
                logging.info(f'Reached requested effective time of {self.req_efftime:.1f}s.')
            elif accum_teff[-1] + prev_teff < self.req_efftime:
                # We will not reach the target before max_exposure_time.
                istop = len(accum_teff) - 1
                logging.info('Will probably not reach requested SNR before max exposure time.')
            else:
                # We expect to reach the target before mjd_max but are not there yet.
                istop = np.argmax(accum_teff + prev_teff >= self.req_efftime)
            # Lookup our expected stop time and time remaining, assuming the shutter stays open.
            mjd_stop = self.mjd_grid[istop]
            # Enforce the minimum exposure time.
            if mjd_stop < mjd_min_exptime:
                if self.action is not None:
                    logging.warning(f'Delaying stop until min exptime of {self.min_exptime_secs}s.')
                    self.action = None
                else:
                    logging.warning(f'Estimated stop occurs before min exptime of {self.min_exptime_secs}s.')
                mjd_stop = mjd_min_exptime
                istop = min(np.searchsorted(self.mjd_grid, mjd_min_exptime), len(self.mjd_grid)-1)
            # Calculate the remaining real time until we expect to stop.
            self.remaining = (mjd_stop - mjd_now) * self.SECS_PER_DAY
            # Calculate the corresponding effective time, including any previous shutters.
            self.proj_efftime = accum_teff[istop] + prev_teff
            treal_stop = (mjd_stop - mjd_open) * self.SECS_PER_DAY
            logging.info(f'Will stop in {self.remaining:.1f}s at teff={self.proj_efftime:.1f}s' +
                f' (target={self.req_efftime:.1f}s), treal={treal_stop:.1f}s (max={self.max_remaining:.1f}s).')
            # Calculate how many cosmic splits are remaining if none exceeds the split maximum.
            # This value is frozen once we reach half of the max split time.
            if self.realtime < self.cosmics_split_time:
                self.nsplit_remaining = int(np.ceil((mjd_stop - mjd_open) * self.SECS_PER_DAY / self.cosmics_split_time))
            # Calculate when the next split should be.
            mjd_split = mjd_open + (mjd_stop - mjd_open) / self.nsplit_remaining
            # Enforce the minimum exposure time.
            if mjd_split < mjd_min_exptime:
                logging.warning(f'Delaying split until min exptime of {self.min_exptime_secs}s.')
                mjd_split = mjd_min_exptime
            self.next_split = (mjd_split - mjd_now) * self.SECS_PER_DAY
            if self.next_split > self.cosmics_split_time:
                logging.warning(f'Clipping next_split from {self.next_split:.1f}s to {self.cosmics_split_time:.1f}s')
                self.next_split = self.cosmics_split_time
            if self.action is None and self.nsplit_remaining > 1:
                logging.info(f'Next split ({self.nopen} of {self.nclose+self.nsplit_remaining}) in {self.next_split:.1f}s.')
                if self.splittable and self.next_split <= 0:
                    self.action = ('split', 'cosmic split')
            # Are we about to stop or split?
            if self.action is None:
                if self.realtime + self.warning_time >= self.max_remaining:
                    self.action = ('warn-stop', 'about to reach max_exposure_time')
                elif self.remaining <= self.warning_time:
                    self.action = ('warn-stop', 'about to reach req_efftime')
                elif self.next_split <= self.warning_time and self.splittable:
                    self.action = ('warn-split', 'about to split')
        if self.action is not None:
            logging.info(f'Recommended action is {self.action}.')
        # Save this update to the transcript.
        if self.ntranscript == self.max_transcript:
            logging.warn(f'Accumulator transcript full with {self.ntranscript} entries.')
        elif self.ntranscript < self.max_transcript:
            src = self.SOURCES.get(src, -1)
            self.transcript[self.ntranscript] = (
                mjd_now, mjd_src, src, self.signal, self.background,
                self.efftime, self.realtime, self.remaining, self.next_split)
        self.ntranscript += 1

        return True

    def save_transcript(self):
        # Replace 64-bit MJD fields with 32-bit offsets in seconds relative to the initial MJD value.
        mjd0 = float(self.transcript['mjd'][0])
        N = self.ntranscript
        D = dict(
            mjd0=mjd0,
            dt=np.float32((self.transcript[:N]['mjd'] - mjd0) * self.SECS_PER_DAY),
            dt_src=np.float32((self.transcript[:N]['mjd_src'] - mjd0) * self.SECS_PER_DAY))
        for field in self.transcript.dtype.names:
            if field in ('mjd', 'mjd_src'):
                continue
            D[field] = self.transcript[:N][field]
        return D

    def get_efftime(self, realtime, signal, background, scale=(0.56 / 0.435)**2 / 1.07):
        """Calculate the effective exposure time corresponding to the specified real
        exposure time, accumulated signal and background rates, and their nominal values
        and MW transparency specified in the last call to :meth:`setup_exposure`.
        """
        # The factor of (1.038 / 1.107) = 0.93767 is to correct for the larger mean
        # ETC/SPEC ratio with the new algorithms.
        sig_factor = self.MW_transp * signal * 0.93767
        rdnoise = self.rdnoise_1ks * 1000 / np.maximum(0.1, realtime)
        bg_factor = (background + rdnoise) / (1 + self.rdnoise_1ks)
        return scale * realtime * sig_factor ** 2 / bg_factor
