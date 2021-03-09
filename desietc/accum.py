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

    def __init__(self, sig_buffer, bg_buffer, grid_resolution):
        """Initialize an effective exposure time accumulator.

        Parameters
        ----------
        sig_buffer : MeasurementBuffer
            Buffer containing recent signal throughput measurements.
        bg_buffer : MeasurementBuffer
            Buffer containing recent background sky-level measurements.
        """
        self.sig_buffer = sig_buffer
        self.bg_buffer = bg_buffer
        self.grid_resolution = grid_resolution
        self.reset()

    def reset(self):
        """Reset accumulated quantities.
        """
        self.last_mjd = desietc.util.date_to_mjd(datetime.datetime.utcnow(), utc_offset=0)
        self.last_updated = desietc.util.mjd_to_date(self.last_mjd, utc_offset=0).isoformat()
        self.efftime = self.realtime = self.efftime_tot = self.realtime_tot = 0.
        self.signal = self.background = 0.
        self.remaining = self.next_split = self.proj_efftime = 0.
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
              sig_nominal, bg_nominal):
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
        sig_nominal : float
            Signal rate in nominal conditions.
        bg_nominal : float
            Background rate in nominal conditions.
        """
        self.req_efftime = req_efftime
        self.max_exposure_time = max_exposure_time
        self.cosmics_split_time = cosmics_split_time
        self.maxsplit = maxsplit
        self.warning_time = warning_time
        self.sig_nominal = sig_nominal
        self.bg_nominal = bg_nominal
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
        # Record this shutter closing.
        mjd = desietc.util.date_to_mjd(timestamp, utc_offset=0)
        self.shutter_close.append(mjd)
        self.shutter_teff.append(self.efftime)
        self.shutter_treal.append((mjd - self.shutter_open[-1]) * self.SECS_PER_DAY)
        self.nclose += 1
        self.shutter_is_open = False
        return True

    def update(self, mjd_now):
        """Update acumulated quantities.

        Parameters
        ----------
        mjd_now : float
            The MJD time of this update.

        Returns
        -------
        bool
            True if the update was successful.
        """
        # Check the state of the shutter.
        if not self.shutter_is_open:
            logging.error(f'update: called with shutter closed [{self.nopen},{self.nclose}].')
            return False
        elif self.nopen != self.nclose + 1:
            logging.error(f'update: invalid shutter state [{self.nopen},{self.nclose}].')
            return False
        # Lookup the real and effective time accumulated on all previous splits for this exposure.
        prev_teff = np.sum(self.shutter_teff)
        prev_treal = np.sum(self.shutter_treal)
        # --- The shutter is open and we have the NTS parameters to use -------------------------
        # Record the timestamp of this update.
        self.last_mjd = mjd_now
        self.last_updated = desietc.util.mjd_to_date(mjd_now, utc_offset=0).isoformat()
        # Get grid indices coresponding to the most recent shutter opening and now.
        mjd_open = self.shutter_open[-1]
        inow = np.searchsorted(self.mjd_grid, mjd_now)
        past, future = slice(0, inow + 1), slice(inow, None)
        # Tabulate the signal and background since the most recent shutter opening.
        self.sig_grid[past] = self.sig_buffer.sample_grid(self.mjd_grid[past])
        self.bg_grid[past] = self.bg_buffer.sample_grid(self.mjd_grid[past])
        # Calculate the mean signal and background during this shutter open period.
        self.signal = np.mean(self.sig_grid[past])
        self.background = np.mean(self.bg_grid[past])
        # Calculate the accumulated effective exposure time for this shutter opening in seconds.
        self.realtime = (mjd_now - mjd_open) * self.SECS_PER_DAY
        self.efftime = self.realtime * self.exptime_factor(
            self.signal, self.background)
        logging.info(f'shutter[{self.nopen}] treal={self.realtime:.1f}s, teff={self.efftime:.1f}s' +
            f' [+{prev_teff:.1f}s] using bg={self.background:.3f}, sig={self.signal:.3f}.')
        self.realime_tot = self.realtime + prev_treal
        self.efftime_tot = self.efftime + prev_teff
        # Have we reached the cutoff time?
        if self.realtime >= self.max_remaining:
            # We have already reached the maximum allowed exposure time.
            self.action = ('stop', 'reached max_exposure_time')
            logging.info(f'Reached maximum exposure time of {self.max_exposure_time:.1f}s.')
            self.proj_efftime = self.efftime + prev_teff
            self.remaining = 0.
            self.next_split = 0.
            return True
        # Forecast the future signal and background.
        self.sig_grid[future] = self.sig_buffer.forecast_grid(self.mjd_grid[future])
        self.bg_grid[future] = self.bg_buffer.forecast_grid(self.mjd_grid[future])
        # Calculate accumulated signal and background, assuming the shutter stays open until until max_exposure_time.
        n_grid = 1 + np.arange(len(self.mjd_grid))
        accum_sig = np.cumsum(self.sig_grid) / n_grid
        accum_bg = np.cumsum(self.bg_grid) / n_grid
        # Calculate the corresponding accumulated effective exposure time in seconds.
        accum_treal = (self.mjd_grid - mjd_open) * self.SECS_PER_DAY
        accum_teff = accum_treal * self.exptime_factor(accum_sig, accum_bg)
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
            logging.warning('Will probably not reach requested SNR before max exposure time.')
        else:
            # We expect to reach the target before mjd_max but are not there yet.
            istop = np.argmax(accum_teff + prev_teff >= self.req_efftime)
        # Lookup our expected stop time and time remaining, assuming the shutter stays open.
        mjd_stop = self.mjd_grid[istop]
        self.remaining = (mjd_stop - mjd_now) * self.SECS_PER_DAY
        # Calculate the corresponding effective time, including any previous shutters.
        self.proj_efftime = accum_teff[istop] + prev_teff
        logging.info(f'Will stop in {self.remaining:.1f}s at teff={self.proj_efftime:.1f}s' +
            f' (target={self.req_efftime:.1f}s).')
        # Calculate how many cosmic splits are remaining if none exceeds the split maximum.
        nsplit_remaining = int(np.ceil((mjd_stop - mjd_open) * self.SECS_PER_DAY / self.cosmics_split_time))
        # Calculate when the next split should be.
        mjd_split = mjd_open + (mjd_stop - mjd_open) / nsplit_remaining
        self.next_split = (mjd_split - mjd_now) * self.SECS_PER_DAY
        if self.action is None and nsplit_remaining > 1:
            logging.info(f'Next split ({self.nopen} of {self.nclose+nsplit_remaining}) in {self.next_split:.1f}s.')
            if self.splittable and self.next_split <= 0:
                self.action = ('split', 'cosmic split')
        if self.action is not None:
            logging.info(f'Recommended action is {self.action}.')
        return True

    def exptime_factor(self, signal, background):
        """Calculate the ratio between effective and real exposure time using the
        specified accumulated signal and background rates, and their nominal values
        and MW transparency specified in the last call to :meth:`setup_exposure`.
        A returned value of one corresponds to real time = effective time.
        """
        return (self.MW_transp * signal / self.sig_nominal) ** 2 / (background / self.bg_nominal)
