import numpy as np

import desietc.util


class Accumulator(object):

    def __init__(self, sig_buffer, bg_buffer):
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
        self.reset()

    def reset(self):
        """Reset accumulated quantities.
        """
        pass

    def setup(self, req_efftime, max_exposure_time, cosmics_split_time, maxsplit):
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
        """
        self.mjd_grid = None
        self.req_efftime = req_efftime
        self.max_exposure_time = max_exposure_time
        self.cosmics_split_time = cosmics_split_time
        self.maxsplit = maxsplit

    def open(self, timestamp, splittable, max_shutter_time):
        """Open the shutter.

        Parameters
        ----------
        timestamp : datetime.datetime
            The UTC timestamp for this shutter opening.
        splittable : bool
            True if this exposure can be split for cosmics.
        max_shutter_time : float
            The maximum time in seconds that the spectrograph shutters will remain open.
        """
        pass

    def close(self, timestamp):
        """Close the shutter.

        Parameters
        ----------
        timestamp : datetime.datetime
            The UTC timestamp for this shutter closing.
        """
        pass

    def update(self, mjd_now):
        """Update acumulated quantities.

        Parameters
        ----------
        mjd_now : float
            The MJD time of this update.
        """
        pass
