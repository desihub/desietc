"""OnlineETC class that intefaces with ICS via callouts implemented by ETCApp.

Original code written by Klaus Honscheid and copied here 16-Feb-2021 from
https://desi.lbl.gov/trac/browser/code/online/ETC/trunk/python/ETC/ETC.py

The ETCApp code is hosted at
https://desi.lbl.gov/trac/browser/code/online/ETC/trunk/python/ETC/ETCApp.py

When this code is run on the mountain under ICS, log messages are written to:

  /data/msdos/dos_home/instances/<instance>/logs/<instance>-ETC-<timestamp>.log

where <instance> is normally "desi_YYYYMMDD" and a new log identified by
<timestamp> is started whenever ETCApp restarts.  The soft link ETC-current.log
points to the most recent log.

OnlineETC uses the following callouts to interact with the ETCApp and the rest of DOS:

  - call_for_acq_image
  - call_for_pm_info
  - call_for_sky_image
  - call_for_gfa_image
  - call_to_update_status
  - call_for_png_dir
  - call_for_exp_dir
  - call_to_request_stop
  - call_to_request_split

The ETCApp keeps the OnlineETC synchronized with data taking by calling the
following methods:

  - prepare_for_exposure: provides NTS params for the next tile to observe

    - start: signals that the GFA acquisition exposure has started

      - start_etc: the spectrograph shutters have just opened
      - stop_etc: the spectrograph shutters have just closed

    - stop: the current exposure has finished and the ETC should save its results

The start/stop_etc methods are called for each cosmic split and the other methods are
called exactly once for each exposure.

The ETC image analysis and exposure-time calculations are handled by a separate
class desietc.etc.ETCAlgorithm that runs in a thread managed by this class.
Synchronization between this worker thread and the main thread that handles
calls to the start/stop/... methods above relies on three threading Events:

  - shutdown: the worker thread exits when this is cleared.
  - image_processing: an exposure that requires ETC tracking is active.
  - etc_processing: the spectrograph shutters are open.

The get_status() method returns a snapshot of our internal attributes, which are
divided into two groups: those updated by the main thread and those updated by
the worker thread.  See the comments in get_status() for details.
"""
import datetime
import sys
import os
import threading
import time
import json

try:
    import DOSlib.logger as logging
except ImportError:
    # Fallback when we are not running as a DOS application.
    import logging

try:
    from DOSlib.PML import SUCCESS, FAILED
except ImportError:
    SUCCESS, FAILED = True, False

import desietc.etc


class OnlineETC():

    def __init__(self, shutdown_event):

        self.shutdown_event = shutdown_event

        # Callouts to the ETC application
        self.call_for_acq_image = None
        self.call_for_pm_info = None
        self.call_for_sky_image = None
        self.call_for_gfa_image = None
        self.call_to_update_status = None
        self.call_for_png_dir = None
        self.call_for_exp_dir = None
        self.call_to_request_stop = None
        self.call_to_request_split = None

        # Initialize the ETC algorithm. This will spawn 6 parallel proccesses (one per GFA)
        # and allocated ~100Mb of shared memory. These resources will be cleared when
        gfa_calib = os.getenv('ETC_GFA_CALIB', None)
        sky_calib = os.getenv('ETC_SKY_CALIB', None)
        if gfa_calib is None or sky_calib is None:
            raise RuntimeError('ETC_GFA_CALIB and ETC_SKY_CALIB must be set.')
        self.ETCalg = desietc.etc.ETCAlgorithm(sky_calib, gfa_calib, parallel=True)

        # initialize status variables
        self.expid = None
        self.req_efftime = None
        self.sbprof = None
        self.max_exposure_time = None
        self.cosmics_split_time = None
        self.maxsplit = None
        self.img_start_time = None
        self.img_stop_time = None
        self.etc_start_time = None
        self.etc_stop_time = None
        self.img_stop_src = None
        self.etc_stop_src = None

        # Create the flags we use to synchronize with the worker thread.
        self.etc_ready = threading.Event()
        self.etc_ready.clear()
        self.image_processing = threading.Event()
        self.image_processing.clear()
        self.etc_processing = threading.Event()
        self.etc_processing.clear()

        # Start the worker thread.
        self.etc_thread = None
        self.start_thread()

    def start_thread(self, max_startup_time=60):
        """Start or restart the ETC worker thread.
        """
        if self.etc_thread is not None and self.etc_thread.is_alive():
            return
        if self.etc_thread is not None:
            logging.error('ETC thread has died so restarting now...')
        self.etc_thread = threading.Thread(target = self._etc)
        self.etc_thread.daemon = True
        self.etc_thread.start()
        # Wait for the ETCAlgorithm to finish its startup.
        elapsed = 0
        while not self.etc_ready.is_set():
            elapsed += 1
            if elapsed ==  max_startup_time:
                raise RuntimeError(f'ETC did not start up after {max_startup_time}s.')
            logging.info(f'[{elapsed}/{max_startup_time}] Waiting for ETC startup...')
            time.sleep(1)
        logging.info('ETCAlgorithm is ready.')

    def _etc(self):
        """This is the ETC algorithm that does the actual work.

        This function normally runs in a separate thread and synchronizes with the
        rest of ICS via two flags: image_processing and etc_processing.

        image_processing is set to indicate that the following are or will soon be available:
         - fiberassign file
         - acquisition image
         - PlateMaker guide stars

        etc_processing is set when the spectrograph shutter has opened so effective exposure
        time tracking should start or resume.

        etc_processing is cleared when the spectrograph shutter has closed so effective exposure
        time tracking should be paused.

        image_processing is cleared to indicate that the sequence of cosmic splits for an
        exposure has finished so we should save our outputs to the exposure directory.

        When image_processing is set, a new status update is generated after:
         - the initial acquisition image has been processed (which takes ~10s in parallel mode)
         - a new guide frame has been processed (which takes ~0.5s)
         - a new sky frame has been processed (which takes ~0.2s)

        The thread that runs this function is started in our constructor.
        """
        logging.info('ETCAlgorithm thread starting.')
        try:
            self.ETCalg.start()
            self.etc_ready.set()
        except Exception as e:
            self.etc_ready.clear()
            logging.error(f'ETCAlgorithm.start failed with: {e}.')

        last_image_processing = last_etc_processing = False

        try:
            while not self.shutdown_event.is_set():

                if self.image_processing.is_set():
                    # An exposure is active.

                    have_new_telemetry = False

                    # Any changes of state to propagate?
                    if not last_image_processing:
                        # A new exposure is starting: pass through prepare_for_exposure args now.
                        self.ETCalg.start_exposure(
                            self.img_start_time, self.expid, self.req_efftime, self.sbprof,
                            self.max_exposure_time, self.cosmics_split_time, self.maxsplit)
                        last_image_processing = True
                        # Set the path where the PNG generated after the acquisition analysis will be written.
                        self.ETCalg.set_image_path(self.call_for_exp_dir(self.expid))
                        # Look for the acquisition image and PlateMaker guide stars next.
                        need_acq_image = need_stars = True

                    elif not last_etc_processing and self.etc_processing.is_set():
                        # Shutter just opened.
                        self.ETCalg.open_shutter(
                            self.expid, self.etc_start_time, self.splittable, self.max_shutter_time)
                        last_etc_processing = True

                    elif last_etc_processing and not self.etc_processing.is_set():
                        # Shutter just closed.
                        self.ETCalg.close_shutter(self.etc_stop_time)
                        last_etc_processing = False
                        have_new_telemetry = True
                        # Save the ETC outputs for this shutter.
                        self.ETCalg.save_exposure(self.call_for_exp_dir(self.expid))
                        # Reset the PNG output path.
                        self.ETCalg.set_image_path(None)

                    # Process a sky frame if available.
                    sky_image = self.call_for_sky_image(wait=None)
                    if sky_image:
                        self.ETCalg.process_sky_frame(sky_image['image'])
                        have_new_telemetry = True

                    if need_acq_image:
                        # Process the acquisition image if available.
                        acq_image = self.call_for_acq_image(wait=None)
                        if acq_image:
                            self.ETCalg.process_acquisition(acq_image['image'])
                            img_path = self.ETCalg.image_path / f'etc-{self.expid:08d}.png'
                            if img_path.exists():
                                self.call_when_image_ready(self.expid, filename=str(img_path))
                            self.ETCalg.read_fiberassign(acq_image['fiberassign'])
                            have_new_telemetry = True
                            need_acq_image = False

                    if need_stars:
                        # Process the PlateMaker guide stars if available.
                        pm_info = self.call_for_pm_info(wait=None)
                        if pm_info:
                            self.ETCalg.set_guide_stars(pm_info['guidestars'])
                            need_stars = False

                    if not need_acq_image and not need_stars:
                        # We have PSF models and guide stars: process a guide frame if available.
                        gfa_image = self.call_for_gfa_image(wait=None)
                        if gfa_image:
                            self.ETCalg.process_guide_frame(gfa_image['image'])
                            have_new_telemetry = True

                    # Is there an action to take associated with new telemetry?
                    if have_new_telemetry and self.ETCalg.accum.action is not None:
                        action, cause = self.ETCalg.accum.action
                        if action == 'stop':
                            self.call_to_request_stop(cause)
                        elif action == 'split' and self.splittable:
                            self.call_to_request_split(cause)

                    # Send a telemetry update if something has changed.
                    if have_new_telemetry:
                        self.call_to_update_status()

                else:
                    # No exposure is active.
                    if last_image_processing:
                        # The previous exposure has just ended.
                        if last_etc_processing and not self.etc_processing.is_set():
                            # The shutter just closed for the last time, but we didn't catch it above
                            # because stop() was called too soon after stop_etc(). Do the same
                            # shutter-closed processing here as above.
                            if self.etc_stop_time is None or self.etc_stop_time < self.etc_start_time:
                                # This should never happen...
                                logging.error('Got stop() after start_etc() without any stop_etc().')
                                # We don't know when the shutter closed, so assume it was now.
                                self.etc_stop_time = datetime.datetime.utcnow()
                            else:
                                logging.warn('Got stop_etc() and stop() in rapid fire: ' +
                                    f'{self.img_stop_time - self.etc_stop_time}')
                            self.ETCalg.close_shutter(self.etc_stop_time)
                            last_etc_processing = False
                            have_new_telemetry = True
                            # Save the ETC outputs for this shutter.
                            self.ETCalg.save_exposure(self.call_for_exp_dir(self.expid))
                            # Reset the PNG output path.
                            self.ETCalg.set_image_path(None)
                            last_etc_processing = False
                        # Stop this exposure.
                        self.ETCalg.stop_exposure(self.img_stop_time)
                        last_image_processing = False

                # Need some delay here to allow the main thread to run.
                time.sleep(0.5)

        finally:
            # The shutdown event has been cleared.
            try:
                self.ETCalg.shutdown()
            except Exception as e:
                logging.error(f'ETCalg.shutdown failed: {e}')
            self.etc_ready.clear()
            logging.info('ETC: processing thread exiting after shutdown.')

    def get_status(self):
        """Return the current ETC status.

        Names used here correspond to columns in the telemetry database, so should be
        descriptive but not too verbose.

        Note that this method is called from the main thread (when changes of state
        are signalled by calls to start, stop, ...), the _etc thread (e.g. after each
        new guide frame) and from ETCApp. Therefore, this method only reads object
        attributes, in order to be thread safe.

        Each variable is updated either by the main thread or by the _etc thread,
        but never both, as indicated in the comments below.

        The returned dictionary contains only JSON serializable python types, so
        timestamps are represented as ISO-format strings and numpy floats are
        converted to python floats.
        """
        etc_status = {}

        # These variables are only updated by the main thread #################

        # Flags used to synchronize with the _etc thread.
        etc_status['img_proc'] = self.image_processing.is_set()
        etc_status['etc_proc'] = self.etc_processing.is_set()
        etc_status['etc_ready'] =  self.etc_ready.is_set()

        # Exposure parameters set in prepare_for_exposure()
        etc_status['expid'] = self.expid
        etc_status['req_efftime'] = self.req_efftime
        etc_status['sbprof'] = self.sbprof
        etc_status['max_exptime'] = self.max_exposure_time
        etc_status['cosmics_split'] = self.cosmics_split_time
        etc_status['maxsplit'] = self.maxsplit

        # Timestamps updated when start(), stop(), start_etc(), stop_etc() is called.
        etc_status['img_start_time'] = self.img_start_time.isoformat() if self.img_start_time else None
        etc_status['img_stop_time'] = self.img_stop_time.isoformat() if self.img_stop_time else None
        etc_status['etc_start_time'] = self.etc_start_time.isoformat() if self.etc_start_time else None
        etc_status['etc_stop_time'] = self.etc_stop_time.isoformat() if self.etc_stop_time else None

        # Stop sources captured by stop_etc(), stop().
        etc_status['img_stop_src'] = self.img_stop_src
        etc_status['etc_stop_src'] = self.etc_stop_src

        # The remaining variables are only updated by the _etc thread #########

        # Git description of the desietc package being used.
        etc_status['desietc'] = self.ETCalg.git or 'uknown'

        # Counters tracked by ETCalg
        etc_status['gfa_count'] = self.ETCalg.total_gfa_count
        etc_status['sky_count'] = self.ETCalg.total_sky_count
        etc_status['desi_count'] = self.ETCalg.total_desi_count

        # Observing conditions updated after each GFA or SKY frame.
        etc_status['seeing'] = self.ETCalg.seeing
        etc_status['ffrac_psf'] = self.ETCalg.ffrac
        etc_status['ffrac_elg'] = None
        etc_status['ffrac_bgs'] = None
        etc_status['ffrac'] = self.ETCalg.ffrac # for sbprof
        etc_status['ffrac_avg'] = self.ETCalg.ffrac_avg # for sbprof
        etc_status['transp'] = self.ETCalg.transp_zenith # at zenith
        etc_status['transp_avg'] = self.ETCalg.transp_avg # at zenith
        etc_status['skylevel'] = self.ETCalg.skylevel

        # ETC effective exposure time tracking.
        etc_status['last_updated'] = self.ETCalg.accum.last_updated
        etc_status['last_mjd'] = self.ETCalg.accum.last_mjd
        etc_status['signal'] = self.ETCalg.accum.signal
        etc_status['background'] = self.ETCalg.accum.background
        etc_status['efftime'] = self.ETCalg.accum.efftime
        etc_status['realtime'] = self.ETCalg.accum.realtime
        etc_status['efftime_tot'] = self.ETCalg.accum.efftime_tot
        etc_status['realtime_tot'] = self.ETCalg.accum.realtime_tot
        etc_status['remaining'] = self.ETCalg.accum.remaining
        etc_status['proj_efftime'] = self.ETCalg.accum.proj_efftime
        etc_status['next_split'] = self.ETCalg.accum.next_split
        etc_status['splittable'] = self.ETCalg.accum.splittable

        # Updated after each stop_etc.
        etc_status['fieldrot'] = None # units TBD

        # Use a roundtrip through json to convert all values to native types.
        etc_status = json.loads(json.dumps(etc_status, cls=desietc.util.NumpyEncoder))

        return etc_status

    def reset(self, all = True, keep_accumulated = False, update_status = True):
        """
        """
        logging.info('OnlineETC.reset')
        # Check that the ETC thread is still running and ready.
        self.start_thread()
        if not self.etc_ready.is_set():
            return FAILED

        # stop processing (if necessary)
        self.etc_processing.clear()

        # accumulated values:
        if keep_accumulated == False or all == True:
            self.ETCalg.reset_counts()
            self.ETCalg.accum.reset()
            self.image_processing.clear()

        # update status
        if update_status == True:
            self.call_to_update_status()

        return SUCCESS

    def configure(self):
        """
        """
        logging.info('OnlineETC.configure')
        # Check that the ETC thread is still running and ready.
        self.start_thread()
        if not self.etc_ready.is_set():
            return FAILED

        # reset internal variables
        self.reset(all=True)

        return SUCCESS

    def prepare_for_exposure(self, expid, req_efftime, sbprof, max_exposure_time,
                             cosmics_split_time, maxsplit):
        """Record the observing parameters for the next exposure, usually from NTS.

        The ETC will not see these parameters until the next call to :meth:`start`.

        Parameters
        ----------
        expid : int
            The exposure id reserved for the first shutter opening of this tile.
        req_efftime : float
            The requested target value of the effective exposure time in seconds.
        sbprof : string
            The surface brightness profile to use for FFRAC calculations. Must be one of
            PSF, ELG, BGS, FLT.
        max_exposure_time : float
            The maximum cummulative exposure time in seconds to allow for this tile,
            summed over all cosmic splits.
        cosmics_split_time : float
            The maximum exposure time in seconds for a single exposure.
        maxsplit : int
            The maximum number of exposures reserved by ICS for this tile.

        Returns
        -------
        SUCCESS or FAILED
        """
        logging.info(f'OnlineETC.prepare_for_exposure {expid}.')
        # Check that the ETC thread is still running and ready.
        self.start_thread()
        if not self.etc_ready.is_set():
            return FAILED

        # Reset status variables, keep seeing, sky level and transparency values
        self.reset(all=True, update_status=False)

        # Store this exposure's parameters.
        self.expid = expid
        self.req_efftime = req_efftime
        self.sbprof = sbprof
        self.max_exposure_time = max_exposure_time
        self.cosmics_split_time = cosmics_split_time
        self.maxsplit

        # Update our status.
        self.call_to_update_status()

        return SUCCESS

    def start(self, start_time):
        """Start processing the exposure specified in the last call to
        :meth:`prepare_for_exposure`.  The ETC will start looking for an
        acquisition image once this is called.

        Parameters
        ----------
        start_time : datetime.datetime
            UTC timestamp of when exposure processing started for this tile.

        Returns
        -------
        SUCCESS or FAILED
        """
        logging.info(f'OnlineETC.start at {start_time}.')
        # Check that the ETC thread is still running and ready.
        self.start_thread()
        if not self.etc_ready.is_set():
            return FAILED

        if not desietc.util.is_datetime(start_time):
            logging.error(f'Invalid start_time (should be datetime): {start_time}.')
            return FAILED
        self.img_start_time = start_time

        # Signal our worker thread.
        self.image_processing.set()

        # Update our status.
        self.call_to_update_status()

        return SUCCESS

    def start_etc(self, expid, start_time, splittable, max_shutter_time):
        """Signal to the ETC that the spectrograph shutters have just opened.

        The ETC will accumulate effective exposure time until the next call
        to to :meth:`stop_etc` or :meth:`stop`.
        """
        logging.info(f'OnlineETC.start_etc at {start_time}.')
        # Check that the ETC thread is still running and ready.
        self.start_thread()
        if not self.etc_ready.is_set():
            return FAILED

        # Check parameters.
        if not isinstance(expid, int):
            logging.error(f'Invalid expid (should be int): {expid}.')
            return FAILED
        if not desietc.util.is_datetime(start_time):
            logging.error(f'Invalid start_time (should be datetime): {start_time}.')
            return FAILED
        if max_shutter_time <= 0 or max_shutter_time > 6 * 3600:
            logging.error(f'Invalid max_shutter_time: {max_shutter_time}.')
            return FAILED
        self.expid = expid
        self.etc_start_time = start_time
        self.splittable = splittable
        self.max_shutter_time = max_shutter_time

        # Signal our worker thread.
        self.etc_processing.set()

        # Update our status.
        self.call_to_update_status()

        return SUCCESS

    def stop_etc(self, source, stop_time):
        """Signal to the ETC that the spectograph shutters have just closed.

        The ETC will continue processing any new sky or guide frames until
        :meth:`stop` is called.
        """
        logging.info(f'OnlineETC.stop_etc at {stop_time} from "{source}".')
        # Check that the ETC thread is still running and ready.
        self.start_thread()
        if not self.etc_ready.is_set():
            return FAILED

        if not desietc.util.is_datetime(stop_time):
            logging.error(f'Invalid stop_time (should be datetime): {stop_time}.')
            return FAILED
        self.etc_stop_time = stop_time
        self.etc_stop_src = source

        # Signal our worker thread.
        self.etc_processing.clear()

        return SUCCESS

    def stop(self, source, stop_time):
        """Signal to the ETC that the current exposure has stopped.

        The ETC will save its processing history for this exposure after this call.

        In case stop is called while the spectrograph shutters are open,
        log an error and clear etc_processing before clearing image_processing.
        """
        logging.info(f'OnlineETC.stop at {stop_time} from "{source}".')
        # Check that the ETC thread is still running and ready.
        self.start_thread()
        if not self.etc_ready.is_set():
            return FAILED

        if not desietc.util.is_datetime(stop_time):
            logging.error(f'Invalid stop_time (should be datetime): {stop_time}.')
            return FAILED
        self.img_stop_time = stop_time
        self.img_stop_src = source

        if self.etc_processing.is_set():
            logging.error('stop: called before stop_etc.')
            self.etc_stop_time = self.img_stop_time
            self.etc_processing.clear()

        # Signal our worker thread that image processing should start.
        self.image_processing.clear()

        # Update our status.
        self.call_to_update_status()

        return SUCCESS

    def shutdown(self, timeout=30):
        """Terminate our worker thread, which will release ETCalg resources.

        In case the worker thread does not exit within timeout, call ETCalg.shutdown
        directly to force it to release any resources it has allocated.
        """
        logging.info('ETC: shutdown called.')
        # Signal to the worker thread that we are shutting down.
        self.shutdown_event.set()
        if self.etc_thread.is_alive():
            self.etc_thread.join(timeout=timeout)
        if self.etc_thread.is_alive():
            logging.error(f'The ETC worker thread did not exit after {timeout}s.')
            try:
                self.ETCalg.shutdown()
            except Exception as e:
                logging.error(f'ETCalg.shutdown failed with {e}')
        self.etc_ready.clear()
        self.etc_thread = None
        return SUCCESS
