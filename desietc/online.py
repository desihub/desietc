"""Online ETC class that intefaces with ICS via callouts implemented by ETCApp.

Original code written by Klaus Honscheid and copied here 16-Feb-2021 from
https://desi.lbl.gov/trac/browser/code/online/ETC/trunk/python/ETC/ETC.py

ETCApp at https://desi.lbl.gov/trac/browser/code/online/ETC/trunk/python/ETC/ETCApp.py

5 callouts are used to interact with the ETCapp and the rest of DOS.
The basic idea is the the etc code checks if the callout variable is callable and then executes the function at
the appropriate time. For example, for a status update, ETCapp sets the callout:

    self.etc.call_to_update = self._update_status

and the etc class could include code like this:

    if callable(self.call_to_update_status):
        self.call_to_update_status()

Arguments can be passed as needed - of course this has to be coordinated with ETCapp (the _update_status method
in the example)

This is the list of current call outs:
    call_for_sky_image
    call_for_gfa_image
    call_for_acq_image
    call_for_pm_info
    call_for_telemetry
    call_to_update_status
    call_to_request_stop
    call_when_about_to_stop
    call_to_request_split
    call_when_about_to_split
    call_when_image_ready

After configure,  start/stop calls control image processing
When active, i.e. processing images, exposure time processing is controlled by the start_etc/stop_etc commands
"""
import datetime
import sys
import os
import threading
import time

from DOSlib.util import raise_error
import DOSlib.logger as Log
from DOSlib.PML import SUCCESS, FAILED

import desietc.etc


class OnlineETC():

    def __init__(self, shutdown_event):

        self.shutdown = shutdown_event

        # callouts to ETC application
        self.call_for_sky_image = None
        self.call_for_gfa_image = None
        self.call_for_acq_image = None
        self.call_for_pm_info = None
        self.call_for_telemetry = None
        self.call_to_update_status = None
        self.call_to_request_stop = None
        self.call_when_about_to_stop = None
        self.call_to_request_split = None
        self.call_when_about_to_split = None
        self.call_when_image_ready = None

        # configuration
        self.image_name = None
        self.png_dir = None
        self.use_obs_day = None
        self.use_exp_dir = None
        self.simulate = False
        self.splittable = False

        # Initialize the ETC algorithm.
        gfa_calib = os.getenv('ETC_GFA_CALIB', None)
        sky_calib = os.getenv('ETC_SKY_CALIB', None)
        if gfa_calib is None or sky_calib is None:
            raise RuntimeError('ETC_GFA_CALIB and ETC_SKY_CALIB must be set.')
        self.ETCalg = desietc.etc.ETCAlgorithm(gfa_calib, sky_calib)

        # Start our processing thread and create the flags we use to synchronize with it.
        self.image_processing = threading.Event()
        self.image_processing.clear()
        self.etc_processing = threading.Event()
        self.etc_processing.clear()
        self.etc_thread = threading.Thread(target = self._etc)
        self.etc_thread.daemon = True
        self.etc_thread.start()
        Log.info('ETC: processing thread running')

    def shutdown(self):
        """Release resources allocated for the ETC algorithm in our constructor.
        """
        self.ETCalg.shutdown()

    def _etc(self):
        """This is the ETC algorithm that does the actual work.

        This function normally runs in a separate thread and synchronizes with the
        rest of ICS via two flags:

         - image_processing: set to indicate that the following are or will soon be available:
            - fiberassign file
            - acquisition image
            - PlateMaker guide stars

         - etc_processing: set when the spectrograph exposure has started and cleared when we
            should save our outputs to the exposure directory.

        The thread that runs this function is started in our constructor.
        """
        Log.info('ETC: processing thread starting.')
        while not self.shutdown.is_set():

            shutter_open = False

            if self.image_processing.is_set():

                have_new_telemetry = False

                # Always process a sky frame if available.
                sky_image = self.call_for_sky_image()
                if sky_image:
                    self.ETCalg.process_sky_frame(sky_image['image'])
                    have_new_telemetry = True

                elif not shutter_open and self.etc_processing.is_set():
                    # Spectrograph shutter has just opened: start ETC tracking.
                    mjd = desietc.util.date_to_mjd(self.etc_proc_start, utc_offset=0)
                    self.ETCalg.start_exposure(
                        self.night, self.expid, mjd, self.target_teff,
                        self.max_exposure_time, self.cosmics_split_time)
                    need_acq_image = need_stars = shutter_open = True

                elif shutter_open:

                    if not self.etc_processing.is_set():
                        # Shutter has just closed: get final estimates and save ETC outputs.
                        mjd = desietc.util.date_to_mjd(self.etc_proc_stop, utc_offset=0)
                        self.ETCalg.stop_exposure(mjd)
                        self.ETCalg.save_exposure(self.get_exp_dir())
                        shutter_open = False
                        continue

                    elif need_acq_image:
                        # Process the acquisition image if available.
                        acq_image = self.call_for_acq_image(wait=None)
                        if acq_image:
                            self.ETCalg.process_acquisition(acq_image['image'])
                            self.ETCalg.read_fiberassign(acq_image['fiberassign'])
                            have_new_telemetry = True
                            need_acq_image = False

                    elif need_stars:
                        # Process the PlateMaker stars if available.
                        pm_info = self.call_for_pm_info(wait=None)
                        if pm_info:
                            self.ETCalg.set_guide_stars(pm_info['guidestars'])
                            need_stars = False

                    elif not (need_acq_image or need_stars):
                        # Process a guide frame if available.
                        gfa_image = self.call_for_gfa_image(wait=None)
                        if gfa_image:
                            self.ETCalg.process_guide_frame(gfa_image['image'])
                            have_new_telemetry = True

                if have_new_telemetry:
                    self.call_to_update_status()

            else:
                Log.info('_etc (%r): Image processing complete' % self.current_expid)
                shutter_open = False

        Log.info('ETC: processing thread exiting.')

    def get_status(self):
        """Capture and return the current ETC status.

        Names used here correspond to columns in the telemetry database, so should be
        descriptive but not too verbose.
        """
        etc_status = {}

        # Use now as the timestamp for this status update.
        etc_status['last_updated'] = datetime.datetime.utcnow().isoformat()

        # Exposure parameters set in prepare_for_exposure()
        etc_status['expid'] = self.current_expid
        etc_status['target_teff'] = self.target_teff
        etc_status['max_exptime'] = self.max_exposure_time
        etc_status['cosmics_split'] = self.cosmics_split_time

        # Exposure parameters set in start_etc()
        etc_status['specid'] = self.current_specid
        etc_status['splittable'] = self.splittable

        # Timestamps captured by start(), start_etc(), stop_etc()
        etc_status['img_proc_start'] = self.img_proc_start
        etc_status['etc_proc_start'] = self.etc_proc_start
        etc_status['etc_proc_stop'] = self.etc_proc_stop

        # Flags used to synchronize with the _etc thread.
        etc_status['img_proc'] = self.image_processing.is_set()
        etc_status['etc_proc'] = self.etc_processing.is_set()

        # Counters tracked by ETCalg
        etc_status['guider_count'] = self.ETCalg.total_guider_count
        etc_status['sky_count'] = self.ETCalg.total_sky_count
        etc_status['acq_count'] = self.ETCalg.total_acq_count

        # Observing conditions updated after each GFA or SKY frame.
        etc_status['seeing'] = self.ETCalg.fwhm
        etc_status['ffrac'] = self.ETCalg.ffrac
        etc_status['transparency'] = self.ETCalg.transp
        etc_status['skylevel'] = self.ETCalg.skylevel

        # ETC effective exposure time tracking.
        etc_status['accum_mjd'] = self.ETCalg.last_update_mjd
        etc_status['accum_sig'] = self.ETCalg.accumulated_signal
        etc_status['accum_bg'] = self.ETCalg.accumulated_background
        etc_status['accum_teff'] = self.ETCalg.accumulated_eff_time
        etc_status['accum_real'] = self.ETCalg.accumulated_real_time

        return etc_status

    def reset(self, all = True, keep_accumulated = False, update_status = True):
        """
        """
        # stop processing (if necessary)
        self.etc_processing.clear()

        # accumulated values:
        if keep_accumulated == False or all == True:
            self.ETCalg.reset_counts()
            self.ETCalg.reset_accumulated()
            self.image_processing.clear()

        # update status
        if update_status == True:
            self.call_to_update_status()

    def configure(self):
        """
        ETC configure - to be completed
        """
        # reset internal variables
        self.reset(all=True)

        # check if _etc thread is still running
        if not self.etc_thread.is_alive():
            self.etc_thread = threading.Thread(target = self._etc)
            self.etc_thread.daemon = True
            self.etc_thread.start()
            Log.warn('ETC: processing thread restarted')
        else:
            Log.info('ETC: processing thread still running')

        Log.info('configure: ETC is ready')
        return SUCCESS

    def prepare_for_exposure(self, expid, target_teff, max_exposure_time, cosmics_split_time = None, count=None, stars = None):
        """
        Prepare ETC for the next exposure
        expid:             next exposure id (int)
        target_teff:       target value of the effective exposure time in seconds (float)
        max_exposure_time: Maximum exposure time in seconds (irrespective of accumulated SNR)
        comics_split_time: Time in second before requesting a cosmic ray split
        stars : Numpy recarray with star information (copy of gfa_targets table from fiberassign)
        count : number of frames to process (Guider)
        """
        assert isinstance(expid, int) and isinstance(snr, (int, float)) and isinstance(max_exposure_time, (int, float)),'Invalid arguments'
        Log.info('ETC (%d): prepare_for_exposure called with requested SNR %f' % (expid, snr))

        # check if _etc thread is still running
        if not self.etc_thread.is_alive():
            self.etc_thread = threading.Thread(target = self._etc)
            self.etc_thread.daemon = True
            self.etc_thread.start()
            Log.warn('ETC (%d): processing thread restarted' % expid)
        else:
            Log.info('ETC (%d): processing thread still running' % expid)

        # Reset status variables, keep seeing, sky level and transparency values
        self.reset(all=True, update_status = False)

        # and store calling arguments
        self.target_teff = target_teff
        self.max_exposure_time = max_exposure_time
        self.cosmics_split_time = cosmics_split_time
        self.current_expid = expid
        if isinstance(count, int):
            self.max_count = count
        else:
            self.max_count = 999999

        # update status
        self.call_to_update_status()

        return SUCCESS

    def start(self, start_time=None, **options):
        """
        start etc image processing
        """
        self.img_proc_start = start_time or datetime.datetime.utcnow()
        Log.info('start: start image processing at %r' % self.img_proc_start)
        if options:
            Log.warn('start: ignoring extra options: %r' % options)

        # Signal our worker thread that image processing should start.
        self.image_processing.set()

        # Update our status.
        self.call_to_update_status()

    def start_etc(self, start_time = None,  splittable = None, specid = None, **options):
        """
        start etc exposure time processing
        """
        self.etc_proc_start = start_time or datetime.datetime.utcnow()
        self.splittable = splittable or False
        self.current_specid = specid
        Log.info('start_etc: start etc processing at %r with splittable=%r, specid=%r' % (
            self.etc_proc_start, self.splittable, self.current_specid))
        if options:
            Log.warn('start_etc: ignoring extra options: %r' % options)

        # Signal our worker thread that etc processing should start.
        self.image_processing.set()

        # Update our status.
        self.call_to_update_status()

    def stop_etc(self, source='OPERATOR', stop_time=None, **options):
        """
        Force ETC exposure time processing to stop
        """
        self.etc_proc_stop = stop_time or datetime.datetime.utcnow()
        Log.info('stop_etc: stop etc processing at %r from source %s ' % (self.etc_proc_stop, source))
        if options:
            Log.warn('stop_etc: ignoring extra options: %r' % options)

        # Signal our worker thread that etc processing should stop.
        self.etc_processing.clear()

        # Update our status.
        self.call_to_update_status()

    def stop(self, source = 'OPERATOR', **options):
        """
        Force ETC image processing to stop
        """
        Log.info('stop: stop image processing request received from source %s' % source)
        if options:
            Log.warn('stop: ignoring extra options: %r' % options)

        # Signal our worker thread that image processing should start.
        self.image_processing.clear()

        # Update our status.
        self.call_to_update_status()
