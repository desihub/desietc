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

        # status
        self.target_snr = None
        self.estimated_snr = 0.0       # cummulative across cosmics splits
        self.current_snr = 0.0         # snr for current spectrograph exposure
        self.will_not_finish = False
        self.about_to_stop_snr = 0.0
        self.about_to_stop_remaining = 0.0
        self.skylevel = None
        self.seeing = None
        self.transparency = None
        self.image_processing_start_time = None
        self.image_processing_last_update = None
        self.etc_processing_start_time = None
        self.etc_processing_last_update = None
        self.accumulated_etc_processing_time = 0.0     # integerated exposure time processing time across cosmics splits
        self.current_etc_processing_time = 0.0         # current exposure time processing time
        self.max_exposure_time = None
        self.cosmics_split_time = None
        self.current_expid = None     # initial exposure id from prepare_for_exposure
        self.current_specid = None    # exposure id of current spectrograph exposure
        self.stop_cause = None        # why was exposure stopped
        self.guider_count = 0         # cummulative guider frame count
        self.sky_count = 0            # cummulative sky frame count
        self.acq_count = 0            # cummulative acq frame count
        self.pm_count = 0             # cummulative pm info count
        self.etc_cycle_count = 0
        self.max_count = 99999
        self.announcement_made = {'stop':None,'split':None,'about_to_stop':None,'about_to_split':None}
        self.stop_cause = None        # why was exposure stop requested
        self.split_cause = None        # why was exposure split requested
        self.about_to_stop_cause = None        # why was exposure about to stop
        self.about_to_split_cause = None        # why was exposure about to split

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
                    self.ETCalg.process_sky(sky_image['image'])
                    have_new_telemetry = True

                elif not shutter_open and self.etc_processing.is_set():
                    # Spectrograph shutter has just opened: start ETC tracking.
                    now = datetime.datetime.utcnow()
                    mjd = desietc.util.date_to_mjd(now, utc_offset=0)
                    cutoff = mjd + self.max_exposure_time / self.ETCalg.SECS_PER_DAY
                    self.ETCalg.start_exposure(
                        self.night, self.expid, mjd, self.target_teff, cutoff, self.cosmics_split_time)
                    need_acq_image = need_stars = shutter_open = True

                elif shutter_open:

                    if not self.etc_processing.is_set():
                        # Shutter has just closed: save ETC outputs.
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

    def reset(self, all = True, keep_accumulated = False, update_status = True):

        # stop processing (if necessary)
        self.etc_processing.clear()

        # accumulated values:
        if keep_accumulated == False or all == True:
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
        self.stars = stars

        # set snr, remaining time thresholds for stop announcement
        self.about_to_stop_snr = self.target_snr * 0.9                # 90% of SNR collected
        self.about_to_stop_remaining = min(self.max_exposure_time * 0.9, (self.max_exposure_time - 20.0) if self.max_exposure_time>20.0 else self.max_exposure_time)   # 90% of max. exposure time or at least 20 s

        # update status
        self.call_to_update_status()

        return SUCCESS

    def start(self, **options):
        """
        start etc image processing
        """
        Log.info('start: start image processing: %r' % options)
        self.cosmics_split_time = options.get('comics_split_time', self.cosmics_split_time)
        self.image_processing_start_time = options.get('start_time', datetime.datetime.utcnow())

        # reset some internal values - again (a bit redundant)
        self.announcement_made = {'stop':None,'split':None,'about_to_stop':None,'about_to_split':None}
        self.stop_cause = None        # why was exposure stop requested
        self.split_cause = None        # why was exposure split requested
        self.about_to_stop_cause = None        # why was exposure about to stop
        self.about_to_split_cause = None        # why was exposure about to split
        self.estimated_snr = 0.0
        self.current_snr = 0.0
        self.etc_cycle_count = 0
        self.guider_count = 0
        self.sky_count = 0
        self.acq_count = 0
        self.pm_count = 0
        self.etc_cycle_count = 0
        self.image_processing_start_time = options.get('start_time', datetime.datetime.utcnow())
        self.image_processing_last_update = options.get('start_time', datetime.datetime.utcnow())
        self.accumulated_etc_processing_time = 0.0
        self.current_etc_processing_time = 0.0
        self.image_processing.set()
        # update status
        if callable(self.call_to_update_status):
            self.call_to_update_status()

    def start_etc(self, start_time = None,  splittable = None, specid = None, **options):
        """
        start etc exposure time  processing
        """
        # reset announcements
        self.announcement_made = {'stop':None,'split':None,'about_to_stop':None,'about_to_split':None}
        self.stop_cause = None        # why was exposure stop requested
        self.split_cause = None        # why was exposure split requested
        self.about_to_stop_cause = None        # why was exposure about to stop
        self.about_to_split_cause = None        # why was exposure about to split
        if splittable is None:
            splittable = False
        self.splittable = splittable
        Log.info('start_etc (%r): start exposure time processing (splittable = %r)' % (self.current_expid, self.splittable))
        self.etc_processing_start_time = start_time

        # initialize current variables
        self.current_snr = 0.0
        self.current_etc_processing_time = 0.0
        self.etc_processing_start_time = options.get('start_time', datetime.datetime.utcnow())
        self.etc_processing_last_update = options.get('start_time', datetime.datetime.utcnow())
        self.etc_processing.set()
        # update status
        if callable(self.call_to_update_status):
            self.call_to_update_status()

    def stop(self, source = 'OPERATOR', stop_time = None, **options):
        """
        Force ETC image processing to stop
        """
        Log.info('stop: stop image processing request received from source %s' % source)
        self.image_processing.clear()
        # update status
        if callable(self.call_to_update_status):
            self.call_to_update_status()

        # add whatever else needs to be done

        return SUCCESS

    def stop_etc(self, source = 'OPERATOR', **options):
        """
        Force ETC exposure time processing to stop
        """
        Log.info('stop_etc: stop exposure time processing request received from source %s' % source)
        self.etc_processing.clear()
        # update status
        if callable(self.call_to_update_status):
            self.call_to_update_status()

        # add whatever else needs to be done

        return SUCCESS

    def get_status(self):
        self.last_updated = datetime.datetime.utcnow().isoformat()
        etc_status = {}
        etc_status['last_updated'] = self.last_updated

        etc_status['expid'] = self.current_expid
        etc_status['specid'] = self.current_specid
        etc_status['splittable'] = self.splittable
        etc_status['estimated_snr'] = None if not isinstance(self.estimated_snr, (int, float)) else round(self.estimated_snr, 3)
        etc_status['current_snr'] = None if not isinstance(self.current_snr, (int, float)) else round(self.current_snr, 3)
        etc_status['target_snr'] = self.target_snr
        etc_status['will_not_finish'] = self.will_not_finish
        etc_status['stop_request'] = self.announcement_made['stop']
        etc_status['about_to_stop'] = self.announcement_made['about_to_stop']
        etc_status['about_to_stop_cause'] = self.about_to_stop_cause
        etc_status['about_to_stop_snr'] = self.about_to_stop_snr
        etc_status['about_to_stop_remaining'] = self.about_to_stop_remaining
        etc_status['split_request'] = self.announcement_made['split']
        etc_status['about_to_split'] = self.announcement_made['about_to_split']
        etc_status['about_to_split_cause'] = self.about_to_split_cause
        etc_status['seeing'] = None if not isinstance(self.seeing, (int, float)) else round(self.seeing, 3)
        etc_status['transparency'] = None if not isinstance(self.transparency, (int, float)) else round(self.transparency, 3)
        etc_status['skylevel'] = None if not isinstance(self.skylevel, (int, float)) else round(self.skylevel, 3)
        etc_status['image_processing_start_time'] = self.image_processing_start_time
        etc_status['processing_images'] = self.image_processing.is_set()
        etc_status['etc_processing_start_time'] = self.etc_processing_start_time
        etc_status['accumulated_etc_processing_time'] = round(self.accumulated_etc_processing_time,3)
        etc_status['processing_etc'] = self.etc_processing.is_set()
        etc_status['max_exposure_time'] = self.max_exposure_time
        etc_status['cosmics_split_time'] = self.cosmics_split_time
        etc_status['total_guider_count'] = self.guider_count
        etc_status['total_sky_count'] = self.sky_count
        etc_status['total_acq_count'] = self.acq_count
        etc_status['total_pm_count'] = self.pm_count
        etc_status['etc_cycles'] = self.etc_cycle_count
        return etc_status

########################## End of ETC Class ########################
