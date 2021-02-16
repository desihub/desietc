#!/usr/bin/env python3

"""
The ETC class includes the actual ETC algorithms.
Communication with the ETCApp framework is done via callouts
Currently, the image queues are managed by the ETCApp, not the ETC, but this can be changed if necessary

Log.debug  (or info, warn, error) are logger functions (messages are displayed on screen and added to log files
for archive and debugging purposes)

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

from DOSlib.util import raise_error
import DOSlib.logger as Log
from DOSlib.PML import SUCCESS, FAILED
import datetime
import sys
import os
import threading
import time
import random

class ETC():
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
        self.access_etcapp = self._app_access
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
        self.last_updated = datetime.datetime.utcnow().isoformat()
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
        # start processing thread
        self.image_processing = threading.Event()
        self.image_processing.clear()
        self.etc_processing = threading.Event()
        self.etc_processing.clear()
        self.etc_thread = threading.Thread(target = self._etc)
        self.etc_thread.daemon = True
        self.etc_thread.start()
        Log.info('ETC: processing thread running')

    def _app_access(self, *args, **kwargs):
        """
        To be overloaded
        """
        pass

    def _etc(self):
        """
        etc worker thread
        request images and run algorithms. Maybe split into separate threads for gfa and skycam images
        simulated algorithm: "measure" seeing, skylevel and transparency when ACTIVE, accumulate SNR when PROCESSING
        """
        step = 0.01
        image_processing_time = 3.0    # for simulation
        etc_processing_time = 2.0      # for simulation
        while not self.shutdown.is_set():
            print_once = True
            while not self.image_processing.is_set() and not self.shutdown.is_set():
                if print_once:
                    Log.info('_etc: waiting for exposure to start')
                    print_once = False
                time.sleep(1)
            # Starting
            last_expid = None
            last_tele = None
            while self.image_processing.is_set() and not self.shutdown.is_set():
                if last_expid != self.current_expid:
                    Log.info('_etc: Now processing exposure %d' % self.current_expid)
                    last_expid = self.current_expid
                # request images
                if callable(self.call_for_gfa_image):
                    gfa_image = None
                    #while gfa_image == None and self.image_processing.is_set() and not self.shutdown.is_set():
                    gfa_image = self.call_for_gfa_image(wait = 1.0)
                    if gfa_image is not None:
                        Log.info(f"_etc: Got acquisition image for expid {gfa_image.get('expid', 'na')}, frame {gfa_image.get('frame', 'na')}")
                        self.guider_count += 1
                # maybe this should be done in parallel and not sequential
                if callable(self.call_for_sky_image):
                    sky_image = None
                    #while sky_image == None and self.image_processing.is_set() and not self.shutdown.is_set():
                    sky_image = self.call_for_sky_image(wait = 1.0)
                    if sky_image is not None:
                        self.sky_count += 1
                # maybe this should be done in parallel and not sequential
                if callable(self.call_for_acq_image):
                    acq_image = None
                    acq_image = self.call_for_acq_image(wait = 1.0)
                    if acq_image is not None:
                        Log.info(f"_etc: Got acquisition image for expid {acq_image.get('expid', 'na')}, frame {acq_image.get('frame', 'na')}")
                        self.acq_count += 1
                # maybe this should be done in parallel and not sequential
                if callable(self.call_for_pm_info):
                    pm_info = None
                    pm_info = self.call_for_pm_info(wait = 1.0)
                    if pm_info is not None:
                        self.pm_count += 1
                # get telemetry update before start image processing
                if callable(self.call_for_telemetry):
                    # for debugging only once for new exposures
                    if last_tele != self.current_expid:
                        telemetry = self.call_for_telemetry()
                        Log.info('Telemetry example: wind speed %r, wind direction %r' % (telemetry.get('wind_speed', 'na'), telemetry.get('wind_direction', 'na')))
                        last_tele = self.current_expid

                # do whatever the etc does, update statistics
                if gfa_image == None:
                    pass  # for now update fake telemetry # continue
                self.etc_cycle_count += 1
                if gfa_image is not None:
                    Log.info('_etc: processing gfa_image (%r)' % self.etc_cycle_count)
                wait = time.time()
                while time.time()<wait+image_processing_time:
                    if not self.image_processing.is_set():
                        break
                    time.sleep(0.2)
                if callable(self.call_when_image_ready):
                    self.call_when_image_ready(self.current_expid, frame=self.etc_cycle_count, filename = None)

                # update telemetry
                if isinstance(self.transparency, float):
                    self.transparency = self.transparency + step if random.random() <= 0.5 else self.transparency - step
                    if self.transparency > 1.0 or self.transparency < 0.0:
                        self.transparency = 0.8
                if isinstance(self.skylevel, float):
                    self.skylevel = self.skylevel + step if random.random() <= 0.5 else self.skylevel - step
                    if self.skylevel < 0.0:
                        self.skylevel = 21.0
                if isinstance(self.seeing, float):
                    self.seeing = self.seeing + step if random.random() <= 0.5 else self.seeing - step
                    if self.seeing < 0.0:
                        self.seeing = 1.0

                self.image_processing_last_update = datetime.datetime.utcnow()
                self.last_updated = datetime.datetime.utcnow().isoformat()

                # Run exposure time process
                if self.etc_processing.is_set():
                    Log.info('_etc (%d): performing exposure time processing' % self.current_expid)
                    new_snr = 0.01 + 0.005 * random.random()
                    self.estimated_snr += new_snr
                    self.current_snr += new_snr

                    now =  datetime.datetime.utcnow()
                    delta = (now - self.etc_processing_last_update).total_seconds()
                    self.etc_processing_last_update = now
                    self.accumulated_etc_processing_time += delta
                    self.current_etc_processing_time += delta

                    Log.info('_etc (%d): estimated SNR: %r, current SNR: %r, accumulated time: %r, current acc. time: %r' % (self.current_expid, round(self.estimated_snr, 3),
                                                                                                                             round(self.current_snr, 3),
                                                                                                                             self.accumulated_etc_processing_time,
                                                                                                                             self.current_etc_processing_time))
                # done exposure time processing

                # are we done? The logic coded here is just for demo purposes
                if isinstance(self.target_snr, (int, float)) and self.estimated_snr >= self.target_snr:
                    self.stop_cause = 'SNR'
                    # should etc loop be stopped now? Or wait for external stop command?
                if isinstance(self.max_exposure_time, (int, float)) and self.accumulated_etc_processing_time >= (self.max_exposure_time+5):
                    self.stop_cause = 'MAXTIME'
                    # should etc loop be stopped now? Or wait for external stop command?
                if self.accumulated_etc_processing_time > self.about_to_stop_remaining:
                    self.about_to_stop_cause = 'remaining'
                elif self.estimated_snr > self.about_to_stop_snr:
                    self.about_to_stop_cause = 'snr'
                if isinstance(self.cosmics_split_time, (float, int)):
                    if (self.current_etc_processing_time > self.cosmics_split_time):
                        self.split_cause = 'SPLIT'
                    elif (self.current_etc_processing_time > (self.cosmics_split_time*0.9)):
                        self.about_to_split_cause = 'SPLIT'
                if self.etc_cycle_count >= self.max_count:
                    self.stop_cause = 'COUNT'

                # inform ETCapp that new information is available (pass information or should ETCapp read attributes?)
                if callable(self.call_to_update_status):
                    self.call_to_update_status()

                # Any action requested this cycle?
                if self.stop_cause != None and not self.announcement_made['stop']:
                    if callable(self.call_to_request_stop):
                        self.call_to_request_stop(cause = self.stop_cause)
                    self.announcement_made['stop'] = True
                elif self.about_to_stop_cause != None and not self.announcement_made['about_to_stop']:
                    if callable(self.call_when_about_to_stop):
                        self.call_when_about_to_stop(cause = self.about_to_stop_cause)
                    self.announcement_made['about_to_stop'] = True
                if self.split_cause != None and not self.announcement_made['split'] and self.splittable:
                    if callable(self.call_to_request_split):
                        self.call_to_request_split(cause = self.split_cause)
                    self.announcement_made['split'] = True
                elif self.about_to_split_cause != None and not self.announcement_made['about_to_split'] and self.splittable:
                    if callable(self.call_when_about_to_split):
                        self.call_when_about_to_split(cause = self.about_to_split_cause)
                    self.announcement_made['about_to_split'] = True
            Log.info('_etc (%r): Image processing complete' % self.current_expid)
        Log.info('ETC: processing thread exists')

    def reset(self, all = True, keep_accumulated = False, update_status = True):
        self.current_snr = 0.0
        self.current_etc_processing_time = 0.0
        self.will_not_finish = False
        self.image_processing_start_time = None
        self.image_processing_last_update = None
        self.etc_processing_start_time = None
        self.etc_processing_last_update = None
        self.stop_cause = None
        self.announcement_made = {'stop':None,'split':None,'about_to_stop':None,'about_to_split':None}
        self.splittable = False

        # stop processing (if necessary)
        self.etc_processing.clear()

        # accumulated values:
        if keep_accumulated == False or all == True:
            self.accumulated_etc_processing_time = 0.0
            self.estimated_snr = 0.0
            self.max_count = 99999
            self.guider_count = 0
            self.sky_count = 0
            self.acq_count = 0
            self.pm_count = 0
            self.etc_cycle_count = 0
            self.image_processing.clear()

        # Clear old exposure info unless all is not True
        if all == True:
            self.target_snr = 0.0
            self.current_expid = None
            self.current_specid = None
            self.cosmics_split_time = None
            self.max_exposure_time = None
            self.about_to_stop_snr = 0.0
            self.about_to_stop_remaining = 0.0

        # update status
        if callable(self.call_to_update_status) and update_status == True:
            self.call_to_update_status()

    def configure(self, contants = 'DEFAULT'):
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

    def prepare_for_exposure(self, expid, snr, max_exposure_time, cosmics_split_time = None, count=None, stars = None):
        """
        Prepare ETC for the next exposure
        expid:             next exposure id (int)
        snr:               Signal/Noise target for next exposure (float)
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
        self.target_snr = snr
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
        if callable(self.call_to_update_status):
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
        etc_status = {}
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
        etc_status['last_updated'] = self.last_updated
        return etc_status

########################## End of ETC Class ########################
