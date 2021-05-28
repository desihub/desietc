"""To test OnlineETC outside of ICS, start the driver program with:

   python -m desietc.app

then, at the command line prompt (#), enter the following sequence of commands (for example):

s     # start a new exposure
f     # playback the first GFA frame
o     # open the spectrograph shutters
f 20  # playback the next 20 GFA/SKY frames
c     # close the spectrograph shutters
?     # request a telemetry update
t     # stop the current exposure
q     # shutdown the OnlineETC instance
"""
import os
import sys
import traceback
import logging
import threading
import pathlib
import time
import datetime
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import desietc.online
import desietc.offline
import desietc.util


# Hijack online.get_utcnow() to return a value we can set.
_offset = datetime.timedelta()

def set_mjd_now(mjd):
    global _offset
    then = desietc.util.mjd_to_date(mjd, utc_offset=0)
    now = datetime.datetime.utcnow()
    _offset = now - then

def hijacked_utcnow():
    global _offset
    utcnow = datetime.datetime.utcnow() - _offset
    #print(f'utcnow is {utcnow}')
    return utcnow

desietc.online.get_utcnow = hijacked_utcnow


class OfflineETCApp:
    """Emulation of the ICS ETCApp that can be used outside ICS to test OnlineETC.
    """
    def __init__(self, max_telemetry_secs=30):
        # Initialize the ETC.
        self.shutdown_event = threading.Event()
        self.shutdown_event.clear()
        self.etc = desietc.online.OnlineETC(self.shutdown_event, max_telemetry_secs=max_telemetry_secs)
        # Prepare to handle callouts.
        self.etc.call_to_request_stop = self.call_to_request_stop
        self.etc.call_to_request_split = self.call_to_request_split
        self.etc.call_when_about_to_stop = self.call_when_about_to_stop
        self.etc.call_when_about_to_split = self.call_when_about_to_split
        self.etc.call_to_update_status = self.call_to_update_status
        self.etc.call_for_acq_image = self.call_for_acq_image
        self.etc.call_for_pm_info = self.call_for_pm_info
        self.etc.call_when_image_ready = self.call_when_image_ready
        self.etc.call_for_sky_image = lambda: self.call_for_frame('sky')
        self.etc.call_for_gfa_image = lambda: self.call_for_frame('gfa')
        self.expdir = pathlib.Path('expdir')
        self.etc.call_for_exp_dir = self.call_for_exp_dir
        self.status_updates = []
        self.assets = None
        self.next_frame = 0
        # Configure the ETC.
        assert self.etc.configure()

    def call_for_exp_dir(self, expid):
        path = self.expdir / str(self.etc.ETCalg.night) / self.etc.ETCalg.exptag
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def call_to_update_status(self):
        status = self.etc.get_status()
        logging.info(f'get_status: {status}')
        self.status_updates.append(status)

    def call_when_image_ready(self, expid, frame=0, filename=None):
        print(f'Image ready for {expid}[{frame}] at {filename}')

    def call_to_request_stop(self, cause):
        logging.info(f'request_stop: cause={cause}')
        self.close_shutter()

    def call_to_request_split(self, cause):
        logging.info(f'request_split: cause={cause}')
        self.close_shutter()
        self.expid += 1

    def call_when_about_to_stop(self, cause):
        logging.info(f'about_to_stop: cause={cause}')

    def call_when_about_to_split(self, cause):
        logging.info(f'about_to_split: cause={cause}')

    def get(self, key):
        return None if self.assets is None else self.assets.get(key, None)

    def start_exposure(self, assets, req_efftime, sbprof, max_exposure_time, cosmics_split_time,
                       maxsplit, warning_time):
        self.assets = assets
        self.expid = self.get('expid')
        self.next_frame = 0
        self.last_frame = 0
        self.etc.prepare_for_exposure(expid=self.expid, req_efftime=req_efftime, sbprof=sbprof,
                                      max_exposure_time=max_exposure_time,
                                      cosmics_split_time=cosmics_split_time, maxsplit=maxsplit,
                                      warning_time=warning_time)
        set_mjd_now(desietc.util.date_to_mjd(self.get('start_time'), utc_offset=0))
        return self.etc.start(start_time=self.get('start_time'))

    def open_shutter(self, splittable=True, max_shutter_time=3600):
        start_mjd = self.get('frames')[self.last_frame]['start']
        set_mjd_now(self.get('frames')[self.last_frame]['stop'])
        start_time = desietc.util.mjd_to_date(start_mjd, utc_offset=0)
        return self.etc.start_etc(
            expid=self.expid, start_time=start_time, splittable=splittable, max_shutter_time=max_shutter_time)

    def close_shutter(self, source='unknown'):
        stop_mjd = self.get('frames')[self.last_frame]['stop']
        stop_time = desietc.util.mjd_to_date(stop_mjd, utc_offset=0)
        return self.etc.stop_etc(source=source, stop_time=stop_time)

    def stop(self, source='unknown'):
        stop_mjd = self.get('frames')[self.last_frame]['stop']
        stop_time = desietc.util.mjd_to_date(stop_mjd, utc_offset=0)
        return self.etc.stop(source=source, stop_time=stop_time)

    def call_for_acq_image(self):
        if self.get('acq_path') is not None:
            assert self.get('frames')[0]['typ'] == 'gfa'
            return dict(
                image=desietc.offline.acq_to_online(self.get('acq_path'), desietc.gfa.GFACamera.guide_names),
                fiberassign=self.get('fassign_path'))
        else:
            return None

    def call_for_pm_info(self):
        if self.get('pm_info') is not None:
            return dict(guidestars=self.get('pm_info'))
        else:
            return None

    def call_for_frame(self, ftype):
        if self.get('frames') is not None and (self.next_frame > self.last_frame) and (self.get('frames')[self.next_frame]['typ'] == ftype):
            path = self.get(f'{ftype}_path')
            names = desietc.sky.SkyCamera.sky_names if ftype == 'sky' else desietc.gfa.GFACamera.guide_names
            fnum = self.get('frames')[self.next_frame]['num']
            self.last_frame += 1
            time.sleep(0.5)
            set_mjd_now(self.get('frames')[self.last_frame]['stop'])
            return dict(image=desietc.offline.fits_to_online(path, names, fnum))
        else:
            return None


def main():

    app = OfflineETCApp()
    nframes = 0
    print('OfflineETCApp is running.')
    #options = dict(req_efftime=1000, sbprof='ELG', max_exposure_time=2000, cosmics_split_time=1, maxsplit=4, warning_time=60)
    options = dict(req_efftime=1000, sbprof='ELG', max_exposure_time=2000, cosmics_split_time=1200, maxsplit=4, warning_time=60)
    #options = dict(req_efftime=300, sbprof='ELG', max_exposure_time=2000, cosmics_split_time=100, maxsplit=4, warning_time=60)
    while True:
        print('Enter a command: s(tart) f(rame) o(pen) c(lose) (s)t(op) q(uit) ?(status)')
        cmdline = input('# ')
        argv = cmdline.split(' ')
        argc = len(argv)
        cmd = argv[0]
        if cmd == 'q':
            print('Shutting down...')
            app.shutdown_event.set()
            while app.etc.etc_thread.is_alive():
                print('Waiting for ETC shutdown...')
                time.sleep(1)
            return 0
        elif cmd == 's':
            path = pathlib.Path('/Users/david/Data/DESI/20210420/')
            expid = 85640
            assets = desietc.offline.fetch_exposure(path, expid, only_complete=True)
            nframes = len(assets['frames'])
            print(app.start_exposure(assets, **options))
        elif cmd == 'o':
            print(app.open_shutter())
        elif cmd == 'c':
            print(app.close_shutter())
        elif cmd == 'h':
            expid = int(argv[1]) if argc > 1 else None
            print(app.etc.get_exposure_summary(expid))
        elif cmd == 't':
            print(app.stop())
        elif cmd == 'f':
            # Look for optional number of frames to release (default is 1).
            nf = int(argv[1]) if argc > 1 else 1
            app.next_frame = min(nframes - 1, app.next_frame + nf)
        elif cmd == '?':
            app.call_to_update_status()
        else:
            if cmd:
                print(f'Ignoring unrecognized input "{cmd}".')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y%m%d %H:%M:%S')
    logging.info('Program start')

    os.environ['ETC_GFA_CALIB'] = '/Users/david/Data/DESI/ETC/GFA_calib.fits'
    os.environ['ETC_SKY_CALIB'] = '/Users/david/Data/DESI/ETC/SKY_calib.fits'

    try:
        retval = main()
        sys.exit(retval)
    except Exception as e:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
