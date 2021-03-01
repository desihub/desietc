import os
import sys
import traceback
import logging
import threading
import pathlib
import time
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import desietc.online
import desietc.offline
import desietc.util


class OfflineETCApp:

    def __init__(self):
        # Initialize the ETC.
        self.shutdown_event = threading.Event()
        self.shutdown_event.clear()
        self.etc = desietc.online.OnlineETC(self.shutdown_event)
        # Prepare to handle callouts.
        self.etc.call_to_request_stop = self.call_to_request_stop
        self.etc.call_to_request_split = self.call_to_request_split
        self.etc.call_to_update_status = self.call_to_update_status
        self.etc.call_for_acq_image = self.call_for_acq_image
        self.etc.call_for_pm_info = self.call_for_pm_info
        self.etc.call_when_image_ready = self.call_when_image_ready
        self.etc.call_for_sky_image = lambda wait: self.call_for_frame('sky', wait)
        self.etc.call_for_gfa_image = lambda wait: self.call_for_frame('gfa', wait)
        self.expdir = pathlib.Path('expdir')
        self.etc.call_for_exp_dir = lambda expid: str(self.expdir / f'{expid:08d}')
        self.status_updates = []
        self.assets = None
        # Configure the ETC.
        assert self.etc.configure()

    def call_to_update_status(self):
        status = self.etc.get_status()
        logging.info(f'get_status: {status}')
        self.status_updates.append(status)

    def call_when_image_ready(self, path, expid, frame=0):
        print(f'Image ready for {expid}[{frame}] at {path}')

    def call_to_request_stop(self, cause):
        logging.info(f'request_stop: cause={cause}')
        self.close_shutter()
        self.stop()

    def call_to_request_split(self, cause):
        logging.info(f'request_split: cause={cause}')
        self.close_shutter()
        self.expid += 1

    def get(self, key):
        return None if self.assets is None else self.assets.get(key, None)

    def start_exposure(self, assets, requested_teff, sbprofile, max_exposure_time, cosmics_split_time):
        self.assets = assets
        self.expid = self.get('expid')
        self.next_frame = 0
        self.last_frame = 0
        self.etc.prepare_for_exposure(self.expid, requested_teff, sbprofile, max_exposure_time, cosmics_split_time)
        (pathlib.Path('expdir') / f'{self.expid:08d}').mkdir(parents=True, exist_ok=True)
        return self.etc.start(start_time=self.get('start_time'))

    def open_shutter(self, splittable=True):
        start_mjd = self.get('frames')[self.last_frame]['when']
        start_time = desietc.util.mjd_to_date(start_mjd, utc_offset=0)
        return self.etc.start_etc(self.expid, start_time=start_time, splittable=splittable)

    def close_shutter(self, source='unknown'):
        stop_mjd = self.get('frames')[self.last_frame]['when']
        stop_time = desietc.util.mjd_to_date(stop_mjd, utc_offset=0)
        return self.etc.stop_etc(source=source, stop_time=stop_time)

    def stop(self, source='unknown'):
        return self.etc.stop(source=source)

    def call_for_acq_image(self, wait=None):
        if self.get('acq_path') is not None:
            assert self.get('frames')[0]['typ'] == 'gfa'
            return dict(
                image=desietc.offline.acq_to_online(self.get('acq_path'), desietc.gfa.GFACamera.guide_names),
                fiberassign=self.get('fassign_path'))
        else:
            if wait: time.sleep(wait)
            return None

    def call_for_pm_info(self, wait=None):
        if self.get('pm_info') is not None:
            return dict(guidestars=self.get('pm_info'))
        else:
            if wait: time.sleep(wait)
            return None

    def call_for_frame(self, ftype, wait=None):
        if self.get('frames') is not None and (self.next_frame > self.last_frame) and (self.get('frames')[self.next_frame]['typ'] == ftype):
            path = self.get(f'{ftype}_path')
            names = desietc.sky.SkyCamera.sky_names if ftype == 'sky' else desietc.gfa.GFACamera.guide_names
            fnum = self.get('frames')[self.next_frame]['num']
            self.last_frame += 1
            return dict(image=desietc.offline.fits_to_online(path, names, fnum))
        else:
            if wait: time.sleep(wait)
            return None


def main():

    app = OfflineETCApp()
    print('OfflineETCApp is running.')
    #options = dict(requested_teff=1000, sbprofile='PSF', max_exposure_time=2000, cosmics_split_time=1200)
    options = dict(requested_teff=1000, sbprofile='PSF', max_exposure_time=2000, cosmics_split_time=30)
    while True:
        print('Enter a command: s(tart) f(rame) o(pen) c(lose) (s)t(op) q(uit)')
        cmd = input('# ')
        if cmd == 'q':
            print('Shutting down...')
            app.shutdown_event.set()
            while app.etc.etc_thread.is_alive():
                print('Waiting for ETC shutdown...')
                time.sleep(1)
            return 0
        elif cmd == 's':
            path = pathlib.Path('/Users/david/Data/DESI/20201218/')
            expid = 68630
            assets = desietc.offline.fetch_exposure(path, expid, only_complete=True)
            nframes = len(assets['frames'])
            print(app.start_exposure(assets, **options))
        elif cmd == 'o':
            print(app.open_shutter())
        elif cmd == 'c':
            print(app.close_shutter())
        elif cmd == 't':
            print(app.stop())
        elif cmd == 'f':
            if app.next_frame < nframes - 1:
                app.next_frame += 1
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
