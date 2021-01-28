"""The top-level ETC algorithm.
"""
import pathlib

try:
    import DOSlib.logger as logging
except ImportError:
    # Fallback when we are not running as a DOS application.
    import logging

import numpy as np

import fitsio

import desietc.sky
import desietc.gfa
import desietc.gmm


class ETC(object):

    def __init__(self, sky_calib, gfa_calib, psf_pixels=25):
        """Initialize once per session.
        """
        # Initialize SKY and GFA camera processors.
        self.SKY = desietc.sky.SkyCamera(calib_name=sky_calib)
        self.GFA = desietc.gfa.GFACamera(calib_name=gfa_calib)
        # Initialize PSF fitting.
        psf_grid = np.arange(psf_pixels + 1) - psf_pixels / 2
        self.GMM = desietc.gmm.GMMFit(psf_grid, psf_grid)

    def process_acquisition(self, data):
        """Process the initial GFA acuquisition images.
        """
        hdr = data['header']
        self.night = hdr['NIGHT']
        self.expid = hdr['EXPID']
        logging.info(f'Processing acquisition image for {self.night}/{self.expid}.')
        # Loop over cameras.
        self.acquisition_results = {}
        for camera in self.GFA.guide_names:
            if camera not in data:
                logging.warn(f'No acquisition image for {camera}.')
                continue
            if not self.preprocess_gfa(camera, data[camera]):
                continue

    def preprocess_gfa(self, camera, data, default_ccdtemp=10):
        hdr = data['header']
        mjd_obs = hdr.get('MJD-OBS', None)
        if mjd_obs is None or mjd_obs < 58484: # 1-1-2019
            logging.error(f'Invalid {camera} MJD_OBS: {mjd_obs}.')
            return False
        exptime = hdr.get('EXPTIME', None)
        if exptime is None or exptime <= 0:
            logging.error(f'Invalid {camera} EXPTIME: {exptime}.')
            return False
        ccdtemp = hdr.get('GCCDTEMP', None)
        if ccdtemp is None:
            ccdtemp = default_ccdtemp
            logging.warning(f'Using default {camera} GCCDTEMP: {ccdtemp}C.')
        try:
            self.GFA.setraw(data['data'], name=camera)
        except ValueError as e:
            logging.error(f'Failed to process {camera} raw data: {e}')
            return False
        self.GFA.data -= self.GFA.get_dark_current(ccdtemp, exptime)
