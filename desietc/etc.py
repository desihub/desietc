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
import desietc.util

# TODO:
#  - apply exptime to guide_star nelec
#  - implement setter/getter for current FWHM,FFRAC0,FFRAC,TRANSP


class ETC(object):

    def __init__(self, sky_calib, gfa_calib, psf_pixels=25, max_dither=7, num_dither=1200):
        """Initialize once per session.

        Parameters
        ----------
        sky_calib : str
            Path to the SKY camera calibration file to use.
        gfa_calib : str
            Path to the GFA camera calibration file to use.
        psf_pixels : int
            Size of postage stamp to use for PSF measurements. Must be odd.
        max_dither : float
            Maximum dither to use in pixels relative to PSF centroid.
        num_dither : int
            Number of (x,y) dithers to use, extending out to max_dither with
            decreasing density.
        """
        # Initialize SKY and GFA camera processors.
        self.SKY = desietc.sky.SkyCamera(calib_name=sky_calib)
        self.GFA = desietc.gfa.GFACamera(calib_name=gfa_calib)
        # Initialize PSF fitting.
        if psf_pixels % 2 == 0:
            raise ValueError('psf_pixels must be odd.')
        psf_grid = np.arange(psf_pixels + 1) - psf_pixels / 2
        self.GMM = desietc.gmm.GMMFit(psf_grid, psf_grid)
        self.psf_pixels = psf_pixels
        self.xdither, self.ydither = desietc.util.diskgrid(num_dither, max_dither, alpha=2)

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
            camera_result = {}
            if camera not in data:
                logging.warn(f'No acquisition image for {camera}.')
                continue
            if not self.preprocess_gfa(camera, data[camera]):
                continue
            # Find PSF-like objects
            self.GFA.get_psfs()
            T, WT =  self.GFA.psf_stack
            if T is None:
                logging.error(f'Unable to find PSFs in {camera} acquisition image.')
                continue
            # Use a smaller stamp for fitting the PSF.
            nT = len(T)
            assert T.shape == WT.shape == (nT, nT)
            assert self.psf_pixels <= nT
            ntrim = (nT - self.psf_pixels) // 2
            S = slice(ntrim, ntrim + self.psf_pixels)
            T, WT = T[S, S], WT[S, S]
            # Save the PSF images.
            camera_result['data'] = T
            camera_result['ivar'] = WT
            self.acquisition_results[camera] = camera_result
            # Fit the PSF to a Gaussian mixture model.
            gmm_params = self.GMM.fit(T, WT, maxgauss=3)
            if gmm_params is None:
                logging.error(f'Unable to fit the PSF in {camera} acquisition image.')
                continue
            camera_result['gmm'] = gmm_params
            camera_result['model'] = self.GMM.predict(gmm_params)
            # Precompute dithered renderings of the model for fast guide frame fits.
            dithered = self.GMM.dither(gmm_params, self.xdither, self.ydither)
            camera_result['dithered'] = dithered
        # Update the current FWHM, FFRAC0 now.
        # ...

    def set_guide_stars(self, gfa_loc, col, row, mag, zeropoint=27.06):
        """
        """
        self.guide_stars = {}
        for camera in self.GFA.guide_names:
            sel = (gfa_loc == camera) & (mag > 0)
            if not np.any(sel):
                logging.warning(f'No guide stars available for {camera}.')
                continue
            self.guide_stars[camera] = dict(
                # Convert from PlateMaker indexing convention to (0,0) centered in bottom-left pixel.
                col=np.array(col[sel]) - 0.5,
                row=np.array(row[sel]) - 0.5,
                mag=np.array(mag[sel]),
                # Convert flux to predicted detected electrons per second in the
                # GFA filter with nominal zenith atmospheric transmission.
                nelec = 10 ** (-(mag[sel] - zeropoint) / 2.5)
            )
        # Update the current FFRAC,TRANSP now.
        # ...

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
        return True
