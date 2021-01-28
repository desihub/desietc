"""The top-level ETC algorithm.
"""
import pathlib

import numpy as np

import desietc.sky
import desietc.gfa
import desietc.gmm


class ETC(object):

    def __init__(self, sky_calib, gfa_calib, psf_pixels=25):
        """
        """
        # Initialize SKY and GFA camera processors.
        self.SKY = desietc.sky.SkyCamera(calib_name=sky_calib)
        self.GFA = desietc.gfa.GFACamera(calib_name=gfa_calib)
        # Initialize PSF fitting.
        psf_grid = np.arange(psf_pixels + 1) - psf_pixels / 2
        self.GMM = desietc.gmm.GMMFit(psf_grid, psf_grid)
