"""Analyze images captured by the DESI Sky Cameras for the online ETC.
"""
import collections

try:
    import DOSlib.logger as logging
except ImportError:
    # Fallback when we are not running as a DOS application.
    import logging

import numpy as np

import scipy.optimize
import scipy.stats

import fitsio

import desietc.util


class BGFitter(object):
    """Fit a histogram of pixel values to a single Gaussian noise model.
    """
    def __init__(self, optimize_args={}):
        # Initialize the args sent to scipy.optimize.minimize()
        self.kwargs = dict(
            method='Nelder-Mead',
            options=dict(maxiter=10000, xatol=1e-4, fatol=1e-4, disp=False))
        self.kwargs.update(optimize_args)

    def predict(self, ntot, mu, std):
        """Predict data with specified parameters and bin edges.
        """
        z = scipy.special.erf((self.xedge - mu) / (np.sqrt(2) * std))
        return 0.5 * ntot * np.diff(z)

    def nll(self, theta):
        """Objective function for minimization, calculates -logL.
        """
        ypred = self.predict(*theta)
        # Use the model to predict the Gaussian inverse variances.
        yvar = np.maximum(1, ypred)
        return 0.5 * np.sum((self.ydata - ypred) ** 2 / yvar)

    def fit(self, data, nbins=30, maxpct=90):
        data = data.reshape(-1)
        clipped, lo, hi = scipy.stats.sigmaclip(data, low=3, high=2)
        if np.issubdtype(data.dtype, np.integer):
            # Align bins to integer boundaries.
            lo, hi = np.floor(lo) - 0.5, np.ceil(hi) + 0.5
            binsize = np.round(np.ceil(hi - lo) / nbins)
            nbins = np.ceil((hi - lo) / binsize)
            self.xedge = lo + np.arange(nbins) * binsize
        else:
            self.xedge = np.linspace(lo, hi, nbins)
        self.ydata, _ = np.histogram(data, bins=self.xedge)
        xc = 0.5 * (self.xedge[1:] + self.xedge[:-1])
        theta0 = np.array([1, np.mean(clipped), np.std(clipped)])
        y0 = self.predict(*theta0)
        imode = np.argmax(y0)
        theta0[0] = self.ydata[imode] / y0[imode]
        y0 = self.predict(*theta0)
        result = scipy.optimize.minimize(self.nll, theta0, **self.kwargs)
        if result.success:
            theta = result.x
        else:
            theta = theta0
        return theta


def load_calib_data(name='SKY_calib.fits'):
    names = collections.defaultdict(list)
    slices = collections.defaultdict(list)
    calibs = collections.defaultdict(list)
    cameras = []
    with fitsio.FITS(str(name)) as hdus:
        meta = hdus[0].read_header()
        nspot = meta['NSPOT']
        spots = hdus['SPOT'].read()
        stampsize = spots.shape[1]
        lo = stampsize // 2
        hi = stampsize - lo
        for k  in range(nspot):
            camera = meta['CAMERA_{0}'.format(k)]
            cameras.append(camera)
            names[camera].append(meta['NAME_{0}'.format(k)])
            y, x = meta['ROW_{0}'.format(k)], meta['COL_{0}'.format(k)]
            slices[camera].append((slice(y - lo, y + hi), slice(x - lo, x + hi)))
            calibs[camera].append(meta['CALIB_{0}'.format(k)])
        mask_data = hdus['MASK'].read()
        spot_data = hdus['SPOT'].read()
    masks, spots, calib = {}, {}, {}
    cameras = np.array(cameras)
    for camera in np.unique(cameras):
        sel = cameras == camera
        masks[camera] = mask_data[sel]
        spots[camera] = spot_data[sel]
    logging.info('Loaded SKY calib data from {0}'.format(name))
    return names, slices, masks, spots, calibs


# Fine tuning coefficients derived from fitting the trends of individual fibers vs the spectroscopic
# r-band sky using data from Mar-Apr 2021.  The fit is linear in log space, so the adjustment model is:
#  x --> coef * x ** pow
finetune_coef = np.array([
    [0.62397108, 0.        , 0.7552137 , 0.93242162, 0.95308126,
     0.97305782, 0.87329783, 1.09370934, 1.12080438, 0.        ],
    [0.        , 0.70029763, 0.91706983, 0.85841961, 0.91394301,
     1.00810978, 0.91657536, 0.        , 0.        , 0.        ]])

finetune_pow = np.array([
    [1.08673508, 1.        , 1.06715913, 1.01187676, 1.0178157 ,
     1.02464004, 1.02520429, 1.01357766, 1.00232129, 1.        ],
    [1.        , 1.08950204, 1.00924252, 1.01807379, 1.01857049,
     1.04251868, 1.02630379, 1.        , 1.        , 1.        ]])

def finetune_adjust(x, icamera, ifiber):
    return finetune_coef[icamera, ifiber] * x ** finetune_pow[icamera, ifiber]

## Mask of fibers to use for each camera.
fiber_mask = np.array([
    [1,0,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,0,0,0]])


class SkyCamera(object):
    """
    """
    sky_names = 'SKYCAM0', 'SKYCAM1'

    def __init__(self, calib_name='SKY_calib.fits'):
        self.names, self.slices, self.masks, self.spots, self.calibs = load_calib_data(calib_name)
        maxstamps = max([len(N) for N in self.names.values()])
        stampsize = self.masks[self.sky_names[0]].shape[1]
        # Initialize reusable arrays of analysis results.
        self.bgfittter = BGFitter()
        self.bgmean = np.zeros(maxstamps)
        self.bgsigma = np.zeros(maxstamps)
        self.flux = np.zeros(maxstamps)
        self.fluxerr = np.zeros(maxstamps)
        self.bgfit = np.zeros(maxstamps)
        self.bgerr = np.zeros(maxstamps)
        # Allocate memory to hold each stamp and its ivar.
        self.data = np.empty((maxstamps, stampsize, stampsize), np.float32)
        self.ivar = np.empty((maxstamps, stampsize, stampsize), np.float32)
        self.valid = np.empty((maxstamps, stampsize, stampsize), np.float32)
        self.model = np.empty((maxstamps, stampsize, stampsize), np.float32)
        self.pull = np.empty((maxstamps, stampsize, stampsize), np.float32)
        # Initialize background fitting.
        self.bgfitter = BGFitter()

    def setraw(self, raw, name, gain=2.5, saturation=65500, refit=True, pullcut=5, chisq_max=5, ndrop_max=3,
               masked=True, finetune=True):
        """
        """
        if name not in self.slices:
            raise ValueError('Invalid SKY name: {0}.'.format(name))
        icamera = self.sky_names.index(name)
        slices = self.slices[name]
        # Copy stamps for each spot into self.data.
        N = self.ndata = len(slices)
        for k, S in enumerate(slices):
            # Fit the background mean and sigma.
            norm, bgmean, bgsigma = self.bgfitter.fit(raw[S])
            self.bgmean[k], self.bgsigma[k] = bgmean, bgsigma
            # Subtract the mean background level.
            self.data[k] = raw[S] - bgmean
            # Estimate the signal contribution to the variance.
            signalvar = np.maximum(0, self.data[k]) / gain
            # Estimate the total inverse variance per pixel.
            self.ivar[k] = 1 / (bgsigma ** 2 + signalvar)
            # Mask any saturated or ~dead pixels.
            dead = self.data[k] < - 5 * bgsigma
            saturated = raw[S] > saturation
            self.ivar[k][dead | saturated] = 0
            # Mask known hot pixels.
            self.ivar[k][self.masks[name][k]] = 0
        # Fit for the spot flux and background level.
        self.flux[:N], self.bgfit[:N], cov = desietc.util.fit_spots(
            self.data[:N], self.ivar[:N], self.spots[name])
        self.fluxerr[:N] = np.sqrt(cov[:, 0, 0])
        self.bgerr[:N] = np.sqrt(cov[:, 1, 1])
        if refit:
            # Save the original masking due to dead / hot / saturated pixels.
            self.valid[:N] = 1. * (self.ivar[:N] > 0)
            # Use the initial fit for an improved estimate of the signal variance.
            signalvar = np.maximum(0, self.flux[:N].reshape(-1, 1, 1) * self.spots[name]) / gain
            #assert np.all(signalvar >= 0)
            self.ivar[:N] = 1 / (self.bgsigma[:N].reshape(-1, 1, 1) ** 2 + signalvar)
            # Mask any pixels with extreme pulls, calculated with the improved ivar.
            model = (self.bgfit[:N].reshape(-1, 1, 1) +
                     self.flux[:N].reshape(-1, 1, 1) * self.spots[name])
            pull = (self.data[:N] - model[:N]) * np.sqrt(self.ivar[:N])
            self.valid[:N] *= 1. * (np.abs(pull) < pullcut)
            # Apply the original + new masking.
            self.ivar[:N] *= self.valid[:N]
            # Refit
            self.flux[:N], self.bgfit[:N], cov = desietc.util.fit_spots(
                self.data[:N], self.ivar[:N], self.spots[name])
            #assert np.all(cov[:, 0, 0] > 0)
            self.fluxerr[:N] = np.sqrt(cov[:, 0, 0])
            #assert np.all(cov[:, 1, 1] > 0)
            self.bgerr[:N] = np.sqrt(cov[:, 1, 1])
        # Give up if we have invalid fluxes or errors.
        if not np.all((self.fluxerr[:N] > 0) & np.isfinite(self.flux[:N])):
            return None, None
        # Calculate the best-fit model for each fiber.
        self.model[:N] = (self.bgfit[:N].reshape(-1, 1, 1) +
                                   self.flux[:N].reshape(-1, 1, 1) * self.spots[name])
        # Calculate the corresponding pull images.
        self.pull[:N] = (self.data[:N] - self.model[:N]) * np.sqrt(self.ivar[:N])
        # Apply per-fiber calibrations.
        calib = self.calibs[name]
        cflux = self.flux[:N] / calib
        cfluxerr = self.fluxerr[:N] / calib
        if finetune:
            coef = finetune_coef[icamera, :N]
            pow = finetune_pow[icamera, :N]
            cflux = coef * np.maximum(0, cflux) ** pow
            cfluxerr = coef * np.maximum(0, cfluxerr) ** pow
        # Which fibers should be used?
        if masked:
            keep = (fiber_mask[icamera, :N] > 0) & (cfluxerr > 0)
            cflux = cflux[keep]
            cfluxerr = cfluxerr[keep]
            N = len(cflux)
        # Calculate the weighted mean over fibers (marginalized over bg levels) and its error.
        wgt = cfluxerr ** -2
        used = np.ones(N, bool)
        snr = self.flux[:N] / self.fluxerr[:N]
        order = np.argsort(snr)[::-1]
        self.ndrop = 0
        while self.ndrop <= ndrop_max:
            ivar = np.sum(wgt[used])
            meanflux = np.sum(wgt[used] * cflux[used]) / ivar
            self.chisq = np.sum(((cflux[used] - meanflux) / cfluxerr[used]) ** 2) / (N - self.ndrop)
            if self.chisq < chisq_max:
                break
            # Drop the remaining fiber with the highest SNR.
            idrop = order[self.ndrop]
            used[idrop] = False
            self.ndrop += 1
        return meanflux, ivar ** -0.5
