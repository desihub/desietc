"""The top-level ETC algorithm.
"""
import time
import multiprocessing

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
# - calculate transp, ffrac from acquisition image
# - update teff after each GFA/SKY update
# - estimate GFA thru errors?
# - implement save_exposure()
# - implement cutoff time logic
# - implement cosmic split logic
# - using multiprocessing for acquisition image analysis
# - flag issues detected in acq img

class ETC(object):

    SECS_PER_DAY = 86400

    def __init__(self, sky_calib, gfa_calib, psf_pixels=25, max_dither=7, num_dither=1200,
                 Ebv_coef=1.0, parallel=True):
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
        Ebv_coef : float
            Coefficient to use for converting Ebv into an equivalent MW
            transparency via 10 ** (-coef * Ebv / 2.5)
        parallel : bool
            Process GFA images in parallel when True.
        """
        self.Ebv_coef = Ebv_coef
        # Initialize PSF fitting.
        if psf_pixels % 2 == 0:
            raise ValueError('psf_pixels must be odd.')
        psf_grid = np.arange(psf_pixels + 1) - psf_pixels / 2
        self.GMM = desietc.gmm.GMMFit(psf_grid, psf_grid)
        self.psf_pixels = psf_pixels
        self.xdither, self.ydither = desietc.util.diskgrid(num_dither, max_dither, alpha=2)
        # Initialize analysis results.
        self.night = None
        self.expid = None
        self.num_guide_frames = 0
        self.num_sky_frames = 0
        self.acquisition_results = None
        self.guide_stars = None
        # Initialize buffers to record our signal- and sky-rate measurements.
        self.thru_measurements = desietc.util.MeasurementBuffer(maxlen=1000, default_value=1)
        self.sky_measurements = desietc.util.MeasurementBuffer(maxlen=200, default_value=1)
        # Initialize parallel processing if requested.
        if parallel:
            context = multiprocessing.get_context(method='spawn')
            npool = len(desietc.gfa.GFACamera.guide_names)
            self.pool = context.Pool(processes=npool)
            logging.info(f'Initialized pool of {npool} parallel GFA processors.')
        # Initialize SKY and GFA camera processors.
        self.SKY = desietc.sky.SkyCamera(calib_name=sky_calib)
        self.GFAs = {
            camera: desietc.gfa.GFACamera(calib_name=gfa_calib, default_name=camera)
            for camera in desietc.gfa.GFACamera.guide_names}

    def process_top_header(self, header, source, update_ok=False):
        """Process the top-level header of an exposure.
        The expected keywords are: NIGHT, EXPID.
        Return True if it is possible to keep going, i.e. unless we get a
        new value of NIGHT or EXPID and update_ok is False.
        """
        if 'NIGHT' not in header:
            logging.error(f'Missing NIGHT keyword in {source}')
        else:
            night = header['NIGHT']
            if night != self.night:
                if update_ok:
                    logging.info(f'Setting NIGHT {night} from {source}.')
                    self.night = night
                else:
                    logging.error(f'Got NIGHT {night} from {source} but expected {self.night}.')
                    return False
        if 'EXPID' not in header:
            logging.error(f'Missing EXPID keyword in {source}')
        else:
            expid = header['EXPID']
            if expid != self.expid:
                if update_ok:
                    logging.info(f'Setting EXPID {expid} from {source}.')
                    self.expid = expid
                    self.exptag = str(self.expid).zfill(8)
                else:
                    logging.error(f'Got EXPID {expid} from {source} but expected {self.expid}.')
                    return False
        return True

    def process_camera_header(self, header, source):
        """Check the header for a single camera.
        The expected keywords are: MJD-OBS, EXPTIME.
        Return True if it is possible to keep going, i.e. MJD-OBS and EXPTIME are both
        present with reasonable values.
        """
        if 'MJD-OBS' not in header:
            logging.error(f'Missing MJD-OBS keyword in {source}.')
            return False
        mjd_obs = header['MJD-OBS']
        if mjd_obs is None or mjd_obs < 58484: # 1-1-2019
            logging.error(f'Invalid MJD-OBS {mjd_obs} from {source}.')
            return False
        if 'EXPTIME' not in header:
            logging.error(f'Missing EXPTIME keyword in {source}.')
        exptime = header['EXPTIME']
        if exptime is None or exptime <= 0:
            logging.error(f'Invalid EXPTIME {exptime} from {source}.')
            return False
        self.mjd_obs = mjd_obs
        self.exptime = exptime
        return True

    def process_acquisition(self, data):
        """Process the initial GFA acquisition images.
        """
        ncamera = 0
        start = time.time()
        if not self.process_top_header(data['header'], 'acquisition image', update_ok=True):
            return False
        logging.info(f'Processing acquisition image for {self.night}/{self.exptag}.')
        # Loop over cameras.
        self.acquisition_results = {}
        for camera in desietc.gfa.GFACamera.guide_names:
            camera_result = {}
            if camera not in data:
                logging.warn(f'No acquisition image for {camera}.')
                continue
            if not self.process_camera_header(data[camera]['header'], f'{camera} acquisition image'):
                continue
            if not self.preprocess_gfa(camera, data[camera], f'{camera} acquisition image'):
                continue
            thisGFA = self.GFAs[camera]
            # Find PSF-like objects
            thisGFA.get_psfs()
            T, WT =  thisGFA.psf_stack
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
            ncamera += 1
        # Update the current FWHM, FFRAC0 now.
        # ...
        # Reset the guide frame counter and guide star data.
        self.num_guide_frames = 0
        self.guide_stars = None
        # Report timing.
        elapsed = time.time() - start
        logging.info(f'Acquisition processing took {elapsed:.2f}s for {ncamera} cameras.')
        return True

    def set_guide_stars(self, gfa_loc, col, row, mag,
                        zeropoint=27.06, fiber_diam_um=107, pixel_size_um=15):
        """Specify the guide star locations and magnitudes to use when analyzing
        each guide frame.  These are normally calculated by PlateMaker.
        """
        if self.acquisition_results is None:
            logging.error(f'Received guide stars with no acquisition image.')
            return False
        if self.guide_stars is not None:
            logging.warning(f'Overwriting previous guide stars for {self.night}/{self.exptag}.')
        max_rsq = (0.5 * fiber_diam_um / pixel_size_um) ** 2
        profile = lambda x, y: 1.0 * (x ** 2 + y ** 2 < max_rsq)
        halfsize = self.psf_pixels // 2
        self.guide_stars = {}
        nstars = []
        # The xy swap here is intentional
        ny, nx = 2 * desietc.gfa.GFACamera.nampx, 2 * desietc.gfa.GFACamera.nampy
        for camera in desietc.gfa.GFACamera.guide_names:
            sel = (gfa_loc == camera) & (mag > 0)
            if not np.any(sel):
                logging.warning(f'No guide stars available for {camera}.')
                nstars.append(0)
                continue
            # Loop over guide stars for this GFA.
            stars = []
            for i in np.where(sel)[0]:
                # Convert from PlateMaker indexing convention to (0,0) centered in bottom-left pixel.
                x0 = col[i] - 0.5
                y0 = row[i] - 0.5
                rmag = mag[i]
                # Convert flux to predicted detected electrons per second in the
                # GFA filter with nominal zenith atmospheric transmission.
                nelec_rate = 10 ** (-(mag[i] - zeropoint) / 2.5)
                # Prepare slices to extract this star in each guide frame.
                iy, ix = np.round(y0).astype(int), np.round(x0).astype(int)
                ylo, yhi = iy - halfsize, iy + halfsize + 1
                xlo, xhi = ix - halfsize, ix + halfsize + 1
                if ylo < 0 or yhi > ny or xlo < 0 or xhi > nx:
                    logging.info(f'Skipping stamp too close to border at ({x0},{y0})')
                    continue
                yslice, xslice = slice(ylo, yhi), slice(xlo, xhi)
                # Calculate an antialiased fiber template for FFRAC calculations.
                fiber_dx, fiber_dy = x0 - ix, y0 - iy
                fiber = desietc.util.make_template(
                    self.psf_pixels, profile, dx=fiber_dx, dy=fiber_dy, normalized=False)
                stars.append(dict(
                    x0=x0, y0=y0, rmag=rmag, nelec_rate=nelec_rate,
                    fiber_dx=fiber_dx, fiber_dy=fiber_dy,
                    yslice=yslice, xslice=xslice, fiber=fiber))
            nstars.append(len(stars))
            if len(stars):
                self.guide_stars[camera] = stars
        if len(self.guide_stars) == 0:
            logging.error(f'No usable guide stars for {self.night}/{self.exptag}.')
            return False
        nstars_msg = '+'.join([str(n) for n in nstars]) + '=' + str(np.sum(nstars))
        logging.info(f'Using {nstars_msg} guide stars for {self.night}/{self.exptag}.')
        # Update the current FFRAC,TRANSP now.
        # ...
        return True

    def process_guide_frame(self, data):
        """Process a guide frame.
        """
        ncamera = nstar = 0
        start = time.time()
        fnum = self.num_guide_frames
        self.num_guide_frames += 1
        if not self.process_top_header(data['header'], f'guide[{fnum}]'):
            return False
        logging.info(f'Processing guide frame {fnum} for {self.night}/{self.exptag}.')
        if self.acquisition_results is None:
            logging.error('Ignoring guide frame before acquisition image.')
            return False
        if self.guide_stars is None:
            logging.error('Ignoring guide frame before guide stars.')
            return False
        # Loop over cameras with acquisition results.
        mjd_obs, exptime, camera_transp, camera_ffrac = [], [], [], []
        for camera, acquisition in self.acquisition_results.items():
            if camera not in self.guide_stars:
                loggining.info(f'Skipping {camera} guide frame {fnum} with no guide stars.')
                continue
            if camera not in data:
                logging.warning(f'Missing {camera} guide frame {fnum}.')
                continue
            if not self.process_camera_header(data[camera]['header'], f'{camera}[{fnum}]'):
                continue
            if not self.preprocess_gfa(camera, data[camera], f'{camera}[{fnum}]'):
                continue
            thisGFA = self.GFAs[camera]
            # Lookup this camera's PSF model.
            psf = self.acquisition_results[camera]
            # Loop over guide stars for this camera.
            star_ffrac, star_transp = [], []
            for istar, star in enumerate(self.guide_stars[camera]):
                # Extract the postage stamp for this star.
                D = thisGFA.data[0, star['xslice'], star['yslice']]
                DW = thisGFA.ivar[0, star['xslice'], star['yslice']]
                # Estimate the actual centroid in pixels, flux in electrons and
                # constant background level in electrons / pixel.
                dx, dy, flux, bg, nll, best_fit = self.GMM.fit_dithered(
                    self.xdither, self.ydither, psf['dithered'], D, DW)
                # Calculate centroid offset relative to the target fiber center.
                dx -= star['fiber_dx']
                dy -= star['fiber_dy']
                # Calculate the corresponding fiber fraction for this star.
                ffrac = np.sum(star['fiber'] * best_fit)
                # Calculate the transparency as the ratio of measured / predicted electrons.
                transp = flux / (star['nelec_rate'] * self.exptime)
                logging.debug(f'{camera}[{fnum},{istar}] dx={dx:.1f} dy={dy:.1f} ffrac={ffrac:.3f} transp={transp:.3f}')
                star_transp.append(transp)
                star_ffrac.append(ffrac)
                nstar += 1
            camera_transp.append(np.nanmedian(star_transp))
            camera_ffrac.append(np.nanmedian(star_ffrac))
            mjd_obs.append(self.mjd_obs)
            exptime.append(self.exptime)
            ncamera += 1
        # Combine all cameras.
        transp = np.nanmedian(camera_transp)
        ffrac = np.nanmedian(camera_ffrac)
        thru = transp * ffrac
        logging.info(f'Guide transp={transp:.3f}, ffrac={ffrac:.3f}, thru={thru:.3f}.')
        # Record this measurement.
        mjd_start, mjd_stop = self.get_mjd_range(mjd_obs, exptime, f'guide[{fnum}]')
        # Use constant error until we have a proper estimate.
        self.thru_measurements.add(mjd_start, mjd_stop, thru, 0.01)
        # Report timing.
        elapsed = time.time() - start
        logging.debug(f'Guide frame processing took {elapsed:.2f}s for {nstar} stars in {ncamera} cameras.')
        return True

    def process_sky(self, data):
        """Process a SKY frame.
        """
        ncamera = 0
        start = time.time()
        fnum = self.num_sky_frames
        self.num_sky_frames += 1
        if not self.process_top_header(data['header'], f'sky[{fnum}]', update_ok=True):
            return False
        logging.info(f'Processing sky frame {fnum} for {self.night}/{self.exptag}.')
        flux, ivar = 0, 0
        mjd_obs, exptime = [], []
        for camera in self.SKY.sky_names:
            if camera not in data:
                logging.warn(f'No {camera} image for frame {fnum}.')
                continue
            if not self.process_camera_header(data[camera]['header'], f'{camera}[{fnum}]'):
                continue
            camera_flux, camera_dflux = self.SKY.setraw(data[camera]['data'], name=camera)
            camera_flux /= self.exptime
            camera_dflux /= self.exptime
            logging.debug(f'{camera}[{fnum}] flux = {camera_flux:.2f} +/- {camera_dflux:.2f}')
            camera_ivar = 1 / camera_dflux ** 2
            flux += camera_ivar * camera_flux
            ivar += camera_ivar
            mjd_obs.append(self.mjd_obs)
            exptime.append(self.exptime)
            ncamera += 1
        # Calculate the weighted average sky flux over all cameras.
        flux /= ivar
        dflux = ivar ** -0.5
        logging.info(f'SKY flux = {flux:.2f} +/- {dflux:.2f}.')
        # Record this measurement.
        mjd_start, mjd_stop = self.get_mjd_range(mjd_obs, exptime, f'sky[{fnum}]')
        self.sky_measurements.add(mjd_start, mjd_stop, flux, dflux)
        # Profile timing.
        elapsed = time.time() - start
        logging.debug(f'Sky frame processing took {elapsed:.2f}s for {ncamera} cameras.')
        return True

    def get_mjd_range(self, mjd_obs, exptime, source, max_jitter=5):
        mjd_obs = np.asarray(mjd_obs)
        mjd_obs_all = np.nanmean(mjd_obs)
        if np.any(np.abs(mjd_obs - mjd_obs_all) * self.SECS_PER_DAY > max_jitter):
            logging.warn(f'MJD_OBS jitter exceeds {max_jitter}s for {source}.')
            mjd_obs_all = np.nanmedian(mjd_obs)
        exptime = np.asarray(exptime)
        exptime_all = np.nanmean(exptime)
        if np.any(np.abs(exptime - exptime_all) * self.SECS_PER_DAY > max_jitter):
            logging.warn(f'EXPTIME jitter exceeds {max_jitter}s for {source}.')
            exptime_all = np.nanmedian(exptime)
        return (mjd_obs_all, mjd_obs_all + exptime_all / self.SECS_PER_DAY)

    def preprocess_gfa(self, camera, data, source, default_ccdtemp=10):
        """Preprocess raw data for the specified GFA.
        Returns False with a log message in case of any problems.
        Otherwise, GFA.data and GFA.ivar are corrected for bias,
        dark-current and and bad pixels.
        """
        hdr = data['header']
        ccdtemp = hdr.get('GCCDTEMP', None)
        if ccdtemp is None:
            ccdtemp = default_ccdtemp
            logging.warning(f'Using default GCCDTEMP {ccdtemp}C for {source}')
        try:
            thisGFA = self.GFAs[camera]
            thisGFA.setraw(data['data'])
        except ValueError as e:
            logging.error(f'Failed to process {source} raw data: {e}')
            return False
        thisGFA.data -= thisGFA.get_dark_current(ccdtemp, self.exptime)
        return True

    def get_accumulated_teff(self, mjd_start, mjd_stop, MW_transparency=1):
        """
        """
        if mjd_stop <= mjd_start:
            logging.warn('get_accumulated_teff called with mjd_stop <= mjd_start.')
            return 0
        # Calculate the average signal throughput.
        _, thru_grid = self.thru_measurements.sample(mjd_start, mjd_stop)
        thru = np.mean(thru_grid)
        # Calculate the integrated sky background.
        _, sky_grid = self.sky_measurements.sample(mjd_start, mjd_stop)
        sky = np.mean(sky_grid)
        # Calculate the accumulated effective exposure time in seconds.
        treal = (mjd_stop - mjd_start) * self.SECS_PER_DAY
        teff = treal / sky * (MW_transparency * thru) ** 2
        logging.info(f'Calculated treal={treal:.1f}s, teff={teff:.1f}s using ' +
            f'sky={sky:.3f}, thru={thru:.3f}, MW={MW_transparency:.3f}.')
        return teff

    def start_exposure(self, night, expid, mjd, Ebv, teff, cutoff, cosmic):
        """
        """
        logging.info(f'Starting {night}/{expid} at {mjd} with teff={teff:.0f}s, cutoff={cutoff:.0f}s, ' +
            f'cosmic={cosmic:.0f}s, Ebv={Ebv:.2f}')
        self.mjd_start = mjd
        self.teff_target = teff
        self.cutoff = cutoff
        self.cosmic = cosmic
        self.Ebv = Ebv
        self.MW_transparency = 10 ** (-self.Ebv_coef * self.Ebv / 2.5)

    def save_exposure(self, path):
        """
        """
        logging.info(f'Saving ETC outputs for {self.night}/{self.exptag} to {path}')
        if not path.exists():
            logging.error(f'Non-existent path: {path}.')
            return
