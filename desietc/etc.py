"""The top-level ETC algorithm.
"""
import time
import json
import pathlib
import multiprocessing
try:
    import multiprocessing.shared_memory
    shared_memory_available = True
except ImportError:
    # We will complain about this later if we actually need it.
    shared_memory_available = False

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
import desietc.plot

# TODO:
# - update teff after each GFA/SKY update
# - estimate GFA thru errors?
# - implement cutoff time logic
# - implement cosmic split logic
# - flag issues detected in acq img

class ETC(object):

    SECS_PER_DAY = 86400
    BUFFER_NAME = 'ETC_{0}_buffer'

    def __init__(self, sky_calib, gfa_calib, psf_pixels=25, max_dither=7, num_dither=1200,
                 Ebv_coef=1.0, nbad_threshold=100, nll_threshold=10, parallel=True):
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
        nbad_threshold : int
            Maximum number of allowed bad overscan pixel values before a GFA is
            flagged as noisy.
        nll_threshold : float
            Maximum allowed GMM fit NLL value before a PSF fit is flagged as
            potentially bad.
        parallel : bool
            Process GFA images in parallel when True.
        """
        self.Ebv_coef = Ebv_coef
        self.nbad_threshold = nbad_threshold
        self.nll_threshold = nll_threshold
        # Initialize PSF fitting.
        if psf_pixels % 2 == 0:
            raise ValueError('psf_pixels must be odd.')
        psf_grid = np.arange(psf_pixels + 1) - psf_pixels / 2
        self.GMM = desietc.gmm.GMMFit(psf_grid, psf_grid)
        self.xdither, self.ydither = desietc.util.diskgrid(num_dither, max_dither, alpha=2)
        self.psf_pixels = psf_pixels
        psf_stacksize = desietc.gfa.GFACamera.psf_stacksize
        if self.psf_pixels > psf_stacksize:
            raise ValueError(f'psf_pixels must be <= {psf_stacksize}.')
        # Use a smaller stamp for fitting the PSF.
        ntrim = (psf_stacksize - self.psf_pixels) // 2
        self.psf_inset = slice(ntrim, ntrim + self.psf_pixels)
        self.measure = desietc.util.PSFMeasure(psf_stacksize)
        # Initialize analysis results.
        self.night = None
        self.expid = None
        self.num_guide_frames = 0
        self.num_sky_frames = 0
        self.acquisition_data = None
        self.guide_stars = None
        self.image_path = None
        # How many GUIDE and SKY cameras do we expect?
        self.ngfa = len(desietc.gfa.GFACamera.guide_names)
        self.nsky = len(desietc.sky.SkyCamera.sky_names)
        # Initialize buffers to record our signal- and sky-rate measurements.
        # Define auxiliary data to save with each GFA or SKY measurement.
        self.thru_measurements = desietc.util.MeasurementBuffer(
            maxlen=1000, default_value=1, aux_dtype=[
                ('ffrac', np.float32, (self.ngfa,)),  # fiber fraction measured from single GFA
                ('transp', np.float32, (self.ngfa,)), # transparency measured from single GFA
                ('dx', np.float32, (self.ngfa,)),     # mean x shift of centroid from single GFA in pixels
                ('dy', np.float32, (self.ngfa,)),     # mean y shift of centroid from single GFA in pixels
            ])
        self.sky_measurements = desietc.util.MeasurementBuffer(
            maxlen=200, default_value=1, aux_dtype=[
                ('flux', np.float32, (self.nsky,)),  # sky flux meausured from a single SKYCAM.
                ('dflux', np.float32, (self.nsky,)), # sky flux uncertainty meausured from a single SKYCAM.
                ('ndrop', np.int32, (self.nsky,)),   # number of fibers dropped from the camera flux estimate.
            ])
        # Initialize the SKY camera processor.
        self.SKY = desietc.sky.SkyCamera(calib_name=sky_calib)
        # Initialize the GFA camera processor(s).
        self.GFAs = {}
        if parallel and not shared_memory_available:
            raise RuntimeError('Python >= 3.8 required for the parallel ETC option.')
        self.parallel = parallel
        if parallel:
            # Allocate shared-memory buffers for each guide camera's GFACamera object.
            bufsize = desietc.gfa.GFACamera.buffer_size
            self.shared_mem = {}
            self.pipes = {}
            self.processes = {}
            for camera in desietc.gfa.GFACamera.guide_names:
                bufname = self.BUFFER_NAME.format(camera)
                self.shared_mem[camera] = multiprocessing.shared_memory.SharedMemory(
                    name=bufname, size=bufsize, create=True)
                self.GFAs[camera] = desietc.gfa.GFACamera(
                    calib_name=gfa_calib, buffer=self.shared_mem[camera].buf)
            nbytes = bufsize * len(self.GFAs)
            logging.info(f'Allocated {nbytes/2**20:.1f}Mb of shared memory.')
            # Initialize per-GFA processes, each with its own pipe.
            context = multiprocessing.get_context(method='spawn')
            for camera in desietc.gfa.GFACamera.guide_names:
                self.pipes[camera], child = context.Pipe()
                self.processes[camera] = context.Process(
                    target=ETC.gfa_process, args=(
                        camera, gfa_calib, self.GMM, self.psf_inset, self.measure, child))
                self.processes[camera].start()
            logging.info(f'Initialized {len(self.GFAs)} GFA processes.')
        else:
            # All GFAs use the same GFACamera object.
            GFA = desietc.gfa.GFACamera(calib_name=gfa_calib)
            for camera in desietc.gfa.GFACamera.guide_names:
                self.GFAs[camera] = GFA
        self.needs_shutdown = parallel

    def shutdown(self):
        """Release any resources allocated in our constructor.
        """
        logging.info('Shutting down ETC...')
        if not self.needs_shutdown:
            return
        # Shutdown our GFA process pool.
        for camera in desietc.gfa.GFACamera.guide_names:
            logging.info(f'Releasing {camera} resources')
            self.pipes[camera].send('quit')
            self.processes[camera].join()
            self.shared_mem[camera].close()
            self.shared_mem[camera].unlink()
        self.needs_shutdown = False

    def set_image_path(self, image_path):
        """Set the path where future images should be written or None to prevent saving images.
        """
        if image_path is not None:
            self.image_path = pathlib.Path(image_path)
            if not self.image_path.exists():
                logging.error(f'Non-existant image_path: {image_path}.')
                self.image_path = None
            else:
                logging.info(f'Images will be written in: {image_path}.')
        else:
            logging.info('Images will no longer be written.')
            self.image_path = None

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

    @staticmethod
    def gfa_process(camera, calib_name, GMM, inset, measure, pipe):
        """Parallel process for a single GFA.
        """
        # Create a GFACamera object that shares its data and ivar arrays
        # with the parent process.
        bufname = ETC.BUFFER_NAME.format(camera)
        shared_mem = multiprocessing.shared_memory.SharedMemory(name=bufname)
        GFA = desietc.gfa.GFACamera(calib_name=calib_name, buffer=shared_mem.buf)
        # Command handling loop.
        while True:
            action = pipe.recv()
            if action == 'quit':
                shared_mem.close()
                pipe.send('bye')
                pipe.close()
                break
            # Handle other actions here...
            elif action == 'measure_psf':
                camera_result, psf_stack = ETC.measure_psf(GFA, GMM, inset, measure)
                pipe.send((camera_result, psf_stack))

    @staticmethod
    def measure_psf(thisGFA, GMM, inset, measure):
        """
        """
        camera_result = {}
        #  Find PSF-like objects.
        nstar = thisGFA.get_psfs()
        camera_result['nstar'] = nstar
        T, WT =  thisGFA.psf_stack
        if T is None:
            return camera_result
        # Measure the FWHM and FFRAC of the stacked PSF.
        fwhm, ffrac = measure.measure(T, WT)
        camera_result['fwhm'] = fwhm if fwhm > 0 else np.nan
        camera_result['ffrac'] = ffrac if ffrac > 0 else np.nan
        # Use a smaller size for GMM fitting.
        T, WT = T[inset, inset], WT[inset, inset]
        # Save a copy of the cropped stacked image.
        psf_stack = np.stack((T, WT))
        # Fit the PSF to a Gaussian mixture model. This is the slow step...
        gmm_params = GMM.fit(T, WT, maxgauss=3)
        if gmm_params is None:
            camera_result.update(dict(gmm=[], nll=0, ngauss=0))
        else:
            camera_result.update(dict(gmm=gmm_params, nll=GMM.best_nll, ngauss=GMM.ngauss))
        return camera_result, psf_stack

    def process_acquisition(self, data):
        """Process the initial GFA acquisition images.
        """
        ncamera = 0
        start = time.time()
        if not self.process_top_header(data['header'], 'acquisition image', update_ok=True):
            return False
        logging.info(f'Processing acquisition image for {self.night}/{self.exptag}.')
        # Pass 1: reduce the raw GFA data and measure the PSF.
        self.acquisition_data = {}
        self.psf_stack = {}
        self.noisy_gfa = set()
        pending = []
        for camera in desietc.gfa.GFACamera.guide_names:
            if camera not in data:
                logging.warn(f'No acquisition image for {camera}.')
                continue
            if not self.process_camera_header(data[camera]['header'], f'{camera} acquisition image'):
                continue
            if not self.preprocess_gfa(camera, data[camera], f'{camera} acquisition image'):
                continue
            if self.parallel:
                self.pipes[camera].send('measure_psf')
                pending.append(camera)
            else:
                self.acquisition_data[camera], self.psf_stack[camera] = self.measure_psf(
                    self.GFAs[camera], self.GMM, self.psf_inset, self.measure)
            ncamera += 1
        for camera in pending:
            logging.info(f'Waiting for {camera}...')
            # Collect the parallel measure_psf outputs.
            self.acquisition_data[camera], self.psf_stack[camera] = self.pipes[camera].recv()
        # Do a second pass to precompute the PSF dithers needed to fit guide stars.
        # This pass always runs in the main process since the dither array is ~6Mb
        # and we don't want to have to pass it between processes.
        nstars = {C: 0 for C in desietc.gfa.GFACamera.guide_names}
        nstars_tot = 0
        badfit = []
        fwhm_vec, ffrac_vec = [], []
        self.dithered_model = {}
        psf_model = {}
        for camera, camera_result in self.acquisition_data.items():
            nstars[camera] = camera_result['nstar']
            if nstars[camera] == 0:
                logging.warn(f'No stars found for {camera}.')
                continue
            if camera_result['nll'] > self.nll_threshold:
                logging.warn(f'Bad fit for {camera} with NLL={camera_result["nll"]:.1f}.')
                badfit.append(camera)
            nstars_tot += nstars[camera]
            fwhm_vec.append(camera_result.get('fwhm', np.nan))
            ffrac_vec.append(camera_result.get('ffrac', np.nan))
            gmm_params = camera_result.get('gmm', None)
            if gmm_params is None:
                logging.warn(f'PSF measurement failed for {camera}.')
                continue
            psf_model[camera] = self.GMM.predict(gmm_params)
            # Precompute dithered renderings of the model for fast guide frame fits.
            self.dithered_model[camera] = self.GMM.dither(gmm_params, self.xdither, self.ydither)
        # Update the current FWHM, FFRAC values now.
        fwhm, ffrac = -1, -1
        if np.any(np.isfinite(fwhm_vec)):
            fwhm = np.nanmedian(fwhm_vec)
        if np.any(np.isfinite(ffrac_vec)):
            ffrac = np.nanmedian(ffrac_vec)
        logging.info(f'Acquisition image quality using {nstars_tot} stars: FWHM={fwhm:.2f}", FFRAC={ffrac:.3}.')
        # Generate an acquisition analysis summary image.
        if self.image_path is not None:
            try:
                desietc.plot.save_acquisition_summary(
                    data['header'], psf_model, self.psf_stack, fwhm, ffrac, nstars, badfit, self.noisy_gfa,
                    self.image_path / f'PSF-{self.exptag}.png')
            except Exception as e:
                logging.error(f'Failed to save acquisition analysis summary image: {e}')
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
        if self.guide_stars is not None:
            logging.warning(f'Overwriting previous guide stars for {self.night}/{self.exptag}.')
        max_rsq = (0.5 * fiber_diam_um / pixel_size_um) ** 2
        profile = lambda x, y: 1.0 * (x ** 2 + y ** 2 < max_rsq)
        halfsize = self.psf_pixels // 2
        self.guide_stars = {}
        self.fiber_templates = {}
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
            templates = []
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
                yslice, xslice = (ylo, yhi), (xlo, xhi)
                # Calculate an antialiased fiber template for FFRAC calculations.
                fiber_dx, fiber_dy = x0 - ix, y0 - iy
                fiber = desietc.util.make_template(
                    self.psf_pixels, profile, dx=fiber_dx, dy=fiber_dy, normalized=False)
                stars.append(dict(
                    x0=x0, y0=y0, rmag=rmag, nelec_rate=nelec_rate,
                    fiber_dx=fiber_dx, fiber_dy=fiber_dy, yslice=yslice, xslice=xslice))
                # Save the template separately from the stars info since we do not want
                # to archive it in the ETC json output.
                templates.append(fiber)
            nstars.append(len(stars))
            if len(stars) > 0:
                self.guide_stars[camera] = stars
                self.fiber_templates[camera] = templates
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
        if self.dithered_model is None:
            logging.error('Ignoring guide frame before acquisition image.')
            return False
        if self.guide_stars is None:
            logging.error('Ignoring guide frame before guide stars.')
            return False
        # Loop over cameras with acquisition results.
        mjd_obs, exptime, camera_transp, camera_ffrac = [], [], [], []
        each_ffrac, each_transp = np.zeros(self.ngfa, np.float32), np.zeros(self.ngfa, np.float32)
        each_dx, each_dy = np.zeros(self.ngfa, np.float32), np.zeros(self.ngfa, np.float32)
        for icam, camera in enumerate(desietc.gfa.GFACamera.guide_names):
            if camera not in self.dithered_model:
                # We do not have acquisition results for this camera.
                continue
            dithered = self.dithered_model[camera]
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
            # Loop over guide stars for this camera.
            star_ffrac, star_transp, star_dx, star_dy = [], [], [], []
            templates = self.fiber_templates[camera]
            for istar, star in enumerate(self.guide_stars[camera]):
                # Extract the postage stamp for this star.
                xslice, yslice = slice(*star['xslice']), slice(*star['yslice'])
                D = thisGFA.data[xslice, yslice]
                DW = thisGFA.ivar[xslice, yslice]
                # Estimate the actual centroid in pixels, flux in electrons and
                # constant background level in electrons / pixel.
                dx, dy, flux, bg, nll, best_fit = self.GMM.fit_dithered(
                    self.xdither, self.ydither, dithered, D, DW)
                # Calculate centroid offset relative to the target fiber center.
                dx -= star['fiber_dx']
                dy -= star['fiber_dy']
                # Calculate the corresponding fiber fraction for this star.
                ffrac = np.sum(templates[istar] * best_fit)
                # Calculate the transparency as the ratio of measured / predicted electrons.
                transp = flux / (star['nelec_rate'] * self.exptime)
                star_transp.append(transp)
                star_ffrac.append(ffrac)
                star_dx.append(dx)
                star_dy.append(dy)
                nstar += 1
            camera_transp.append(np.nanmedian(star_transp))
            camera_ffrac.append(np.nanmedian(star_ffrac))
            # Prepare the auxiliary data saved to the ETC json file.
            if nstar > 0:
                each_ffrac[icam] = camera_ffrac[-1]
                each_transp[icam] = camera_transp[-1]
                each_dx[icam] = np.nanmean(star_dx)
                each_dy[icam] = np.nanmean(star_dy)
                logging.debug(
                    f'{camera}[{fnum}] ffrac={each_ffrac[icam]:.3f} transp={each_transp[icam]:.3f} ' +
                    f'dx={each_dx[icam]:.2f} dy={each_dy[icam]:.2f} nstar={nstar}.')
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
        self.thru_measurements.add(
            mjd_start, mjd_stop, thru, 0.01,
            aux_data=(each_ffrac, each_transp, each_dx, each_dy))
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
        each_flux, each_dflux = np.zeros(self.nsky, np.float32), np.zeros(self.nsky, np.float32)
        each_ndrop = np.zeros(self.nsky, np.int32)
        for i, camera in enumerate(self.SKY.sky_names):
            if camera not in data:
                logging.warn(f'No {camera} image for frame {fnum}.')
                continue
            if not self.process_camera_header(data[camera]['header'], f'{camera}[{fnum}]'):
                continue
            camera_flux, camera_dflux = self.SKY.setraw(data[camera]['data'], name=camera)
            camera_flux /= self.exptime
            camera_dflux /= self.exptime
            # Prepare the auxiliary data saved to the ETC json file.
            each_flux[i] = camera_flux
            each_dflux[i] = camera_dflux
            each_ndrop[i] = self.SKY.ndrop
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
        self.sky_measurements.add(
            mjd_start, mjd_stop, flux, dflux,
            aux_data=(each_flux, each_dflux, each_ndrop))
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
            thisGFA.setraw(data['data'], name=camera)
        except ValueError as e:
            logging.error(f'Failed to process {source} raw data: {e}')
            return False
        thisGFA.data -= thisGFA.get_dark_current(ccdtemp, self.exptime)
        # Flag this camera if it appears to have excessive noise
        if thisGFA.nbad_overscan >= self.nbad_threshold:
            if camera not in self.noisy_gfa:
                logging.warn(f'{camera} has excessive noise in {source}.')
                self.noisy_gfa.add(camera)
        return True

    def get_accumulated_teff(self, mjd_start, mjd_stop, MW_transp=1):
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
        teff = treal / sky * (MW_transp * thru) ** 2
        logging.info(f'Calculated treal={treal:.1f}s, teff={teff:.1f}s using ' +
            f'sky={sky:.3f}, thru={thru:.3f}, MW={MW_transp:.3f}.')
        return teff

    def start_exposure(self, night, expid, mjd, teff, cutoff, cosmic):
        """
        """
        logging.info(f'Starting {night}/{expid} at {mjd} with teff={teff:.0f}s, cutoff={cutoff:.0f}s, ' +
            f'cosmic={cosmic:.0f}s')
        self.exp_data = dict(
            expid=expid,
            mjd_start=mjd,
            teff=teff,
            cutof=cutoff,
            cosmic=cosmic,
        )

    def read_fiberassign(self, fname):
        """
        """
        self.fassign_data = dict(MW_transp=1)
        if not pathlib.Path(fname).exists():
            logging.error(f'Non-existent fiberassign file: {fname}.')
            return False
        fassign = fitsio.read(fname, ext='FIBERASSIGN')
        sel = (fassign['OBJTYPE'] == 'TGT')
        ntarget = np.count_nonzero(sel)
        if ntarget == 0:
            logging.error(f'Fiberassign file has not targets: {fname}.')
            return False
        Ebv = np.nanmedian(fassign[sel]['EBV'])
        MW_transp = 10 ** (-self.Ebv_coef * Ebv / 2.5)
        self.fassign_data = dict(ntarget=ntarget, Ebv=Ebv, MW_transp=MW_transp)
        logging.info(f'Tile has {ntarget} targets with median(Ebv)={Ebv:.5f} and MW transparency {MW_transp:.5f}.')
        return True

    def save_exposure(self, path):
        """
        """
        logging.info(f'Saving ETC outputs for {self.night}/{self.exptag} to {path}')
        if not path.exists():
            logging.error(f'Non-existent path: {path}.')
            return
        # Save all measurements after mjd_start.
        mjd1, mjd2 = self.exp_data['mjd_start'], None
        # Build a data structure to save via json.
        save = dict(
            expinfo=self.exp_data,
            fassign=self.fassign_data,
            acquisition=self.acquisition_data,
            guide_stars=self.guide_stars,
            thru=self.thru_measurements.save(mjd1, mjd2),
            sky=self.sky_measurements.save(mjd1, mjd2)
        )
        # Encode numpy types using python built-in types for serialization.
        fname = path / f'ETC-{self.exptag}.json'
        with open(fname, 'w') as f:
            json.dump(save, f, cls=desietc.util.NumpyEncoder)
            logging.info(f'Wrote {fname} for {self.night}/{self.exptag}')
        # Copy the acquisition analysis summary image.
        if self.image_path is not None:
            name = f'PSF-{self.exptag}.png'
            if (self.image_path / name).exists() and not (path / name).exists():
                logging.info(f'Copying {name} to {path}.')
                shutil.copy(self.image_path / name, path / name)
