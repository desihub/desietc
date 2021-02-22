"""The top-level ETC algorithm.
"""
import time
import json
import pathlib
import datetime
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
# - estimate GFA thru errors & use stars from all GFAs together
# - save git hash / version to json output
# - save cpu timing to json
# - truncate digits for np.float32 json output
# - warn if SNR is decreasing.

class ETCAlgorithm(object):

    SECS_PER_DAY = 86400
    BUFFER_NAME = 'ETC_{0}_buffer'

    def __init__(self, sky_calib, gfa_calib, psf_pixels=25, max_dither=7, num_dither=1200,
                 Ebv_coef=2.165, nbad_threshold=100, nll_threshold=10, grid_resolution=0.5, parallel=True):
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
        grid_resolution : float
            Resolution of grid (in seconds) to use for SNR calculations.
        parallel : bool
            Process GFA images in parallel when True.
        """
        self.Ebv_coef = Ebv_coef
        self.nbad_threshold = nbad_threshold
        self.nll_threshold = nll_threshold
        self.grid_resolution = grid_resolution / self.SECS_PER_DAY
        self.gfa_calib = gfa_calib
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
        self.exp_data = {}
        self.num_guide_frames = 0
        self.num_sky_frames = 0
        self.acquisition_data = None
        self.guide_stars = None
        self.image_path = None
        self.reset_counts()
        # How many GUIDE and SKY cameras do we expect?
        self.ngfa = len(desietc.gfa.GFACamera.guide_names)
        self.nsky = len(desietc.sky.SkyCamera.sky_names)
        # Initialize buffers to record our signal- and sky-rate measurements.
        aux_dtype = [
            ('sig', np.float32),                  # accumulated signal estimate
            ('bg', np.float32),                   # accumulated background estimate
            ('teff', np.float32),                 # accumulated effective time estimate in seconds
            ('tproj', np.float32),                # projected real time remaining in seconds
            ('final', np.float32),                # projected final effective time in seconds
            ('split', np.float32),                # projected time until next split in seconds
        ]
        # Define auxiliary data to save with each GFA or SKY measurement.
        self.thru_measurements = desietc.util.MeasurementBuffer(
            maxlen=1000, default_value=1, aux_dtype=[
                ('ffrac', np.float32, (self.ngfa,)),  # fiber fraction measured from single GFA
                ('transp', np.float32, (self.ngfa,)), # transparency measured from single GFA
                ('dx', np.float32, (self.ngfa,)),     # mean x shift of centroid from single GFA in pixels
                ('dy', np.float32, (self.ngfa,)),     # mean y shift of centroid from single GFA in pixels
            ] + aux_dtype)
        self.sky_measurements = desietc.util.MeasurementBuffer(
            maxlen=200, default_value=1, aux_dtype=[
                ('flux', np.float32, (self.nsky,)),   # sky flux meausured from a single SKYCAM.
                ('dflux', np.float32, (self.nsky,)),  # sky flux uncertainty meausured from a single SKYCAM.
                ('ndrop', np.int32, (self.nsky,)),    # number of fibers dropped from the camera flux estimate.
            ] + aux_dtype)
        # Initialize the SKY camera processor.
        self.SKY = desietc.sky.SkyCamera(calib_name=sky_calib)
        self.parallel = parallel
        # Check that shared mem is available if we need it.
        if parallel and not shared_memory_available:
            raise RuntimeError('Python >= 3.8 required for the parallel ETC option.')

    def start(self):
        """Perform startup resource allocation. Must be paired with a call to shutdown().
        """
        self.shutdown()
        logging.info('Starting up ETC...')
        # Initialize the GFA camera processor(s).
        self.GFAs = {}
        if self.parallel:
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
                    calib_name=self.gfa_calib, buffer=self.shared_mem[camera].buf)
            nbytes = bufsize * len(self.GFAs)
            logging.info(f'Allocated {nbytes/2**20:.1f}Mb of shared memory.')
            # Initialize per-GFA processes, each with its own pipe.
            context = multiprocessing.get_context(method='spawn')
            for camera in desietc.gfa.GFACamera.guide_names:
                self.pipes[camera], child = context.Pipe()
                self.processes[camera] = context.Process(
                    target=ETCAlgorithm.gfa_process, args=(
                        camera, self.gfa_calib, self.GMM, self.psf_inset, self.measure, child))
                self.processes[camera].start()
            logging.info(f'Initialized {len(self.GFAs)} GFA processes.')
        else:
            # All GFAs use the same GFACamera object.
            GFA = desietc.gfa.GFACamera(calib_name=self.gfa_calib)
            for camera in desietc.gfa.GFACamera.guide_names:
                self.GFAs[camera] = GFA
        self.needs_shutdown = self.parallel

    def shutdown(self):
        """Release any resources allocated in our constructor.
        """
        if not getattr(self, 'needs_shutdown', False):
            return
        logging.info('Shutting down ETC...')
        # Shutdown the process and release the shared memory allocated for each GFA.
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

    def reset_counts(self):
        self.total_guider_count = 0
        self.total_acq_count = 0
        self.total_sky_count = 0

    def check_top_header(self, header, source):
        """Check EXPID in the top-level header of an exposure.
        """
        if 'EXPID' not in header:
            logging.error(f'Missing EXPID keyword in {source}')
        elif 'expid' in self.exp_data:
            expid = header['EXPID']
            if expid != self.exp_data['expid']:
                logging.error(f'Got EXPID {expid} from {source} but expected {self.expid}.')

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
        """Parallel process entry point for a single GFA.
        """
        # Create a GFACamera object that shares its data and ivar arrays
        # with the parent process.
        bufname = ETCAlgorithm.BUFFER_NAME.format(camera)
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
                camera_result, psf_stack = ETCAlgorithm.measure_psf(GFA, GMM, inset, measure)
                pipe.send((camera_result, psf_stack))

    @staticmethod
    def measure_psf(thisGFA, GMM, inset, measure):
        """Measure the PSF for a single GFA.

        This static method can either called from :meth:`gfa_process` or from the
        main thread.
        """
        camera_result = {}
        #  Find PSF-like objects.
        nstar = thisGFA.get_psfs()
        camera_result['nstar'] = nstar
        T, WT =  thisGFA.psf_stack
        if T is None:
            return camera_result, None
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
        self.total_acq_count += 1
        logging.info(f'Processing acquisition image [{self.total_acq_count}] for {self.exptag}.')
        self.check_top_header(data['header'], 'acquisition image')
        # Pass 1: reduce the raw GFA data and measure the PSF.
        self.acquisition_data = {}
        self.psf_stack = {}
        self.noisy_gfa = set()
        pending = []
        acq_mjd, acq_exptime = None, None
        for camera in desietc.gfa.GFACamera.guide_names:
            if camera not in data:
                logging.warn(f'No acquisition image for {camera}.')
                continue
            if not self.process_camera_header(data[camera]['header'], f'{camera} acquisition image'):
                continue
            acq_mjd = acq_mjd or data[camera]['header']['MJD-OBS']
            acq_exptime = acq_exptime or data[camera]['header']['EXPTIME']
            if not self.preprocess_gfa(camera, data[camera], f'{camera} acquisition image'):
                continue
            if self.parallel:
                self.pipes[camera].send('measure_psf')
                pending.append(camera)
            else:
                self.acquisition_data[camera], self.psf_stack[camera] = self.measure_psf(
                    self.GFAs[camera], self.GMM, self.psf_inset, self.measure)
            ncamera += 1
        # Save the acquisition timing.
        self.exp_data['acq_mjd'] = acq_mjd
        self.exp_data['acq_exptime'] = acq_exptime
        # Collect results from any parallel processes.
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
            gmm_params = camera_result.get('gmm', [])
            if len(gmm_params) == 0:
                logging.warn(f'PSF measurement failed for {camera}.')
                continue
            psf_model[camera] = self.GMM.predict(gmm_params)
            # Precompute dithered renderings of the model for fast guide frame fits.
            self.dithered_model[camera] = self.GMM.dither(gmm_params, self.xdither, self.ydither)
        # Update the current FWHM, FFRAC values now.
        self.fwhm, self.ffrac = 0., 0.
        if np.any(np.isfinite(fwhm_vec)):
            self.fwhm = np.nanmedian(fwhm_vec)
        if np.any(np.isfinite(ffrac_vec)):
            self.ffrac = np.nanmedian(ffrac_vec)
        logging.info(f'Acquisition image quality using {nstars_tot} stars: ' +
            f'FWHM={self.fwhm:.2f}", FFRAC={self.ffrac:.3}.')
        # Generate an acquisition analysis summary image.
        if nstars_tot > 0 and self.image_path is not None:
            try:
                desietc.plot.save_acquisition_summary(
                    acq_mjd, self.exptag, psf_model, self.psf_stack, self.fwhm, self.ffrac, nstars,
                    badfit, self.noisy_gfa, self.image_path / f'etc-{self.exptag}.png')
            except Exception as e:
                logging.error(f'Failed to save acquisition analysis summary image: {e}')
        # Reset the guide frame counter and guide star data.
        self.num_guide_frames = 0
        self.guide_stars = None
        # Report timing.
        elapsed = time.time() - start
        logging.info(f'Acquisition processing took {elapsed:.2f}s for {ncamera} cameras.')
        return True

    def set_guide_stars(self, pm_info, zeropoint=27.06, fiber_diam_um=107, pixel_size_um=15):
        """Specify the guide star locations and magnitudes to use when analyzing
        each guide frame.  These are normally calculated by PlateMaker.
        """
        if self.guide_stars is not None:
            logging.warning(f'Overwriting previous guide stars for {self.exptag}.')
        max_rsq = (0.5 * fiber_diam_um / pixel_size_um) ** 2
        profile = lambda x, y: 1.0 * (x ** 2 + y ** 2 < max_rsq)
        halfsize = self.psf_pixels // 2
        self.guide_stars = {}
        self.fiber_templates = {}
        nstars = []
        _, ny, nx = desietc.gfa.GFACamera.buffer_shape
        for camera in desietc.gfa.GFACamera.guide_names:
            sel = (pm_info['GFA_LOC'] == camera) & (pm_info['MAG'] > 0) & (pm_info['GUIDE_FLAG']==1)
            if not np.any(sel):
                logging.warning(f'No guide stars available for {camera}.')
                nstars.append(0)
                continue
            # Loop over guide stars for this GFA.
            stars = []
            templates = []
            for i in np.where(sel)[0]:
                # Convert from PlateMaker indexing convention to (0,0) centered in bottom-left pixel.
                x0 = pm_info[i]['COL'] - 0.5
                y0 = pm_info[i]['ROW'] - 0.5
                rmag = pm_info[i]['MAG']
                # Convert flux to predicted detected electrons per second in the
                # GFA filter with nominal zenith atmospheric transmission.
                nelec_rate = 10 ** (-(rmag - zeropoint) / 2.5)
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
            logging.error(f'No usable guide stars for {self.exptag}.')
            return False
        nstars_msg = '+'.join([str(n) for n in nstars]) + '=' + str(np.sum(nstars))
        logging.info(f'Using {nstars_msg} guide stars for {self.exptag}.')
        return True

    def process_guide_frame(self, data):
        """Process a guide frame.
        """
        ncamera = nstar = 0
        start = time.time()
        fnum = self.num_guide_frames
        self.num_guide_frames += 1
        self.total_guider_count += 1
        logging.info(f'Processing guide frame {fnum} [{self.total_guider_count}] for {self.exptag}.')
        self.check_top_header(data['header'], f'guide[{fnum}]')
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
                logging.debug(f'Skipping {camera} guide frame {fnum} with no guide stars.')
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
                D = thisGFA.data[yslice, xslice]
                DW = thisGFA.ivar[yslice, xslice]
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
        # Did we get any useful data?
        if ncamera == 0:
            return False
        # Combine all cameras.
        transp = np.nanmedian(camera_transp) if np.any(np.isfinite(camera_transp)) else 0.
        ffrac = np.nanmedian(camera_ffrac) if np.any(np.isfinite(camera_ffrac)) else 0.
        thru = transp * ffrac
        logging.info(f'Guide transp={transp:.3f}, ffrac={ffrac:.3f}, thru={thru:.3f}.')
        # Record this measurement.
        mjd_start, mjd_stop = self.get_mjd_range(mjd_obs, exptime, f'guide[{fnum}]')
        # Use constant error until we have a proper estimate.
        self.thru_measurements.add(
            mjd_start, mjd_stop, thru, 0.01,
            aux_data=(each_ffrac, each_transp, each_dx, each_dy, 0, 0, 0, 0, 0, 0))
        # Update our accumulated signal if the shutter is open.
        nopen, nclose = len(self.shutter_open), len(self.shutter_close)
        if nopen == nclose + 1:
            self.update_accumulated(mjd_stop)
            self.thru_measurements.set_last(
                sig=self.accumulated_signal, bg=self.accumulated_background,
                teff=self.accumulated_eff_time, tproj=self.time_remaining,
                final=self.projected_eff_time, split=self.split_remaining)
        # Report timing.
        elapsed = time.time() - start
        logging.debug(f'Guide frame processing took {elapsed:.2f}s for {nstar} stars in {ncamera} cameras.')
        return True

    def process_sky_frame(self, data):
        """Process a SKY frame.
        """
        ncamera = 0
        start = time.time()
        fnum = self.num_sky_frames
        self.num_sky_frames += 1
        self.total_sky_count += 1
        logging.info(f'Processing sky frame {fnum} [{self.total_sky_count}] for {self.exptag}.')
        self.check_top_header(data['header'], f'sky[{fnum}]')
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
            aux_data=(each_flux, each_dflux, each_ndrop, 0, 0, 0, 0, 0, 0))
        # Update our accumulated background if the shutter is open.
        nopen, nclose = len(self.shutter_open), len(self.shutter_close)
        if nopen == nclose + 1:
            self.update_accumulated(mjd_stop)
            self.sky_measurements.set_last(
                sig=self.accumulated_signal, bg=self.accumulated_background,
                teff=self.accumulated_eff_time, tproj=self.time_remaining,
                final=self.projected_eff_time, split=self.split_remaining)
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

    def reset_accumulated(self):
        """
        """
        self.accumulated_mjd = desietc.util.date_to_mjd(datetime.datetime.utcnow(), utc_offset=0)
        self.accumulated_eff_time = self.accumulated_real_time = 0.
        self.accumulated_signal = self.accumulated_background = 0.
        self.time_remaining = self.split_remaining = self.projected_eff_time = 0.
        self.shutter_open = []
        self.shutter_close = []
        self.shutter_teff = []
        self.action = None

    def start_exposure(self, timestamp, expid, target_teff, target_type, max_exposure_time,
                       cosmics_split_time, max_splits, splittable):
        """Start a new exposure using parameters:
        expid:              next exposure id (int)
        target_teff:        target value of the effective exposure time in seconds (float)
        target_type:        a string describing the type of target to assume (DARK/BRIGHT/...)
        max_exposure_time:  Maximum exposure time in seconds (irrespective of accumulated SNR)
        cosmics_split_time: Time in second before requesting a cosmic ray split
        max_splits:         Maximum number of allowed cosmic splits.
        splittable:         Never do splits when this is False.
        """
        max_hours = 6
        if max_exposure_time <= 0 or max_exposure_time > max_hours * 3600:
            logging.warn(f'max_exposure_time={max_exposure_time} looks fishy: using {max_hours} hours.')
            max_exposure_time = max_hours * 3600
        self.exp_data = dict(
            expid=expid,
            target_teff=target_teff,
            target_type=target_type,
            max_exposure_time=max_exposure_time,
            cosmics_split_time=cosmics_split_time,
            max_splits=max_splits,
            splittable=splittable
        )
        self.exptag = str(expid).zfill(8)
        self.night = desietc.util.mjd_to_night(desietc.util.date_to_mjd(timestamp, utc_offset=0))
        logging.info(f'Start {self.night}/{self.exptag} at {timestamp} with teff={target_teff:.1f}s, type={target_type}, '
                     + f'max={max_exposure_time:.1f}s, split={cosmics_split_time:.1f}s, '
                     + f'maxsplit={max_splits}, splittable={splittable}.')
        self.reset_accumulated()

    def open_shutter(self, timestamp):
        """
        """
        mjd = desietc.util.date_to_mjd(timestamp, utc_offset=0)
        nopen, nclose = len(self.shutter_open), len(self.shutter_close)
        if nopen != nclose:
            logging.error(f'open_shutter called after {nopen} opens, {nclose} closes.')
            # Reset and try to keep going...
            self.shutter_open = []
            self.shutter_close = []
        if nopen == 0:
            self.exp_data['mjd_start'] = mjd
            delta = self.exp_data['max_exposure_time'] / self.SECS_PER_DAY
            self.exp_data['mjd_max'] = mjd + delta
            # Initialize a MJD grid to use for SNR calculations.
            # Grid values are bin centers, with spacing ~ grid_resolution.
            ngrid = int(np.ceil(delta / self.grid_resolution))
            self.mjd_grid = mjd + (np.arange(ngrid) + 0.5) * delta / ngrid
            # Initialize a shutter mask with 0=closed, 1=open.
            self.open_grid = np.zeros(ngrid)
            # Initialize signal and background rate grids.
            self.sig_grid = np.zeros_like(self.mjd_grid)
            self.bg_grid = np.zeros_like(self.mjd_grid)
            logging.debug(f'Created mjd_grid with {ngrid} values.')
        # Record this shutter opening.
        self.shutter_open.append(mjd)
        self.open_grid[self.mjd_grid >= mjd] = 1
        logging.info(f'Shutter open[{nopen}] at {timestamp}.')

    def close_shutter(self, timestamp):
        """
        """
        mjd = desietc.util.date_to_mjd(timestamp, utc_offset=0)
        nopen, nclose = len(self.shutter_open), len(self.shutter_close)
        if nopen != nclose + 1:
            logging.error(f'close_shutter called after {nopen} opens, {nclose} closes.')
            # Reset and try to keep going...
            self.shutter_open = [ mjd ]
            self.shutter_close = []
        # Update our SNR.
        self.update_accumulated(mjd)
        # Record this shutter closing.
        self.shutter_close.append(mjd)
        self.shutter_teff.append(self.accumulated_eff_time)
        self.open_grid[self.mjd_grid >= mjd] = 0
        logging.info(f'Shutter close[{nclose}] at {timestamp} after {self.accumulated_real_time:.1f}s ' +
            f'with actual teff={self.accumulated_eff_time:.1f}s')

    def stop_exposure(self, timestamp):
        """
        """
        mjd = desietc.util.date_to_mjd(timestamp, utc_offset=0)
        nopen, nclose = len(self.shutter_open), len(self.shutter_close)
        if nopen != nclose:
            logging.error(f'stop_exposure called after {nopen} opens, {nclose} closes.')
        # Record the stop index in our grid.
        self.grid_stop = np.searchsorted(self.mjd_grid, mjd) + 1
        logging.info(f'Stop {self.exptag} at {timestamp}')

    def exptime_factor(self, signal, background, MW_transp):
        """Calculate the ratio between effective and real exposure time using the
        specified (relative) accumulated signal and background rates, and MW
        transparency of the current tile.  These inputs are normalized so that
        values of 1 correspond to real = effective.
        """
        # Lookup the MW transparency to use, or default to 1.
        return (MW_transp * signal) ** 2 / background

    def update_accumulated(self, mjd_now):
        """Update our estimates of the accumulated signal, background and
        effective exposure time.  These values are reset to 0 by
        :meth:`start_exposure` then updated by each subsequent call to
        :meth:`process_guide_frame` or :meth:`process_sky`.
        """
        # Check that we have the NTS parameters for this exposure.
        for key in 'mjd_start', 'mjd_max', 'target_teff', 'cosmics_split_time', 'max_splits', 'splittable':
            if key not in self.exp_data:
                logging.error('update_accumulated: missing required NTS parameter "{key}".')
                return False
        mjd_start, mjd_max = self.exp_data['mjd_start'], self.exp_data['mjd_max']
        if mjd_now < mjd_start:
            logging.error('update_accumulated() called with mjd_now < mjd_start.')
            return False
        target = self.exp_data['target_teff']
        cosmic_split = self.exp_data['cosmics_split_time']
        max_splits = self.exp_data['max_splits']
        splittable = self.exp_data['splittable']
        # Check the state of the shutter.
        nopen, nclose = len(self.shutter_open), len(self.shutter_close)
        if nopen == nclose:
            logging.error(f'update_accumulated: called with shutter closed [{nopen},{nclose}].')
            return False
        elif nopen != nclose + 1:
            logging.error(f'update_accumulated: invalid shutter state [{nopen},{nclose}].')
            return False
        # Lookup the effective time accumulated on all previous splits for this exposure.
        prev_teff = 0 if nopen == 1 else self.shutter_teff[-1]
        # --- The shutter is open and we have the NTS parameters to use -------------------------
        # Record the timestamp of this update.
        self.accumulated_mjd = mjd_now
        # Get grid indices coresponding to the most recent shutter opening and now.
        mjd_open = self.shutter_open[-1]
        iopen, inow = np.searchsorted(self.mjd_grid, [mjd_open, mjd_now])
        past, future = slice(iopen, inow + 1), slice(inow, None)
        assert np.all(self.open_grid[iopen:])
        # Tabulate the signal and background since the most recent shutter opening.
        self.sig_grid[past] = self.thru_measurements.sample_grid(self.mjd_grid[past])
        self.bg_grid[past] = self.sky_measurements.sample_grid(self.mjd_grid[past])
        # Calculate the mean signal and background during this shutter open period.
        self.accumulated_signal = np.mean(self.sig_grid[past])
        self.accumulated_background = np.mean(self.bg_grid[past])
        # Lookup the MW transparency to use, or default to 1.
        MW_transp = self.fassign_data.get('MW_transp', 1.)
        # Calculate the accumulated effective exposure time for this shutter opening in seconds.
        self.accumulated_real_time = (mjd_now - mjd_open) * self.SECS_PER_DAY
        self.accumulated_eff_time = self.accumulated_real_time * self.exptime_factor(
            self.accumulated_signal, self.accumulated_background, MW_transp)
        logging.info(f'shutter[{nopen}] treal={self.accumulated_real_time:.1f}s, teff={self.accumulated_eff_time:.1f}s' +
            f' [+{prev_teff:.1f}s] using bg={self.accumulated_background:.3f}, sig={self.accumulated_signal:.3f}.')
        # Forecast the future signal and background.
        self.sig_grid[future] = self.thru_measurements.forecast_grid(self.mjd_grid[future])
        self.bg_grid[future] = self.sky_measurements.forecast_grid(self.mjd_grid[future])
        # Calculate accumulated signal and background, assuming the shutter stays open until mjd_max.
        n_grid = 1 + np.arange(len(self.mjd_grid) - iopen)
        accum_sig = np.cumsum(self.sig_grid[iopen:]) / n_grid
        accum_bg = np.cumsum(self.bg_grid[iopen:]) / n_grid
        # Calculate the corresponding accumulated effective exposure time in seconds.
        accum_treal = (self.mjd_grid[iopen:] - mjd_open) * self.SECS_PER_DAY
        accum_teff = accum_treal * self.exptime_factor(accum_sig, accum_bg, MW_transp)
        # When do we expect to close the shutter.
        self.action = None
        if mjd_now >= mjd_max:
            # We have already reached the maximum allowed exposure time.
            istop = inow
            self.action = ('stop', 'max exptime')
            logging.info(f'Maximum exposure time reached.')
        elif self.accumulated_eff_time + prev_teff >= target:
            # We have already reached the target.
            istop = inow
            self.action = ('stop', 'target reached')
            logging.info(f'Target reached.')
        elif accum_teff[-1] + prev_teff < target:
            # We will not reach the target before the mjd_max cutoff.
            istop = len(accum_teff) - 1
            logging.warn(f'Will not reach target before max exposure time.')
        else:
            # We expect to reach the target before mjd_max but are not there yet.
            istop = np.argmax(accum_teff + prev_teff >= target)
        # Lookup our expected stop time and time remaining, assuming the shutter stays open.
        mjd_stop = self.mjd_grid[iopen + istop]
        self.time_remaining = (mjd_stop - mjd_now) * self.SECS_PER_DAY
        # Calculate the corresponding effective time, including any previous shutters.
        self.projected_eff_time = accum_teff[istop] + prev_teff
        logging.info(f'Will stop in {self.time_remaining:.1f}s at teff={self.projected_eff_time:.1f}s' +
            f' (target={target:.1f}s).')
        # Calculate how many cosmic splits are remaining, for the currrent shutter.
        nsplit_remaining = int(np.ceil((mjd_stop - mjd_open) * self.SECS_PER_DAY / cosmic_split))
        # Calculate when the next split should be.
        mjd_split = mjd_open + (mjd_stop - mjd_open) / nsplit_remaining
        self.split_remaining = (mjd_split - mjd_now) * self.SECS_PER_DAY
        logging.info(f'Next split ({nopen} of {nclose+nsplit_remaining}) in {self.split_remaining:.1f}s')
        if (splittable and self.action is None and
            nsplit_remaining > 1 and nopen < max_splits and self.split_remaining <= 0):
            self.action = ('split', 'cosmic split')
        if self.action is not None:
            logging.info(f'Recommended action is {self.action}.')
        return True

    def read_fiberassign(self, fname):
        """
        """
        self.fassign_data = dict(MW_transp=1)
        if not pathlib.Path(fname).exists():
            logging.error(f'Non-existent fiberassign file: {fname}.')
            return False
        logging.info(f'Reading fiber assignments from {fname}.')
        # Read the header for some metadata.
        header = fitsio.read_header(fname, ext=0)
        # Check for required keywords.
        self.fassign_data = {}
        for key in ('TILEID', 'TILERA', 'TILEDEC', 'FIELDROT'):
            if key not in header:
                logging.warn(f'Fiberassign file is missing header keyword {key}.')
                self.fassign_data[key] = None
            else:
                self.fassign_data[key] = header[key]
        tileid = self.fassign_data['TILEID']
        # Read the binary table of fiber assignments for this tile.
        fassign = fitsio.read(fname, ext='FIBERASSIGN')
        # Calculate the median EBV of non-sky targets.
        sel = (fassign['OBJTYPE'] == 'TGT') & np.isfinite(fassign['EBV'])
        ntarget = np.count_nonzero(sel)
        if ntarget < 100:
            logging.warn(f'Fiberassignment for tile {tileid} only has {ntarget} targets.')
            return False
        if ntarget > 0:
            Ebv = np.median(fassign[sel]['EBV'])
        else:
            Ebv = 0.
        # Calculate the corresponding MW transparency factor.
        MW_transp = 10 ** (-self.Ebv_coef * Ebv / 2.5)
        # Save what we need later and will write to our json file.
        self.fassign_data.update(dict(ntarget=ntarget, Ebv=Ebv, MW_transp=MW_transp))
        logging.info(f'Tile {tileid} has {ntarget} targets with median(Ebv)={Ebv:.5f} and MW transp {MW_transp:.5f}.')
        return True

    def save_exposure(self, path):
        """
        """
        logging.info(f'Saving ETC outputs for {self.exptag} to {path}')
        if not path.exists():
            logging.error(f'Non-existent path: {path}.')
            return
        # Save all measurements after mjd_start.
        mjd1, mjd2 = self.exp_data['mjd_start'], None
        # Trim the grid to the stop time.

        # Build a data structure to save via json.
        save = dict(
            expinfo=self.exp_data,
            fassign=self.fassign_data,
            acquisition=self.acquisition_data,
            guide_stars=self.guide_stars,
            shutter=dict(
                open=self.shutter_open,
                close=self.shutter_close,
                teff=self.shutter_teff,
            ),
            thru=self.thru_measurements.save(mjd1, mjd2),
            sky=self.sky_measurements.save(mjd1, mjd2),
            grid=dict(
                mjd=self.mjd_grid[:self.grid_stop],
                sig=self.sig_grid[:self.grid_stop],
                bg=self.bg_grid[:self.grid_stop],
                open=self.open_grid[:self.grid_stop],
            )
        )
        # Encode numpy types using python built-in types for serialization.
        fname = path / f'etc-{self.exptag}.json'
        with open(fname, 'w') as f:
            json.dump(save, f, cls=desietc.util.NumpyEncoder)
            logging.info(f'Wrote {fname} for {self.exptag}')
        # Copy the acquisition analysis summary image.
        if self.image_path is not None:
            name = f'psf-{self.exptag}.png'
            if (self.image_path / name).exists() and not (path / name).exists():
                logging.info(f'Copying {name} to {path}.')
                shutil.copy(self.image_path / name, path / name)
