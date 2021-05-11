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
import desietc.accum
import desietc.util
import desietc.plot

# TODO:
# - estimate GFA thru errors & use stars from all GFAs together
# - ignore GFAs with flagged issues (noise, ...)?
# - save git hash / version to json output
# - save cpu timing to json
# - truncate digits for np.float32 json output
# - warn if SNR is decreasing.

class ETCAlgorithm(object):

    SECS_PER_DAY = 86400
    BUFFER_NAME = 'ETC_{0}_buffer'
    FFRAC_NOM = dict(PSF=0.56198, ELG=0.41220, BGS=0.18985)

    def __init__(self, sky_calib, gfa_calib, psf_pixels=25, guide_pixels=31, max_dither=7, num_dither=1200,
                 Ebv_coef=2.165, X_coef=0.114, ffrac_ref=0.56, nbad_threshold=100, nll_threshold=100,
                 avg_secs=300, avg_min_values=8, grid_resolution=0.5, min_exptime_secs=0, parallel=True):
        """Initialize once per session.

        Parameters
        ----------
        sky_calib : str
            Path to the SKY camera calibration file to use.
        gfa_calib : str
            Path to the GFA camera calibration file to use.
        psf_pixels : int
            Size of stamp to use for acquisition image PSF measurements. Must be odd.
        guide_pixels : int
            Size of stamp to use for guide star measurements. Must be odd.
        max_dither : float
            Maximum dither to use in pixels relative to PSF centroid.
        num_dither : int
            Number of (x,y) dithers to use, extending out to max_dither with
            decreasing density.
        Ebv_coef : float
            Coefficient to use for converting Ebv into an equivalent MW
            transparency via 10 ** (-coef * Ebv / 2.5)
        X_coef : float
            Coefficient to use to convert observed transparency at airmass X
            into an equivalent X=1 value via 10 ** (-coef * (X-1) / 2.5)
        ffrac_ref : float
            Reference value of the fiber fraction that defines nominal conditions.
        nbad_threshold : int
            Maximum number of allowed bad overscan pixel values before a GFA is
            flagged as noisy.
        nll_threshold : float
            Maximum allowed GMM fit NLL value before a PSF fit is flagged as
            potentially bad.
        avg_secs : float
            Compute running averages over this time interval in seconds.
        avg_min_values : int
            A running average requires at least this many values.
        grid_resolution : float
            Resolution of grid (in seconds) to use for SNR calculations.
        min_exptime_secs : float
            Minimum allowed spectrograph exposure time in seconds. A stop or split
            request will never be issued until this interval has elapsed after
            the spectrograph shutters open.
        parallel : bool
            Process GFA images in parallel when True.
        """
        self.Ebv_coef = Ebv_coef
        self.X_coef = X_coef
        self.nbad_threshold = nbad_threshold
        self.nll_threshold = nll_threshold
        self.avg_secs = avg_secs
        self.avg_min_values = avg_min_values
        self.gfa_calib = gfa_calib
        # Capture a git description of the code we are running.
        self.git = desietc.util.git_describe()
        if self.git:
            logging.info(f'Running desietc {self.git}')
        else:
            logging.warning('Could not determine git info.')
        # Initialize PSF fitting.
        if psf_pixels % 2 == 0:
            raise ValueError('psf_pixels must be odd.')
        psf_grid = np.arange(psf_pixels + 1) - psf_pixels / 2
        self.GMMpsf = desietc.gmm.GMMFit(psf_grid, psf_grid)
        self.psf_pixels = psf_pixels
        psf_stacksize = desietc.gfa.GFACamera.psf_stacksize
        if self.psf_pixels > psf_stacksize:
            raise ValueError(f'psf_pixels must be <= {psf_stacksize}.')
        # Use a smaller stamp for PSF measurements.
        ntrim = (psf_stacksize - self.psf_pixels) // 2
        self.psf_inset = slice(ntrim, ntrim + self.psf_pixels)
        self.measure = desietc.util.PSFMeasure(psf_stacksize)
        # Initialize guide star analysis.
        if guide_pixels % 2 == 0:
            raise ValueError('guide_pixels must be odd.')
        self.guide_pixels = guide_pixels
        guide_grid = np.arange(guide_pixels + 1) - guide_pixels / 2
        self.GMMguide = desietc.gmm.GMMFit(guide_grid, guide_grid)
        self.xdither, self.ydither = desietc.util.diskgrid(num_dither, max_dither, alpha=2)
        # Create the mask of background pixels.
        bg_radius = 11 # pixels
        BGprofile = lambda x, y: 1.0 * ((x ** 2 + y ** 2 > bg_radius ** 2))
        self.BGmask = (desietc.util.make_template(guide_pixels, BGprofile, dx=0, dy=0, normalized=False) > 0.5)
        # Initialize analysis results.
        self.exp_data = {}
        self.num_guide_frames = 0
        self.num_sky_frames = 0
        self.acquisition_data = None
        self.guide_stars = None
        self.image_path = None
        # Initialize observing conditions updated after each GFA or SKY frame.
        self.seeing = None
        self.ffrac_psf = None
        self.rel_ffrac_sbprof = None
        self.transp_obs = None
        self.transp_zenith = None
        self.skylevel = None
        # Initialize variables used for telemetry.
        self.ffrac_avg = None
        self.transp_avg = None
        self.thru_avg = None
        self.ffrac_psf = None
        self.ffrac_elg = None
        self.ffrac_bgs = None
        # Initialize call counters.
        self.reset_counts()
        # How many GUIDE and SKY cameras do we expect?
        self.ngfa = len(desietc.gfa.GFACamera.guide_names)
        self.nsky = len(desietc.sky.SkyCamera.sky_names)
        # Initialize buffers to record our signal- and sky-rate measurements.
        self.thru_measurements = desietc.util.MeasurementBuffer(
            maxlen=1000, default_value=1, aux_dtype=[
                ('ffrac_gfa', np.float32, (self.ngfa,)),  # fiber fraction measured from single GFA
                ('transp_gfa', np.float32, (self.ngfa,)), # transparency measured from single GFA
                ('dx_gfa', np.float32, (self.ngfa,)),     # mean x shift of centroid from single GFA in pixels
                ('dy_gfa', np.float32, (self.ngfa,)),     # mean y shift of centroid from single GFA in pixels
                ('transp_obs', np.float32),               # Observed transparency averaged over all guide stars
                ('transp_zenith', np.float32),            # Zenith transparency averaged over all guide stars
                ('ffrac_psf', np.float32),                # FFRAC for PSF profile averaged over all guide stars
                ('ffrac_elg', np.float32),                # FFRAC for ELG profile averaged over all guide stars
                ('ffrac_bgs', np.float32),                # FFRAC for BGS profile averaged over all guide stars
                ('thru_psf', np.float32),                 # TRANSP*FFRAC for PSF profile
            ])
        self.sky_measurements = desietc.util.MeasurementBuffer(
            maxlen=200, default_value=1, padding=900, aux_dtype=[
                ('flux', np.float32, (self.nsky,)),   # sky flux meausured from a single SKYCAM.
                ('dflux', np.float32, (self.nsky,)),  # sky flux uncertainty meausured from a single SKYCAM.
                ('ndrop', np.int32, (self.nsky,)),    # number of fibers dropped from the camera flux estimate.
            ])
        # Initialize exposure accumulator.
        self.accum = desietc.accum.Accumulator(
            sig_buffer=self.thru_measurements, bg_buffer=self.sky_measurements,
            grid_resolution=grid_resolution, min_exptime_secs=min_exptime_secs)
        # Initialize the SKY camera processor.
        self.SKY = desietc.sky.SkyCamera(calib_name=sky_calib)
        # Prepare for parallel processing if necessary.
        self.GFAs = {}
        self.parallel = parallel
        if parallel:
            # Check that shared mem is available.
            if not shared_memory_available:
               raise RuntimeError('Python >= 3.8 required for the parallel ETC option.')
            # Initialize unallocated resources.
            self.shared_mem = {camera: None for camera in desietc.gfa.GFACamera.guide_names}
            self.pipes = {camera: None for camera in desietc.gfa.GFACamera.guide_names}
            self.processes = {camera: None for camera in desietc.gfa.GFACamera.guide_names}
        else:
            # All GFAs use the same GFACamera object.
            GFA = desietc.gfa.GFACamera(calib_name=self.gfa_calib)
            for camera in desietc.gfa.GFACamera.guide_names:
                self.GFAs[camera] = GFA

    def start(self):
        """Perform startup resource allocation. Must be paired with a call to shutdown().
        """
        if not self.parallel:
            return
        logging.info('Starting up ETC...')
        # Initialize the GFA camera processor(s).
        # Allocate shared-memory buffers for each guide camera's GFACamera object.
        bufsize = desietc.gfa.GFACamera.buffer_size
        for camera in desietc.gfa.GFACamera.guide_names:
            if self.shared_mem[camera] is not None:
                logging.warning(f'Shared memory already allocated for {camera}?')
            bufname = self.BUFFER_NAME.format(camera)
            try:
                self.shared_mem[camera] = multiprocessing.shared_memory.SharedMemory(
                    name=bufname, size=bufsize, create=True)
            except FileExistsError:
                logging.error(f'Reconnecting to previously allocated shared memory for {camera}...')
                self.shared_mem[camera] = multiprocessing.shared_memory.SharedMemory(
                    name=bufname, size=bufsize, create=False)
            self.GFAs[camera] = desietc.gfa.GFACamera(
                calib_name=self.gfa_calib, buffer=self.shared_mem[camera].buf)
        nbytes = bufsize * len(self.GFAs)
        logging.info(f'Allocated {nbytes/2**20:.1f}Mb of shared memory.')
        # Initialize per-GFA processes, each with its own pipe.
        context = multiprocessing.get_context(method='spawn')
        for camera in desietc.gfa.GFACamera.guide_names:
            if self.pipes[camera] is not None and self.processes[camera] is not None and self.processes[camera].is_alive():
                logging.error(f'Process already running for {camera}. Will not restart...')
            else:
                self.pipes[camera], child = context.Pipe()
                self.processes[camera] = context.Process(
                    target=ETCAlgorithm.gfa_process, daemon=True, args=(
                        camera, self.gfa_calib, self.GMMpsf, self.psf_inset, self.measure, child))
                self.processes[camera].start()
        logging.info(f'Initialized {len(self.GFAs)} GFA processes.')

    def shutdown(self, timeout=5):
        """Release any resources allocated in our constructor.
        """
        if not self.parallel:
            return
        logging.info('Shutting down ETC...')
        # Shutdown the process and release the shared memory allocated for each GFA.
        for camera in desietc.gfa.GFACamera.guide_names:
            logging.info(f'Releasing {camera} resources')
            try:
                if self.processes[camera] is not None and self.processes[camera].is_alive():
                    self.pipes[camera].send('quit')
                    self.processes[camera].join(timeout=timeout)
                    logging.info(f'Shutdown pipe and process for {camera}.')
                else:
                    logging.warning(f'Process for {camera} is not alive.')
            except Exception as e:
                logging.error(f'Failed to shutdown pipe and process for {camera}: {e}')
            self.pipes[camera] = None
            self.processes[camera] = None
            try:
                if self.shared_mem[camera] is not None:
                    self.shared_mem[camera].close()
                    self.shared_mem[camera].unlink()
                    logging.info(f'Released shared memory for {camera}.')
                else:
                    logging.warning(f'No shared memory allocated for {camera}.')
            except Exception as e:
                logging.error(f'Failed to released shared memory for {camera}: {e}')
            self.shared_mem[camera] = None
        self.GFAs = {}

    def set_image_path(self, image_path, create=False):
        """Set the path where future images should be written or None to prevent saving images.
        """
        if image_path is not None:
            self.image_path = pathlib.Path(image_path)
            # Try to create if requested and necessary.
            if create:
                try:
                    self.image_path.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logging.error(f'Failed to create {self.image_path}: {e}')
                    self.image_path = None
                    return False
            else:
                if not self.image_path.exists():
                    logging.error(f'Non-existent image path: {self.image_path}')
                    self.image_path = None
                    return False
            logging.info(f'Images will be written in: {image_path}.')
        else:
            logging.info('Images will no longer be written.')
            self.image_path = None

    def reset_counts(self):
        self.total_gfa_count = 0
        self.total_sky_count = 0
        self.total_desi_count = 0

    def check_top_header(self, header, source):
        """Check EXPID in the top-level GUIDER or SKY header of an exposure.
        """
        if 'EXPID' not in header:
            logging.error(f'Missing EXPID keyword in {source}')
        elif 'expid' in self.exp_data:
            expid = header['EXPID']
            expected = self.exp_data['expid']
            if expid != expected:
                logging.error(f'Got EXPID {expid} from {source} but expected {expected}.')

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
        logging.info(f'Processing acquisition image for {self.exptag}.')
        self.check_top_header(data['GUIDER']['header'], 'acquisition image')
        # Pass 1: reduce the raw GFA data and measure the PSF.
        self.acquisition_data = {}
        self.psf_stack = {}
        self.noisy_gfa = set()
        pending = []
        # Use the first per-camera header for these values since the top header is missing EXPTIME
        # and has MJD-OBS a few seconds earlier (the request time?)
        need_hdr = True
        hdr = {'MJD-OBS': None, 'EXPTIME': None, 'MOUNTAZ': None, 'MOUNTEL': 1., 'MOUNTHA': 0.}
        acq_mjd, acq_exptime, acq_airmass = None, None, None
        for camera in desietc.gfa.GFACamera.guide_names:
            if camera not in data:
                logging.warning(f'No acquisition image for {camera}.')
                continue
            if not self.process_camera_header(data[camera]['header'], f'{camera} acquisition image'):
                continue
            if need_hdr:
                for key in hdr:
                    value = data[camera]['header'].get(key, None)
                    if value is None:
                        logging.error(f'Acquisition header missing "{key}": using default {hdr[key]}.')
                    else:
                        hdr[key] = value
                need_hdr = False
            if not self.preprocess_gfa(camera, data[camera], f'{camera} acquisition image'):
                continue
            if self.parallel:
                self.pipes[camera].send('measure_psf')
                pending.append(camera)
            else:
                self.acquisition_data[camera], self.psf_stack[camera] = self.measure_psf(
                    self.GFAs[camera], self.GMMpsf, self.psf_inset, self.measure)
            ncamera += 1
        # Calculate the atmospheric extintion factor to use.
        X = desietc.util.cos_zenith_to_airmass(np.sin(np.deg2rad(hdr['MOUNTEL'])))
        self.atm_extinction = 10 ** (-self.X_coef * (X - 1) / 2.5)
        logging.info(f'Using atmospheric extinction {self.atm_extinction:.4f} at X={X:.3f}.')
        # Save the header info.
        self.exp_data.update(dict(
            acq_mjd=hdr['MJD-OBS'],
            acq_exptime=hdr['EXPTIME'],
            hour_angle=hdr['MOUNTHA'],
            elevation=hdr['MOUNTEL'],
            azimuth=hdr['MOUNTAZ'],
            airmass=np.float32(X),
            atm_extinction=np.float32(self.atm_extinction),
        ))
        self.total_gfa_count += 1
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
                logging.warning(f'No stars found for {camera}.')
                continue
            if camera_result['nll'] > self.nll_threshold:
                logging.warning(f'Bad fit for {camera} with NLL={camera_result["nll"]:.1f}.')
                badfit.append(camera)
            nstars_tot += nstars[camera]
            fwhm_vec.append(camera_result.get('fwhm', np.nan))
            ffrac_vec.append(camera_result.get('ffrac', np.nan))
            gmm_params = camera_result.get('gmm', [])
            if len(gmm_params) == 0:
                logging.warning(f'PSF measurement failed for {camera}.')
                continue
            psf_model[camera] = self.GMMpsf.predict(gmm_params)
            # Precompute dithered renderings of the model for fast guide frame fits.
            self.dithered_model[camera] = self.GMMguide.dither(gmm_params, self.xdither, self.ydither)
        # Update the current FWHM, FFRAC values now.
        self.seeing, ffrac_psf = 0., 0.
        if np.any(np.isfinite(fwhm_vec)):
            self.seeing = np.nanmedian(fwhm_vec)
        if np.any(np.isfinite(ffrac_vec)):
            self.ffrac_psf = np.nanmedian(ffrac_vec)
        self.exp_data.update(dict(acq_fwhm=np.float32(self.seeing), acq_ffrac=np.float32(self.ffrac_psf)))
        logging.info(f'Acquisition image quality using {nstars_tot} stars: ' +
            f'FWHM={self.seeing:.2f}", FFRAC={self.ffrac_psf:.3}.')
        # Generate an acquisition analysis summary image.
        if nstars_tot > 0 and self.image_path is not None:
            try:
                png_name = self.image_path / f'etc-{self.exptag}.png'
                desietc.plot.save_acquisition_summary(
                    hdr['MJD-OBS'], self.exptag, psf_model, self.psf_stack, self.seeing, self.ffrac_psf,
                    nstars, badfit, self.noisy_gfa, png_name)
                logging.info(f'Wrote {png_name}')
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
        halfsize = self.guide_pixels // 2
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
                fiber = desietc.gfa.get_fiber_profile(x0, y0, camera, self.guide_pixels)
                stars.append(dict(
                    x0=np.float32(x0), y0=np.float32(y0), rmag=np.float32(rmag),
                    nelec_rate=np.float32(nelec_rate),
                    fiber_dx=np.float32(fiber_dx), fiber_dy=np.float32(fiber_dy),
                    yslice=yslice, xslice=xslice))
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
        nstars_tot = np.sum(nstars)
        self.guide_stars['nstars'] = nstars_tot
        nstars_msg = '+'.join([str(n) for n in nstars]) + '=' + str(nstars_tot)
        logging.info(f'Using {nstars_msg} guide stars for {self.exptag}.')
        return True

    def process_guide_frame(self, data, timestamp):
        """Process a guide frame.
        """
        ncamera = nstar = 0
        start = time.time()
        fnum = self.num_guide_frames
        self.num_guide_frames += 1
        self.total_gfa_count += 1
        logging.info(f'Processing guide frame {fnum} [{self.total_gfa_count}] for {self.exptag} at {timestamp}.')
        self.check_top_header(data['GUIDER']['header'], f'guide[{fnum}]')
        if self.dithered_model is None:
            logging.error('Ignoring guide frame before acquisition image.')
            return False
        if self.guide_stars is None:
            logging.error('Ignoring guide frame before guide stars.')
            return False
        nstars_tot = self.guide_stars['nstars']
        star_fluxsum = np.zeros(nstars_tot)
        star_profsum = np.zeros(nstars_tot)
        star_fluxnorm = np.zeros(nstars_tot)
        star_ffracs = np.zeros((nstars_tot, 3))
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
                FIBER = templates[istar]
                fluxnorm = star['nelec_rate'] * self.exptime

                ###### Original PSF model fit

                # Estimate the actual centroid in pixels, flux in electrons and
                # constant background level in electrons / pixel.
                dx, dy, flux, bg, nll, best_fit = self.GMMguide.fit_dithered(
                    self.xdither, self.ydither, dithered, D, DW)
                # Calculate centroid offset relative to the target fiber center.
                dx -= star['fiber_dx']
                dy -= star['fiber_dy']
                # Calculate the corresponding fiber fraction for this star.
                ffrac = np.sum(FIBER * best_fit)
                # Calculate the transparency as the ratio of measured / predicted electrons.
                transp = flux / fluxnorm
                star_transp.append(transp)
                star_ffrac.append(ffrac)
                star_dx.append(dx)
                star_dy.append(dy)

                ###### New pixel-level analysis

                # Blur by 0.15 pixel to reduce artifacts from isolated pixels with large ivar.
                D, DW = desietc.util.blur(D, DW)
                # Estimate and subtract the flat background level.
                bg_per_pixel = desietc.util.robust_median(D[self.BGmask])
                D -= bg_per_pixel
                # Sum the signal flux over the full stamp.
                star_fluxsum[nstar] = D.sum()
                # Sum the signal flux over the fiber profile.
                star_profsum[nstar] = (D * FIBER).sum()
                # Calculate fiber fractions.
                star_ffracs[nstar] = desietc.util.get_fiber_fractions(D, FIBER)
                # Calculate the expected total PSF flux with transparency=1.
                star_fluxnorm[nstar] = fluxnorm
                #logging.debug(f'{camera}[{fnum}] {star_fluxsum[-1]:.1f} {star_profsum[-1]:.1f} {star_fluxnorm[-1]:.1f} {star_ffracs[-1][0]:.4f} {star_ffracs[-1][1]:.4f} {star_ffracs[-1][2]:.4f}')

                nstar += 1

            camera_transp.append(np.nanmedian(star_transp))
            camera_ffrac.append(np.nanmedian(star_ffrac))
            # Prepare the auxiliary data saved to the ETC json file.
            if len(star_dx) > 0:
                each_ffrac[icam] = camera_ffrac[-1]
                each_transp[icam] = camera_transp[-1]
                each_dx[icam] = np.nanmean(star_dx)
                each_dy[icam] = np.nanmean(star_dy)
                logging.debug(
                    f'{camera}[{fnum}] ffrac={each_ffrac[icam]:.3f} transp={each_transp[icam]:.3f} ' +
                    f'dx={each_dx[icam]:.2f} dy={each_dy[icam]:.2f} nstar={len(star_dx)}.')
            mjd_obs.append(self.mjd_obs)
            exptime.append(self.exptime)
            ncamera += 1
        # Did we get any useful data?
        if ncamera == 0:
            return False

        # Combine all cameras...
        self.transp_obs = desietc.util.robust_median(star_fluxsum[:nstar] / star_fluxnorm[:nstar])
        self.thru_psf = desietc.util.robust_median(star_profsum[:nstar] / star_fluxnorm[:nstar])
        self.ffrac_psf = self.thru_psf / self.transp_obs
        elg_ratio = desietc.util.robust_median(star_ffracs[:nstar,1] / star_ffracs[:nstar,0])
        bgs_ratio = desietc.util.robust_median(star_ffracs[:nstar,2] / star_ffracs[:nstar,0])
        self.thru_elg = self.thru_psf * elg_ratio
        self.thru_bgs = self.thru_psf * bgs_ratio
        self.ffrac_elg = self.ffrac_psf * elg_ratio
        self.ffrac_bgs = self.ffrac_psf * bgs_ratio
        #logging.debug(f'[{fnum}] {self.transp_obs:.4f} {self.ffrac_psf:.4f} {self.thru_psf:.4f} {self.thru_elg:.4f} {self.thru_bgs:.4f}')

        # Calculate the throughput for the specified source profile *relative* to nominal conditions.
        # The constants below were calculated with GalSim for the nominal 1.1" Moffat profile with beta=3.5
        # convolved with a hlr=0.45" Sersic n=1 (ELG) or a hlr=1.5" Sersic n=4 (BGS) round profile.
        sbprof = self.exp_data['sbprof']
        if sbprof == 'ELG':
            self.thru_sbprof_rel = self.thru_elg / self.FFRAC_NOM['ELG']
        elif sbprof == 'BGS':
            self.thru_sbprof_rel = self.thru_bgs / self.FFRAC_NOM['BGS']
        elif sbprof == 'FLT':
            self.thru_sbprof_rel = self.transp_obs
        else:
            self.thru_sbprof_rel = self.thru_psf / self.FFRAC_NOM['PSF']

        # Adjust the transparency to X=1.
        self.transp_zenith = self.transp_obs / self.atm_extinction
        # Record this measurement.
        logging.info(f'Guide transp={self.transp_obs:.3f}, PSF-ffrac={self.ffrac_psf:.3f}, {sbprof}-thru-rel={self.thru_sbprof_rel:.3f}.')
        mjd_start, mjd_stop = self.get_mjd_range(mjd_obs, exptime, f'guide[{fnum}]')
        # Use constant error until we have a proper estimate.
        self.thru_measurements.add(
            mjd_start, mjd_stop, self.thru_sbprof_rel, 0.01,
            aux_data=(each_ffrac, each_transp, each_dx, each_dy,
                      self.transp_obs, self.transp_zenith,
                      self.ffrac_psf, self.ffrac_elg, self.ffrac_bgs, self.thru_psf))
        # Update our accumulated signal if the shutter is open.
        if self.accum.shutter_is_open:
            mjd_now = desietc.util.date_to_mjd(timestamp, utc_offset=0)
            self.accum.update('GFA', mjd_stop, mjd_now)
        # Update running averages.
        self.transp_avg = self.thru_measurements.average(mjd_stop, self.avg_secs, self.avg_min_values, field='transp_zenith')
        self.ffrac_avg = self.thru_measurements.average(mjd_stop, self.avg_secs, self.avg_min_values, field='ffrac_psf')
        if self.ffrac_avg != None:
            self.ffrac_avg /= self.FFRAC_NOM['PSF']
        self.thru_avg = self.thru_measurements.average(mjd_stop, self.avg_secs, self.avg_min_values, field='thru_psf')
        if self.thru_avg != None:
            self.thru_avg /= self.FFRAC_NOM['PSF']
        # Report timing.
        elapsed = time.time() - start
        logging.info(f'Guide frame processing took {elapsed:.2f}s for {nstar} stars in {ncamera} cameras.')
        return True

    def process_sky_frame(self, data, timestamp):
        """Process a SKY frame.
        """
        ncamera = 0
        start = time.time()
        fnum = self.num_sky_frames
        self.num_sky_frames += 1
        self.total_sky_count += 1
        logging.info(f'Processing sky frame {fnum} [{self.total_sky_count}] for {self.exptag} at {timestamp}.')
        self.check_top_header(data['SKY']['header'], f'sky[{fnum}]')
        flux, ivar = 0, 0
        mjd_obs, exptime = [], []
        each_flux, each_dflux = np.zeros(self.nsky, np.float32), np.zeros(self.nsky, np.float32)
        each_ndrop = np.zeros(self.nsky, np.int32)
        for i, camera in enumerate(self.SKY.sky_names):
            if camera not in data:
                logging.warning(f'No {camera} image for frame {fnum}.')
                continue
            if not self.process_camera_header(data[camera]['header'], f'{camera}[{fnum}]'):
                continue
            camera_flux, camera_dflux = self.SKY.setraw(data[camera]['data'], name=camera)
            logging.warning(f'Hardcoding EXPTIME=60s (was {self.exptime:.1f}s).')
            self.exptime = 60
            camera_flux /= self.exptime
            camera_dflux /= self.exptime
            # Prepare the auxiliary data saved to the ETC json file.
            each_flux[i] = camera_flux
            each_dflux[i] = camera_dflux
            each_ndrop[i] = self.SKY.ndrop
            logging.info(f'{camera}[{fnum}] flux = {camera_flux:.2f} +/- {camera_dflux:.2f} exptime={self.exptime:.1f}s')
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
        self.skylevel = flux
        # Record this measurement.
        mjd_start, mjd_stop = self.get_mjd_range(mjd_obs, exptime, f'sky[{fnum}]')
        self.sky_measurements.add(
            mjd_start, mjd_stop, flux, dflux,
            aux_data=(each_flux, each_dflux, each_ndrop))
        # Update our accumulated background if the shutter is open.
        if self.accum.shutter_is_open:
            mjd_now = desietc.util.date_to_mjd(timestamp, utc_offset=0)
            self.accum.update('SKY', mjd_stop, mjd_now)
        # Profile timing.
        elapsed = time.time() - start
        logging.info(f'Sky frame processing took {elapsed:.2f}s for {ncamera} cameras.')
        return True

    def get_mjd_range(self, mjd_obs, exptime, source, max_jitter=5):
        mjd_obs = np.asarray(mjd_obs)
        mjd_obs_all = np.nanmean(mjd_obs)
        if np.any(np.abs(mjd_obs - mjd_obs_all) * self.SECS_PER_DAY > max_jitter):
            logging.warning(f'MJD_OBS jitter exceeds {max_jitter}s for {source}: {mjd_obs}.')
            mjd_obs_all = np.nanmedian(mjd_obs)
        exptime = np.asarray(exptime)
        exptime_all = np.nanmean(exptime)
        if np.any(np.abs(exptime - exptime_all) > max_jitter):
            logging.warning(f'EXPTIME jitter exceeds {max_jitter}s for {source}: {exptime}')
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
            if np.all(data['data'] == 0):
                logging.warning(f'Ignoring {source} raw data that is all zeros.')
                return False
            thisGFA.setraw(data['data'], name=camera)
        except ValueError as e:
            logging.error(f'Failed to process {source} raw data: {e}')
            return False
        thisGFA.data -= thisGFA.get_dark_current(ccdtemp, self.exptime)
        # Flag this camera if it appears to have excessive noise
        if thisGFA.nbad_overscan >= self.nbad_threshold:
            if camera not in self.noisy_gfa:
                logging.warning(f'{camera} has excessive noise in {source}.')
                self.noisy_gfa.add(camera)
        return True

    def start_exposure(self, timestamp, expid, req_efftime, sbprof, max_exposure_time, cosmics_split_time,
                       maxsplit, warning_time):
        """Start a new exposure using parameters:

        Parameters
        ----------
        timestamp : datetime.datetime
            UTC timestamp for when the exposure sequence started.
        expid : int
            The exposure id reserved for the first shutter opening of this tile.
        req_efftime : float
            The requested target value of the effective exposure time in seconds.
        sbprof : string
            The surface brightness profile to use for FFRAC calculations. Must be one of
            PSF, ELG, BGS, FLT.
        max_exposure_time : float
            The maximum cummulative exposure time in seconds to allow for this tile,
            summed over all cosmic splits.
        cosmics_split_time : float
            The maximum exposure time in seconds for a single exposure.
        maxsplit : int
            The maximum number of exposures reserved by ICS for this tile.
        warning_time : float
            Warn when a stop or split is expected within this interval in seconds.
        """
        max_hours = 6
        if max_exposure_time <= 0 or max_exposure_time > max_hours * 3600:
            logging.warning(f'max_exposure_time={max_exposure_time} looks fishy: using {max_hours} hours.')
            max_exposure_time = max_hours * 3600
        if sbprof not in ('PSF', 'ELG', 'BGS', 'FLT'):
            logging.error(f'Got invalid sbprof "{sbprof}" so defaulting to "ELG".')
            sbprof = 'ELG'
        self.exptag = str(expid).zfill(8)
        self.night = desietc.util.mjd_to_night(desietc.util.date_to_mjd(timestamp, utc_offset=0))
        self.exp_data = dict(
            expid=expid,
            night=self.night,
            req_efftime=req_efftime,
            sbprof=sbprof,
            max_exposure_time=max_exposure_time,
            cosmics_split_time=cosmics_split_time,
            maxsplit=maxsplit,
            warning_time=warning_time,
        )
        logging.info(f'Start {self.night}/{self.exptag} at {timestamp} with req_efftime={req_efftime:.1f}s, sbprof={sbprof}, '
                     + f'max_exposure_time={max_exposure_time:.1f}s, cosmics_split_time={cosmics_split_time:.1f}s, '
                     + f'maxsplit={maxsplit}, warning_time={warning_time:.1f}s.')
        # Initialize accumulation for the upcoming sequence of cosmic splits.
        self.accum.setup(
            req_efftime, max_exposure_time, cosmics_split_time, maxsplit, warning_time, rdnoise_1ks=0.40)

    def open_shutter(self, expid, timestamp, splittable, max_shutter_time):
        """Record the shutter opening.

        Parameters
        ----------
        expid : int
            The exposure id for this shutter opening.
        timestamp : datetime.datetime
            The UTC timestamp for this shutter opening.
        splittable : bool
            True if this exposure can be split for cosmics.
        max_shutter_time : float
            The maximum time in seconds that the spectrograph shutters will remain open.
        """
        # Lookup the MW transparency to use, or default to 1.
        MW_transp = self.fassign_data.get('MW_transp', 1.)
        # Start accumulating.
        if not self.accum.open(timestamp, splittable, max_shutter_time, MW_transp):
            return
        # Save the current expid and timestamp.
        self.exp_data['expid'] = expid
        self.exptag = str(expid).zfill(8)
        self.exp_data['open'] = desietc.util.date_to_mjd(timestamp, utc_offset=0)
        self.exp_data['max_shutter_time'] = max_shutter_time
        self.exp_data['splittable'] = splittable
        self.total_desi_count += 1
        logging.info(f'Shutter open[{self.accum.nopen},{self.total_desi_count}] for expid {expid} at {timestamp}.')

    def close_shutter(self, timestamp):
        """
        """
        if not self.accum.close(timestamp):
            return
        self.exp_data['close'] = desietc.util.date_to_mjd(timestamp, utc_offset=0)
        # Use float32 values so they are rounded in the json output.
        self.exp_data['efftime'] = np.float32(self.accum.efftime)
        self.exp_data['realtime'] = np.float32(self.accum.realtime)
        self.exp_data['signal'] = np.float32(self.accum.signal)
        self.exp_data['background'] = np.float32(self.accum.background)
        for aux_name in ('transp_obs', 'transp_zenith', 'ffrac_psf', 'ffrac_elg', 'ffrac_bgs', 'thru_psf'):
            self.exp_data[aux_name] = np.float32(self.accum.aux_mean[aux_name])
        logging.info(f'Shutter close[{self.accum.nclose}] at {timestamp} after {self.accum.realtime:.1f}s ' +
            f'with actual teff={self.accum.efftime:.1f}s')

    def stop_exposure(self, timestamp):
        """
        """
        logging.info(f'Stop {self.exptag} at {timestamp}')

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
                logging.warning(f'Fiberassign file is missing header keyword {key}.')
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
            logging.warning(f'Fiberassignment for tile {tileid} only has {ntarget} targets.')
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
        path = pathlib.Path(path)
        logging.info(f'Saving ETC outputs for {self.exptag} to {path}')
        if not path.exists():
            logging.error(f'Non-existent path: {path}.')
            return
        # Save all measurements recorded so far, starting 5 minutes before the shutter opens.
        mjd = self.exp_data.get('open') - 5 * 60 / self.SECS_PER_DAY
        # Build a data structure to save via json.
        try:
            save = dict(
                desietc=self.git or 'unknown',
                expinfo=self.exp_data,
                fassign=self.fassign_data,
                acquisition=self.acquisition_data,
                guide_stars=self.guide_stars,
                shutter=dict(
                    open=self.accum.shutter_open,
                    close=self.accum.shutter_close,
                    teff=np.float32(self.accum.shutter_teff),
                    treal=np.float32(self.accum.shutter_treal),
                ),
                thru=self.thru_measurements.save(mjd),
                sky=self.sky_measurements.save(mjd),
                accum=self.accum.transcript[:self.accum.ntranscript],
            )
        except Exception as e:
            logging.error(f'save_exposure: error building output dict: {e}')
            raise e
            save = {}
        # Encode numpy types using python built-in types for serialization.
        try:
            fname = path / f'etc-{self.exptag}.json'
            with open(fname, 'w') as f:
                json.dump(save, f, cls=desietc.util.NumpyEncoder)
                logging.info(f'Wrote {fname} for {self.exptag}')
        except Exception as e:
            logging.error(f'save_exposure: error saving json: {e}')
