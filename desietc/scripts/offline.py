"""Run ETC analysis offline from exposure FITS files.
"""
import os
import sys
import time
import argparse
import warnings
import glob
import logging
import multiprocessing

from pathlib import Path

import numpy as np

try:
    from PIL import Image
    img_format = 'jpg'
except ImportError:
    img_format = 'png'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fitsio

from desietcimg.gfa import *
from desietcimg.gmm import *
from desietcimg.plot import *
from desietcimg.util import *
from desietcimg.sky import SkyCamera

# Globals shared by process_one below.
GFA = None
GMM = None
SKY = None
rawbuf = np.empty((1, 1032, 2248), np.uint32)


def process_guide_sequence(stars, exptime, maxdither=3, ndither=31, zoomdither=2,
                           zeropoint=27.06, fiber_diam_um=107, pixel_size_um=15):
    """
    """
    global GFA, GMM
    T, WT = GFA.psf_stack
    stampsize = GMM.shape[0]
    assert stampsize % 2 == 1, 'stampsize must be odd'
    halfsize = stampsize // 2
    nstars = len(stars)
    if nstars <= 0:
        logging.warning('No guide stars available')
        return
    nexp, ny, nx = GFA.data.shape
    if nexp <= 1:
        logging.warning('No guide frames to process')
        return
    logging.info('Processing {0} guide exposures with {1} stars'.format(nexp, nstars))
    assert exptime.shape == (nexp,)
    nexp -= 1 # do not include the acquisition image.
    exptime = exptime[1:]
    nT = len(T)
    assert T.shape == WT.shape == (nT, nT)
    # Trim the PSF to the stamp size to use for fitting.
    assert stampsize <= nT
    ntrim = (nT - stampsize) // 2
    S = slice(ntrim, ntrim + stampsize)
    T, WT = T[S, S], WT[S, S]
    # Fit the PSF to a Gaussian mixture model.
    gmm_params = GMM.fit(T, WT, maxgauss=3)
    if gmm_params is None:
        logging.error('Unable to fit PSF model')
        return
    # Prepare dithered fits.
    xdither, ydither = diskgrid(ndither, maxdither, alpha=2)
    dithered = GMM.dither(gmm_params, xdither, ydither)
    # Initialize fiber templates for each guide star target centroid.
    max_rsq = (0.5 * fiber_diam_um / pixel_size_um) ** 2
    profile = lambda x, y: 1.0 * (x ** 2 + y ** 2 < max_rsq)
    # Initialize stacked results.
    Dsum = np.zeros((nstars, stampsize, stampsize))
    WDsum = np.zeros((nstars, stampsize, stampsize))
    Msum = np.zeros((nstars, stampsize, stampsize))
    fit_params = np.empty((nstars, nexp, 5))
    for istar in range(nstars):
        # Lookup the platemaker target coordinates and magnitude for this guide star.
        x0, y0, rmag = stars[istar]
        # Convert from PlateMaker indexing convention to (0,0) centered in bottom-left pixel.
        y0 -= 0.5
        x0 -= 0.5
        if rmag == 0:
            # rmag == 0 indicates that we have no magnitude info.
            nelec_pred = 0
        else:
            # Convert rflux to predicted detected electrons with
            # perfect atmospheric transmission and fiber acceptance.
            nelec_pred = 10 ** (-(rmag - zeropoint) / 2.5) * exptime
        # Build a stamp centered on these coordinates.
        iy, ix = np.round(y0).astype(int), np.round(x0).astype(int)
        ylo, yhi = iy - halfsize, iy + halfsize + 1
        xlo, xhi = ix - halfsize, ix + halfsize + 1
        if ylo < 0 or yhi > ny or xlo < 0 or xhi > nx:
            logging.info('Skipping stamp too close to border at ({0},{1})'.format(x0, y0))
            continue
        SY, SX = slice(ylo, yhi), slice(xlo, xhi)
        # Prepare a fiber template centered on the target position.
        fiber_dx, fiber_dy = x0 - ix, y0 - iy
        fiber = make_template(stampsize, profile, dx=fiber_dx, dy=fiber_dy, normalized=False)
        # Do not include the acquisition image.
        Dframes = GFA.data[1:, SY, SX]
        WDframes = GFA.ivar[1:, SY, SX]
        for iexp in range(nexp):
            D, WD = Dframes[iexp].copy(), WDframes[iexp].copy()
            # Estimate centroid, flux and constant background.
            dx, dy, flux, bg, nll, best_fit = GMM.fit_dithered(xdither, ydither, dithered, D, WD)
            if iexp == 0 and np.all(nelec_pred == 0):
                if exptime[0] > 0:
                    # Use the first measured flux as the reference value for transparency.
                    nelec_pred = flux * exptime / exptime[0]
                else:
                    if istar == 0:
                        logging.warning(
                            'First guide frame has EXPTIME=0 so unable to calculate transparency')
                    nelec_pred = flux * np.ones_like(exptime)
            # Calculate the flux fraction within the fiber aperture using the best-fit model.
            fiberfrac = np.sum(fiber * best_fit)
            # Accumulate this exposure.
            Dsum[istar] += D * WD
            WDsum[istar] += WD
            # Coadd per-exposure best-fit models for star.
            Msum[istar] += flux * best_fit
            # Record the best-fit parameters for this guide star.
            # Centroids are relative to the nominal fiber center (fiber_dx, fiber_dy)
            # which will be within 1 pixel of the stamp center.
            fit_params[istar, iexp] = (
                dx - fiber_dx, dy - fiber_dy, flux / nelec_pred[iexp], fiberfrac, nll)
        Dsum[istar] = np.divide(
            Dsum[istar], WDsum[istar], out=np.zeros_like(Dsum[istar]), where=WDsum[istar] > 0)
    return Dsum, WDsum, Msum, fit_params, gmm_params


def process_one(inpath, night, expid, guiding, camera, mjdobs, exptime, ccdtemp, framepath,
                stars, maxdither, ndither):
    """Process a single camera of a single exposure.

    Returns
        tuple or None
            Returns None in case of an error, or a tuple (initial, frames) of results from processing
            the initial image and any subsequent guiding frames (when giuding is True and stars is
            not None).
    """
    global GFA
    with fitsio.FITS(str(inpath), mode='r') as hdus:
        if camera not in hdus:
            logging.error('Missing HDU {0}'.format(camera))
            return None
        logging.info('Processing {0}'.format(camera))
        try:
            if guiding and stars is not None:
                # Process all exposures.
                raw = hdus[camera][:, :, :]
            else:
                if guiding:
                    # Only process the initial acquisition image of a guideing sequence.
                    raw = rawbuf[0] = hdus[camera][0, :, :]
                    exptime, ccdtemp = exptime[0], ccdtemp[0]
                else:
                    # There is only one exposure to process.
                    raw = rawbuf[0] = hdus[camera][:, :]
        except ValueError as e:
            logging.error('Failed to read raw data:')
            logging.error(e)
            return None
        try:
            GFA.setraw(raw, name=camera)
        except ValueError as e:
            logging.error('Failed to process raw data:')
            logging.error(e)
            return None
        GFA.data -= GFA.get_dark_current(ccdtemp, exptime)
        if camera.startswith('GUIDE'):
            GFA.get_psfs(iexp=0)
            stamps = GFA.psfs
            if GFA.psf_stack[0] is not None and stars is not None:
                stars_result = process_guide_sequence(stars, exptime, maxdither=maxdither, ndither=ndither)
                if stars_result is not None and framepath is not None:
                    Dsum, WDsum, Msum, fit_params, gmm_params = stars_result
                    try:
                        fig, ax = plot_guide_stars(Dsum, WDsum, Msum, fit_params, night, expid, camera)
                        fname = framepath / 'guide_{0}_{1}.{2}'.format(camera, expid, img_format)
                        plt.savefig(fname, pil_kwargs=dict(quality=80))
                        plt.close(fig)
                        logging.info('Wrote {0}'.format(fname))
                    except Exception as e:
                        logging.error(f'Error plotting guide results: {e}.')
                    # Write a CSV file of per-frame median results for this GFA.
                    med_params = np.nanmedian(fit_params, axis=0)
                    nexp, npar = med_params.shape
                    assert nexp == len(exptime) - 1
                    fname = framepath / 'guide_{0}_{1}.csv'.format(camera, expid)
                    with open(fname, 'w') as out:
                        for iexp in range(nexp):
                            mjd = mjdobs[iexp + 1] + 0.5 * exptime[iexp + 1] / 86400
                            print(mjd, file=out, end='')
                            for ipar in range(npar):
                                print(',{0}'.format(med_params[iexp, ipar]), file=out, end='')
                            print('', file=out)
                        logging.info('Wrote {0}'.format(fname))
                result = GFA.psf_stack, stars_result
            else:
                result = GFA.psf_stack, None
        else:
            GFA.get_donuts(iexp=0)
            stamps = GFA.donuts[0] + GFA.donuts[1]
            result = GFA.donut_stack, None
        if framepath is not None:
            if not np.isscalar(exptime):
                # Use values for the acquisition image of a guide sequence.
                exptime, ccdtemp = exptime[0], ccdtemp[0]
            label = '{0} {1} {2:.1f}s {3:.1f}C'.format(night, expid, exptime, ccdtemp)
            fig, ax = plot_data(GFA.data[0], GFA.ivar[0], downsampling=2, label=label,
                                stamps=stamps, colorhist=True)
            plt.savefig(framepath / 'frame_{0}_{1}.{2}'.format(camera, expid, img_format),
                        pil_kwargs=dict(quality=80))
            plt.close(fig)
        return result


def process_sky(inpath, outpath):
    try:
        with fitsio.FITS(str(inpath)) as hdus:
            hdr = hdus[0].read_header()
            if 'NIGHT' not in hdr:
                logging.warning('Missing NIGHT from hdu SKY')
                # Hack until this gets fixed.
                data = hdus['SKYCAM0T'].read()
                night = str(data['NIGHT'][0])
            else:
                night = str(hdr['NIGHT'])
            program = hdr.get('PROGRAM', 'No PROGRAM specified')
            expid = '{0:08d}'.format(hdr['EXPID'])
            if 'EXPTIME' in hdr:
                exptime = hdr['EXPTIME']
            elif 'SKYTIME' in hdr:
                exptime = hdr['SKYTIME']
            else:
                logging.error('Missing EXPTIME or SKYTIME from HDU0')
                return
            if exptime is None or float(exptime) <= 0:
                logging.error(f'Invalid exptime in {inpath}')
                return
            nframes = hdr['FRAMES']
            logging.info('Processing {0} x {1:.1f}s SKYCAM frames from {2}'.format(nframes, exptime, inpath))
            outpath = outpath / night / expid
            outpath.mkdir(parents=True, exist_ok=True)
            fout = open(outpath / 'sky_{0}.csv'.format(expid), 'w')
            f, df = np.zeros((2, nframes)), np.zeros((2, nframes))
            for j, camera in enumerate(('SKYCAM0', 'SKYCAM1')):
                if not camera in hdus:
                    logging.warn('Missing {0} HDU'.format(camera))
                    continue
                if camera + 'T' in hdus:
                    meta = hdus[camera + 'T'].read()
                    mjd = meta['MJD-OBS'] + 0.5 * exptime / 86400
                else:
                    logging.warn('Missing {0} HDU so MJD=0'.format(camera+'T'))
                    mjd = np.zeros(nframes)
                framedata = hdus[camera].read()
                if len(framedata) != nframes:
                    logging.error('Data size does not match header FRAMES')
                for k, data in enumerate(framedata):
                    f[j, k], df[j, k] = SKY.setraw(data, name=camera)
                    print('{0},{1},{2},{3:.3f},{4:.3f},{5:.3f},{6}'.format(
                        camera, k, mjd[k], f[j, k], df[j, k], SKY.chisq, SKY.ndata - SKY.ndrop), file=fout)
            fout.close()
    except OSError as e:
        logging.error(f'Error reading {inpath}')
        return
    # Normalize to the exposure time.
    f /= exptime
    df /= exptime
    # Print the mean sky levels to the log.
    logging.info('Mean SKYCAM0 = {0:.3f} SKYCAM1 = {1:.3f} [arb units]'.format(*f.mean(axis=1)))
    # Plot the results.
    fig = plt.figure(figsize=(9, 5))
    t = np.arange(1, nframes + 1)
    plt.errorbar(t - 0.1, f[0], df[0], label='SKYCAM0', fmt='o')
    plt.errorbar(t + 0.1, f[1], df[1], label='SKYCAM1', fmt='o')
    plt.legend(ncol=2, title=program)
    plt.xlabel('{0} {1} Frame #'.format(night, expid))
    plt.ylabel('Sky Level / ({0:.1f}s) [arb units]'.format(exptime))
    plt.tight_layout()
    plt.savefig(outpath / 'sky_{0}.png'.format(expid))
    plt.close(fig)


def process(inpath, args, pool=None, pool_timeout=300):
    """Process a single GFA exposure.
    """
    global GFA
    if not inpath.exists():
        logging.error('Non-existant path: {0}'.format(inpath))
        return
    if args.dry_run:
        print(f'{inpath}')
        return
    # Is this a skycam exposure?
    if inpath.name.startswith('sky'):
        process_sky(inpath, args.outpath)
        return
    # Is this a guiding exposure?
    guiding = inpath.name.startswith('guide')
    # Lookup the NIGHT, EXPID, EXPTIME from the primary header.
    hdr_ext = 'GUIDER' if guiding else 'GFA'
    hdr = fitsio.read_header(str(inpath), ext=hdr_ext)
    for k in 'NIGHT', 'EXPID', 'EXPTIME':
        if k not in hdr:
            logging.info('Skipping exposure with missing {0}: {1}'.format(k, inpath))
            return
    cameras_key = 'ACQCAM' if guiding else 'IMAGECAM'
    cameras = hdr.get(cameras_key)
    if cameras is None:
        logging.info('Skipping exposure with missing {0}/{1}: {2}'
                     .format(hdr_ext, cameras_key, inpath))
        return
    if cameras == 'GUIDE0,FOCUS1,GUIDE2,GUIDE3,FOCUS4,GUIDE5,FOCUS6,GUIDE7,GUIDE8,FOCUS':
        # The full lists of 10 cameras exceeds the 71-char max length in the FITS standard.
        # https://heasarc.gsfc.nasa.gov/docs/software/fitsio/c/c_user/node20.html
        logging.warning('Patching {0}/{1} keyword value.'.format(hdr_ext, cameras_key))
        cameras = 'GUIDE0,FOCUS1,GUIDE2,GUIDE3,FOCUS4,GUIDE5,FOCUS6,GUIDE7,GUIDE8,FOCUS9'
    if cameras == 'G0,G2,G3,G5,G7,G8,F1,F4,F6,F9':
        # Data after the Nov-2020 restart needs this for GFA FITS files.
        cameras = 'GUIDE0,FOCUS1,GUIDE2,GUIDE3,FOCUS4,GUIDE5,FOCUS6,GUIDE7,GUIDE8,FOCUS9'
    night = str(hdr['NIGHT'])
    expid = '{0:08d}'.format(hdr['EXPID'])
    if hdr['EXPTIME'] == 0:
        logging.info('Skipping zero EXPTIME: {0}/{1}'.format(night, expid))
        return
    stars_expected = {}
    if guiding and args.guide_stars:
        assert GMM is not None, 'GMM not initialized.'
        PlateMaker, GuiderExpected = None, None
        try:
            GuiderExpected, _, _ = load_guider_centroids(inpath.parent, expid)
        except (ValueError, KeyError):
            logging.warning('Guider centroids json file not readable.')
        try:
            PlateMaker = fitsio.read(str(inpath), ext='PMGSTARS')
        except IOError as e:
            logging.warning('PMGSTARS extension not found.')
        if PlateMaker is not None:
            # Use PlateMaker (row, col) for expected positions of each guide star.
            for camera in np.unique(PlateMaker['GFA_LOC']):
                stars = PlateMaker[PlateMaker['GFA_LOC'] == camera]
                stars_expected[camera] = np.array((stars['COL'], stars['ROW'], stars['MAG'])).T
        elif GuiderExpected is not None:
            # Fall back to guider centroids.  I assume the JSON files uses the same coordinate
            # convention as PlateMaker since it copies the PlateMaker values when both are present.
            for camera in GuiderExpected:
                nstars = len(GuiderExpected[camera])
                if nstars > 0:
                    stars_expected[camera] = np.array(
                        (GuiderExpected[camera][:, 0], GuiderExpected[camera][:, 1], np.zeros(nstars))).T
        else:
            logging.warning(f'No guide stars available for {night}/{expid}.')
    # Prepare the output path.
    outpath = args.outpath / night / expid
    outpath.mkdir(parents=True, exist_ok=True)
    # Are there already existing outputs?
    fitspath = outpath / 'gfaetc_{0}.fits'.format(expid)
    if fitspath.exists() and not args.overwrite:
        logging.info('Will not overwrite outputs in {0}'.format(outpath))
        return
    # Process each camera in the input.
    logging.info('Processing {0} from {1}'.format(cameras, inpath))
    results = {}
    framepath = outpath if args.save_frames else None
    only_cameras = [] if args.only_cameras is None else args.only_cameras.split(',')
    for camera in cameras.split(','):
        if camera not in GFA.gfa_names:
            logging.warning('Ignoring invalid camera name in header: "{0}".'.format(camera))
            continue
        if len(only_cameras) > 0 and camera not in only_cameras:
            logging.info('Skipping {0} because only-cameras is "{1}".'.format(camera, args.only_cameras))
            continue
        if guiding and camera.startswith('FOCUS'):
            # Guiding exposures do not record FOCUS data.
            continue
        # Fetch this camera's CCD temperatures and exposure times.
        if guiding:
            info = fitsio.read(str(inpath), ext=camera + 'T', columns=('MJD-OBS', 'EXPTIME', 'GCCDTEMP'))
            # stars might be None if --guide-stars arg not present or no guide stars available.
            stars = stars_expected.get(camera, None)
        else:
            info = fitsio.read_header(str(inpath), ext=camera)
            stars = None
        mjdobs = info['MJD-OBS']
        exptime = info['EXPTIME']
        ccdtemp = info['GCCDTEMP']
        process_args = (inpath, night, expid, guiding, camera, mjdobs, exptime, ccdtemp, framepath,
                        stars, args.max_dither, args.num_dither)
        if pool is None:
            result = process_one(*process_args)
            if result is None:
                logging.error('Error processing HDU {0}'.format(camera))
            else:
                results[camera] = result
        else:
            results[camera] = pool.apply_async(process_one, process_args)
    if pool:
        # Collect the pooled results.
        for camera in results:
            try:
                result = results[camera].get(timeout=pool_timeout)
                if result is None:
                    logging.error('Error pool processing HDU {0}'.format(camera))
                else:
                    results[camera] = result
            except TimeoutError:
                logging.error('Timeout waiting for {0} pool result'.format(camera))
                del results[camera]
    # Save the output FITS file.
    with fitsio.FITS(str(fitspath), 'rw', clobber=True) as hdus:
        meta = {k: hdr.get(k) for k in (
            'NIGHT', 'EXPID', 'MJD-OBS', 'EXPTIME', 'PROGRAM', 'HEXPOS', 'TRUSTEMP',
            'ADC1PHI', 'ADC2PHI', 'MOUNTHA', 'MOUNTAZ', 'MOUNTEL', 'MOUNTDEC')}
        hdus.write(np.zeros(1), header=meta)
        for camera in results:
            # Retrieve the result of processing the initial image and any subsequent guide frames.
            initial, frames = results[camera]
            if camera.startswith('GUIDE'):
                hdus.write(np.stack(initial).astype(np.float32), extname=camera)
            else:
                L, R = initial
                if L is not None:
                    hdus.write(np.stack(L).astype(np.float32), extname=camera + 'L')
                if R is not None:
                    hdus.write(np.stack(R).astype(np.float32), extname=camera + 'R')
            if frames is not None:
                (Dsum, WDsum, Msum, fit_params, gmm_params) = frames
                hdus.write(np.stack((Dsum, WDsum, Msum)).astype(np.float32), extname=camera + 'G')
                hdus.write(fit_params.astype(np.float32), extname=camera + 'P')
                hdus.write(gmm_params.astype(np.float32), extname=camera + 'M')
    try:
        # Produce a summary plot of the delivered image quality measured from the first image.
        fig = plot_image_quality({camera: result[0] for camera, result in results.items()}, meta)
        # Save the summary plot.
        figpath = outpath / 'gfadiq_{0}.png'.format(expid)
        plt.savefig(figpath)
        plt.close(fig)
        logging.info('Wrote {0}'.format(figpath))
    except Exception as e:
        logging.warning('Failed to create image quality plot.')

def get_gfa_exposures(inpath, checkpath, night, expstart=None, expstop=None, sky=False, gfa=True):
    """Return a list of existing paths to completed GFA exposures for night.
    """
    assert sky or gfa
    paths = []
    # Has a directory for this night been created yet?
    checknight = checkpath / str(night)
    innight = inpath / str(night)
    if not checknight.exists() or not innight.exists():
        return paths
    patterns = ['gfa-{0}.fits.fz', 'guide-{0}.fits.fz'] if gfa else []
    if sky:
        patterns.append('sky-{0}.fits.fz')
    # Loop over all exposure paths under this night.
    for exppath in checknight.glob('????????'):
        expid = str(exppath)[-8:]
        expnum = int(expid)
        if expstart is not None and expnum < expstart:
            continue
        if expstop is not None and expnum >= expstop:
            continue
        for pattern in patterns:
            path = innight / expid / pattern.format(expid)
            if path.exists():
                paths.append(path)
    return set(paths)


def etcoffline(args):
    # Are we running on a recognized host with special defaults?
    host = None
    if os.getenv('NERSC_HOST') is not None:
        host = 'NERSC'
        logging.info('Detected a NERSC host.')
    elif os.getenv('DOS_HOME') is not None:
        host = 'DOS'

    # Determine where the input raw data is located.
    if args.inpath is None:
        if host == 'NERSC':
            args.inpath = '/global/cfs/cdirs/desi/spectro/data/'
        elif host == 'DOS':
            args.inpath = '/exposures/desi/'
        else:
            print('No input path specified with --inpath.')
            sys.exit(-1)
    args.inpath = Path(args.inpath)
    if not args.inpath.exists():
        print('Non-existant input path: {0}'.format(args.inpath))
        sys.exit(-2)
    logging.info('Input path is {0}'.format(args.inpath))

    # Determine which directory to check for completed exposures.
    if args.checkpath is None:
        if host == 'DOS':
            args.checkpath = '/data/dts/exposures/raw/'
        else:
            args.checkpath = args.inpath
    if args.checkpath != args.inpath:
        args.checkpath = Path(args.checkpath)
        if not args.checkpath.exists():
            print('Non-existant check path: {0}'.format(args.checkpath))
            sys.exit(-2)
        logging.info('Check path is {0}'.format(args.inpath))

    if args.night is None:
        print('Missing required argument: night.')
        sys.exit(-1)
    nightpath = args.inpath / str(args.night)
    if not nightpath.exists():
        print('Non-existant directory for night: {0}'.format(nightpath))

    # Determine where the outputs will go.
    if args.outpath is None:
        print('No output path specified with --outpath.')
        sys.exit(-1)
    args.outpath = Path(args.outpath)
    if not args.outpath.exists():
        print('Non-existant output path: {0}'.format(args.outpath))
        sys.exit(-2)
    logging.info('Output path is {0}'.format(args.outpath))

    # Locate the GFA calibration data.
    if args.calibpath is None:
        if host == 'NERSC':
            args.calibpath = '/global/cfs/cdirs/desi/cmx/gfa/calib/GFA_calib.fits'
        elif host == 'DOS':
            # Should use a more permanent path than this which is synched via svn.
            args.calibpath = '/data/desiobserver/gfadiq/GFA_calib.fits'
        else:
            print('No GFA calibration data path specified with --calibpath.')
            sys.exit(-1)
    args.calibpath = Path(args.calibpath)
    if not args.calibpath.exists():
        print('Non-existant GFA calibration path: {0}'.format(args.calibpath))
        sys.exit(-2)

    args.sky = args.sky or args.sky_only
    if args.sky:
        # Locate the SKY calibration data.
        if args.skycalibpath is None:
            if host == 'NERSC':
                args.skycalibpath = '/global/cfs/cdirs/desi/cmx/sky/calib/SKY_calib.fits'
            elif host == 'DOS':
                # Should use a more permanent path than this which is synched via svn.
                args.skycalibpath = '/data/desiobserver/gfadiq/SKY_calib.fits'
            else:
                print('No SKY calibration data path specified with --skycalibpath.')
                sys.exit(-1)
        args.skycalibpath = Path(args.skycalibpath)
        if not args.skycalibpath.exists():
            print('Non-existant SKY calibration path: {0}'.format(args.skycalibpath))
            sys.exit(-2)

    # Initialize the GFA analysis object.
    global GFA
    GFA = GFACamera(calib_name=args.calibpath)

    if args.guide_stars:
        # Initialize the global guide star Gaussian mixture model.
        global GMM
        psf_grid = np.arange(args.psf_pixels + 1) - args.psf_pixels / 2
        GMM = GMMFit(psf_grid, psf_grid)

    if args.sky:
        global SKY
        SKY = SkyCamera(calib_name=args.skycalibpath)

    if args.npool > 0:
        # Initialize multiprocessing.
        context = multiprocessing.get_context(method='fork')
        if args.npool > context.cpu_count():
            args.npool = context.cpu_count()
            log.warning('Reducing --npool to {0}'.format(args.npool))
        pool = context.Pool(processes=args.npool)
        logging.info('Initialized multiprocessing pool with {0} workers'.format(args.npool))
    else:
        pool = None

    if args.expid is not None:
        exposures = set()
        # Loop over comma-separated tokens.
        for token in args.expid.split(','):
            # Process a token of the form N or N1-N2.
            limits = [int(expid) for expid in token.split('-')]
            if len(limits) == 1:
                start, stop = limits[0], limits[0] + 1
            elif len(limits) == 2:
                start, stop = limits[0], limits[1] + 1
            else:
                print('Invalid --expid (should be N or N1-N2): "{0}"'.format(args.expid))
                sys.exit(-1)
            exposures |= get_gfa_exposures(
                args.inpath, args.checkpath, args.night, start, stop,
                sky=args.sky, gfa=not args.sky_only)
        for path in sorted(exposures):
            process(path, args, pool)
        return

    if args.batch or args.watch:
        # Find the existing exposures on this night.
        existing = get_gfa_exposures(args.inpath, args.checkpath, args.night,
                                     sky=args.sky, gfa=not args.sky_only)
        if args.batch:
            for path in sorted(existing):
                process(path, args, pool)
        if args.watch:
            logging.info('Watching for new exposures...hit ^C to exit')
            try:
                while True:
                    time.sleep(args.watch_interval)
                    newexp = get_gfa_exposures(
                        args.inpath, args.checkpath, args.night,
                        sky=args.sky, gfa=not args.sky_only) - existing
                    for path in sorted(newexp):
                        process(path, args, pool)
                    existing |= newexp
            except KeyboardInterrupt:
                logging.info('Bye.')
                pass


def main():
    # https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(
        description='Run ETC analysis offline from exposure FITS files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--logpath', type=str, metavar='PATH',
        help='Path where logging output should be written')
    parser.add_argument('--debug', action='store_true',
        help='print traceback and enter debugger after an exception')
    parser.add_argument('--night', type=int, metavar='YYYYMMDD',
        help='Night of exposure to process in the format YYYYMMDD')
    parser.add_argument('--expid', type=str, metavar='N',
        help='Exposure(s) to process specified as N or N1-N2 or N1,N2-N3 etc')
    parser.add_argument('--batch', action='store_true',
        help='Process all existing exposures on night')
    parser.add_argument('--watch', action='store_true',
        help='Wait for and process new exposures on night')
    parser.add_argument('--watch-interval', type=float, metavar='T', default=2,
        help='Interval in seconds to check for new exposures with --watch')
    parser.add_argument('--save-frames', action='store_true',
        help='Save images of each GFA frame')
    parser.add_argument('--guide-stars', action='store_true',
        help='Measure guide stars in each frame of any guiding sequences')
    parser.add_argument('--max-dither', type=float, default=7,
        help='Maximum dither in pixels to use for guide star fits')
    parser.add_argument('--num-dither', type=int, default=1200,
        help='Number of dithers to use between (-max,+max)')
    parser.add_argument('--psf-pixels', type=int, default=25,
        help='Size of PSF stamp to use for guide star measurements')
    parser.add_argument('--sky', action='store_true',
        help='Also process sky camera data')
    parser.add_argument('--sky-only', action='store_true',
        help='Only process sky camera data')
    parser.add_argument('--only-cameras', type=str, default=None,
        help='Only analyze data from these comma-separated cameras')
    parser.add_argument('--dry-run', action='store_true',
        help='Print file names only with no further processing')
    parser.add_argument('--overwrite', action='store_true',
        help='Overwrite existing outputs')
    parser.add_argument('--inpath', type=str, metavar='PATH',
        help='Path where raw data is organized under YYYYMMDD directories')
    parser.add_argument('--outpath', type=str, metavar='PATH',
        help='Path where outputs willl be organized under YYYYMMDD directories')
    parser.add_argument('--checkpath', type=str, metavar='PATH',
        help='Optional path where links are created to indicate a complete exposure')
    parser.add_argument('--calibpath', type=str, metavar='PATH',
        help='Path to GFA calibration FITS file to use')
    parser.add_argument('--skycalibpath', type=str, metavar='PATH',
        help='Path to SKYCAM calibration FITS file to use')
    parser.add_argument('--npool', type=int, default=0, metavar='N',
        help='Number of workers in multiprocessing pool')
    args = parser.parse_args()

    # Configure logging.
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(filename=args.logpath, level=level,
        format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y%m%d %H:%M:%S')

    try:
        retval = etcdepth(args)
        sys.exit(retval)
    except Exception as e:
        if args.debug:
            # https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            print(e)
            sys.exit(-1)


if __name__ == '__main__':
    main()
