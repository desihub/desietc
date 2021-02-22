"""Run ETC analysis offline from exposure FITS files.

Expects files to be organized in directories YYYYMMDD/EXPID/
under --inpath.  Reads the following files, when available:

 - sky-EXPID.fits.fz : SKYCAM raw data
 - guide-EXPID.fits.fz : In-focus GFA raw acq & guider frames, PlateMaker guide stars
 - centroids-EXPID.json : Results of online guider analysis
 - gfa-EXPID.fits.fz : Raw data for all GFA cameras

For local tests use:

etcreplay --debug --gfa-calib ~/Data/DESI/ETC/GFA_calib.fits --sky-calib  ~/Data/DESI/ETC/SKY_calib.fits --inpath ~/Data/DESI --outpath ~/Data/DESI/ETC --night 20201218 --expid 68630 --overwrite --parallel

Requires that matplotlib is installed, in addition to the desietc
dependencies.
"""
import os
import sys
import time
import pdb
import traceback
import argparse
import warnings
import re
import pathlib
import logging

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import desietc.etc
import desietc.offline


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
    args.inpath = pathlib.Path(args.inpath)
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
        args.checkpath = pathlib.Path(args.checkpath)
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
    args.outpath = pathlib.Path(args.outpath)
    if not args.outpath.exists():
        print('Non-existant output path: {0}'.format(args.outpath))
        sys.exit(-2)
    # Create the night output path if necessary.
    args.outpath = args.outpath / str(args.night)
    args.outpath.mkdir(exist_ok=True)
    logging.info('Output path is {0}'.format(args.outpath))

    # Locate the GFA calibration data.
    if args.gfa_calib is None:
        if host == 'NERSC':
            args.gfa_calib = '/global/cfs/cdirs/desi/cmx/gfa/calib/GFA_calib.fits'
        elif host == 'DOS':
            # Should use a more permanent path than this which is synched via svn.
            args.gfa_calib = '/data/desiobserver/gfadiq/GFA_calib.fits'
        else:
            print('No GFA calibration data path specified with --gfa-calib.')
            sys.exit(-1)
    args.gfa_calib = pathlib.Path(args.gfa_calib)
    if not args.gfa_calib.exists():
        print('Non-existant GFA calibration path: {0}'.format(args.gfa_calib))
        sys.exit(-2)

    # Locate the SKY calibration data.
    if args.sky_calib is None:
        if host == 'NERSC':
            args.sky_calib = '/global/cfs/cdirs/desi/cmx/sky/calib/SKY_calib.fits'
        elif host == 'DOS':
            # Should use a more permanent path than this which is synched via svn.
            args.sky_calib = '/data/desiobserver/gfadiq/SKY_calib.fits'
        else:
            print('No SKY calibration data path specified with --sky-calib.')
            sys.exit(-1)
    args.sky_calib = pathlib.Path(args.sky_calib)
    if not args.sky_calib.exists():
        print('Non-existant SKY calibration path: {0}'.format(args.sky_calib))
        sys.exit(-2)

    # Initialize the global ETC algorithm.
    ETC = desietc.etc.ETCAlgorithm(
        args.sky_calib, args.gfa_calib, args.psf_pixels, args.max_dither, args.num_dither,
        args.Ebv_coef, args.nbad_threshold, args.nll_threshold, args.grid_resolution, args.parallel)

    # Enable GMM debug messages if requested.
    if args.debug and args.gmm_debug:
        if args.parallel:
            print('Ignoring --gmm-debug with --parallel')
        else:
            ETC.GMM.set_debug(True)

    def process(expid):
        nonlocal nprocessed
        success = desietc.offline.replay_exposure(
            ETC, nightpath, expid,
            outpath=args.outpath, dry_run=args.dry_run, overwrite=args.overwrite,
            only_complete=args.only_complete)
        if success:
            nprocessed += 1

    nprocessed = 0

    # Wrap the processing of exposures to ensure that ETC.shutdown is always called.
    try:
        ETC.start()
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
                for expid in range(start, stop):
                    process(expid)

        expid_pattern = re.compile('^[0-9]{8}$')
        get_exposures = lambda: set([
            int(p.name) for p in nightpath.glob('????????') if expid_pattern.match(p.name)])

        if args.batch or args.watch:
            # Find the existing exposures on this night.
            existing = get_exposures()
            if args.batch:
                for expid in sorted(existing):
                    process(expid)
            if args.watch:
                logging.info('Watching for new exposures...hit ^C to exit')
                try:
                    while True:
                        time.sleep(args.watch_interval)
                        newexp = get_exposures() - existing
                        for expid in sorted(newexp):
                            process(expid)
                        existing |= newexp
                except KeyboardInterrupt:
                    logging.info('Bye.')
                    pass
    finally:
        ETC.shutdown()

    logging.info(f'Processed {nprocessed} exposures.')


def main():
    # https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(
        description='Run ETC analysis offline from exposure FITS files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--debug', action='store_true',
        help='provide verbose and debugging output')
    parser.add_argument('--traceback', action='store_true',
        help='print traceback and enter debugger after an exception')
    parser.add_argument('--logpath', type=str, metavar='PATH',
        help='Path where logging output should be written')
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
    parser.add_argument('--psf-pixels', type=int, default=25,
        help='Size of PSF stamp to use for guide star measurements')
    parser.add_argument('--max-dither', type=float, default=7,
        help='Maximum dither in pixels to use for guide star fits')
    parser.add_argument('--num-dither', type=int, default=1200,
        help='Number of dithers to use between (-max,+max)')
    parser.add_argument('--Ebv-coef', type=float, default=1,
        help='Coefficient to use for MW extinction')
    parser.add_argument('--nbad-threshold', type=int, default=100,
        help='Maximum allowed bad overscan pixels before warning')
    parser.add_argument('--nll-threshold', type=float, default=10,
        help='Maximum allowed GMM fit NLL value before warning')
    parser.add_argument('--grid-resolution', type=float, default=0.5,
        help='Resolution of ETC calculations in seconds')
    parser.add_argument('--gmm-debug', action='store_true',
        help='Generate debug log messages during GMM.fit')
    parser.add_argument('--dry-run', action='store_true',
        help='Check FITS file names and headers with no ETC processing')
    parser.add_argument('--inpath', type=str, metavar='PATH',
        help='Path where raw data is organized under YYYYMMDD directories')
    parser.add_argument('--checkpath', type=str, metavar='PATH',
        help='Optional path where links are created to indicate a complete exposure')
    parser.add_argument('--outpath', type=str, metavar='PATH',
        help='Path where outputs willl be organized under YYYYMMDD directories')
    parser.add_argument('--overwrite', action='store_true',
        help='Overwrite existing outputs')
    parser.add_argument('--only-complete', action='store_true',
        help='Only process exposures with all expected FITS files')
    parser.add_argument('--gfa-calib', type=str, metavar='PATH',
        help='Path to GFA calibration FITS file to use')
    parser.add_argument('--sky-calib', type=str, metavar='PATH',
        help='Path to SKYCAM calibration FITS file to use')
    parser.add_argument('--parallel', action='store_true',
        help='Process GFA cameras in parallel')
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
    # Silence matplotlib debug logging.
    logging.getLogger('matplotlib.font_manager').level = max(level, logging.INFO)
    logging.getLogger('matplotlib.ticker').disabled = max(level, logging.INFO)

    try:
        retval = etcoffline(args)
        sys.exit(retval)
    except Exception as e:
        if args.traceback:
            # https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            print(e)
            sys.exit(-1)


if __name__ == '__main__':
    main()
