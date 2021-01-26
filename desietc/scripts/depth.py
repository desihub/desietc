"""Calculate exposure depths using ETC and spectograph pipeline outputs.

Requires that pandas is installed and that either pyyaml and psycopg2
are installed (for a direct db connection), or else that requests is installed
(for an indirect http connection to the db).
"""
import argparse
import logging
import pdb
import traceback
import sys
import os
from pathlib import Path

import numpy as np

import fitsio

import desietc.db
import desietc.spectro


# TODO: propagate actual GFA, SKY exposure times from FITS file


def load_etc_sky(name, exptime):
    """Read the ETC results for a sky camera exposure from a CSV file.
    Return arrays of MJD and relative flux values.
    """
    data = np.loadtxt(
        name, delimiter=',', dtype=
        {'names':('camera','frame','mjd','flux','dflux','chisq','nfiber'),
            'formats':('S7','i4','f8','f4','f4','f4','i4')})
    # Calculate a weighted average of sky camera flux in each frame.
    frames = np.unique(data['frame'])
    nframe = len(frames)
    if not np.all(frames == np.arange(nframe)):
        logging.warning(f'Unexpected sky frames in {name}')
        return None, None
    mjd = np.empty(nframe)
    flux = np.empty(nframe)
    for frame in frames:
        sel = data['frame'] == frame
        # Calculate sky exposure midpoint.
        mjd[frame] = np.mean(data['mjd'][sel]) + 0.5 * exptime / 86400
        ivar = data['dflux'][sel] ** -0.5
        # Calculate ivar-weighted mean relative flux.
        flux[frame] = np.sum(ivar * data['flux'][sel]) / np.sum(ivar) / exptime
    return mjd, flux


def load_etc_gfa(names, exptime):
    assert np.isfinite(exptime)
    mjd, transp, ffrac = [], [], []
    for name in names:
        data = np.loadtxt(name, delimiter=',', dtype=
            {'names':('mjd','dx','dy','transp','ffrac','nll'),
             'formats': ('f8','f4','f4','f4','f4','f4')})
        mjd.append(data['mjd'] + 0.5 * exptime / 86400)
        transp.append(data['transp'])
        ffrac.append(data['ffrac'])
        if not np.all(np.isfinite(data['mjd']) & np.isfinite(data['transp']) & np.isfinite(data['ffrac'])):
            logging.warning(f'Ignoring some NaN values in {name}')
    mjd = np.vstack(mjd)
    transp = np.vstack(transp)
    ffrac = np.vstack(ffrac)
    return np.nanmean(mjd, axis=0), np.nanmedian(transp, axis=0), np.nanmedian(ffrac, axis=0)


def check_exptime(value, label='exptime', default=None):
    """Check for None, <=0 or nan/inf and use a default if specified.
    Logs a warning message when a bad value is found.
    """
    if value is None or value <= 0 or not np.isfinite(value):
        logging.warning(f'Found invalid {label}={value}, using default={default}.')
        value = default
    return value


def etcdepth(args):
    # Check required paths.
    DESIROOT = Path(args.desiroot or os.getenv('DESI_ROOT', '/global/cfs/cdirs/desi'))
    logging.info(f'DESIROOT={DESIROOT}')
    RELEASE = DESIROOT / 'spectro' / 'redux' / args.release
    if not RELEASE.exists():
        raise RuntimeError(f'Non-existent {RELEASE}')
    logging.info(f'RELEASE={RELEASE}')
    ETC = Path(args.etcpath or (DESIROOT / 'spectro' / 'ETC'))
    if not ETC.exists():
        raise RuntimeError(f'Non-existent {ETC}')
    logging.info(f'ETC={ETC}')
    # Initialize online database access.
    db = desietcimg.db.DB(http_fallback=not args.direct)
    # Connect to the exposures table.
    expdb = desietcimg.db.Exposures(db, 'id,night,tileid,exptime,skytime,guidtime,mjd_obs,program')
    # Determine the list of tiles to process.
    tiles = set(args.tiles.split(','))
    try:
        numeric = all([int(tile) > 0 for tile in tiles])
    except ValueError:
        numeric = False
    if numeric:
        expdata = expdb.select(
            f"tileid IN ({args.tiles}) AND exptime>={args.min_exptime} AND night>20200100 AND flavor='science'",
            maxrows=1000)
    elif args.tiles == 'SV1':
        expdata = expdb.select(
            db.where(night=(20201201,None), exptime=(args.min_exptime,None), program='SV%', flavor='science'),
            maxrows=1000)
    else:
        raise ValueError(f'Cannot interpret --tiles {args.tiles}')
    # Loop over exposures for each tile.
    specs = [spec for spec in range(10) if spec not in args.badspec]
    logging.info(f'Processing cameras {args.cameras} for {"".join([str(spec) for spec in specs])}.')
    nexp = len(expdata)
    tiles = set(expdata['tileid'])
    logging.info(f'Processing {nexp} exposures for tiles: {tiles}')
    all_meta = []
    sky_spectra = {C:[] for C in args.cameras}
    throughputs = {C:[] for C in args.cameras}
    sky_grid, thru_grid = [], []
    npix = {C: len(desietcimg.spectro.fullwave[desietcimg.spectro.cslice[C]]) for C in args.cameras}
    for tile in tiles:
        sel = expdata['tileid'] == tile
        ebv = None
        logging.info(f'tile {tile} exposures {list(expdata["id"][sel])}')
        for _, row in expdata[sel].iterrows():
            night = str(int(row['night']))
            expid = int(row['id'])
            exptag = str(expid).zfill(8)
            try:
                path = desietcimg.spectro.get_path(RELEASE, night, expid)
                if not path.exists():
                    logging.error(f'Missing pipeline results for {night}/{expid}.')
                    continue
                if check_exptime(row['exptime']) is None:
                    logging.error(f'Skipping {night}/{exptag} with invalid database EXPTIME.')
                    continue
                if ebv is None:
                    # Calculate the median E(B-V) for the targets on this tile.
                    ebv = desietcimg.spectro.get_ebv(path, specs=specs)
                # Get the exposure header and check for required keys.
                hdr = desietcimg.spectro.get_hdr(path)
                badhdr = False
                missing = [key for key in ('MJD-OBS','EXPTIME','GUIDTIME','SKYTIME','SKYDEC','SKYRA','AIRMASS')
                        if key not in hdr]
                if any(missing):
                    logging.error(f'Missing keywords for {night}/{exptag}: {",".join(missing)}.')
                    continue
                # Get the throughput and detected sky in elec/s/angstrom for this exposure.
                thru = desietcimg.spectro.get_thru(path, specs=specs, cameras=args.cameras)
                sky = desietcimg.spectro.get_sky(path, specs=specs, cameras=args.cameras)
                for c in args.cameras:
                    throughputs[c].append([thru[c].flux, thru[c].ivar])
                    sky_spectra[c].append([sky[c].flux, sky[c].ivar])
                # Should use actual MJD_OBS,EXPTIME instead of db request values
                mjd_spectro = hdr['MJD-OBS']
                exptime_spectro = check_exptime(hdr['EXPTIME'], 'EXPTIME', row['exptime'])
                RA, DEC, X = hdr['SKYRA'], hdr['SKYDEC'], hdr['AIRMASS']
                mjd_grid = mjd_spectro + (0.5 + np.arange(args.ngrid)) / args.ngrid * exptime_spectro / 86400
                exp_dtype = [
                    ('NIGHT','i4'),('EXPID','i4'),('TILEID','i4'),
                    ('SKYRA','f4'),('SKYDEC','f4'),('AIRMASS','f4'),
                    ('MJD-OBS','f8'),('EXPTIME','f4'),('EBV','f4')]
                exp_meta = (int(night), expid, tile, RA, DEC, X, mjd_spectro, exptime_spectro, ebv)
                # Process any available ETC results for this exposure.
                etcdir = ETC / night / exptag
                if not etcdir.exists():
                    logging.error(f'Missing ETC exposure data for {night}/{exptag}')
                else:
                    # Process ETC SKY results for this exposure.
                    exptime_sky = check_exptime(hdr['SKYTIME'], 'SKYTIME', 60.)
                    sky = etcdir / f'sky_{exptag}.csv'
                    if sky.exists():
                        mjd_sky, flux_sky = load_etc_sky(sky, exptime_sky)
                        # Interpolate sky level to MJD grid.
                        flux_sky_grid = np.interp(mjd_grid, mjd_sky, flux_sky)
                        sky_grid.append(flux_sky_grid)
                    else:
                        sky_grid.append(np.zeros_like(mjd_grid))
                        logging.warning(f'Missing ETC sky data for {night}/{exptag}')
                    # Process ETC GFA results for this exposure.
                    exptime_gfa = check_exptime(hdr['GUIDTIME'], 'GUIDTIME', 5.)
                    gfas = sorted(etcdir.glob(f'guide_GUIDE?_{exptag}.csv'))
                    if gfas:
                        mjd_gfa, transp_gfa, ffrac_gfa = load_etc_gfa(gfas, exptime_gfa)
                        thru_gfa = transp_gfa * ffrac_gfa
                        left = np.mean(thru_gfa[:16])
                        right = np.mean(thru_gfa[-16:])
                        thru_grid.append(np.interp(mjd_grid, mjd_gfa, thru_gfa, left=left, right=right))
                    else:
                        thru_grid.append(np.zeros_like(mjd_grid))
                        logging.warning(f'Missing ETC guide data for {night}/{exptag}')
                all_meta.append(exp_meta)

                testout = np.vstack(thru_grid).astype(np.float32)
                assert np.all(np.isfinite(testout)), f'Bad output for {tile},{night},{exptag}'

            except Exception as e:
                logging.error(f'Giving up on {night}/{exptag}: {e}')

    # Save the results.
    if args.save:
        fits = fitsio.FITS(args.save, 'rw', clobber=True)
        fits.write(np.array(all_meta, dtype=exp_dtype), extname='ETC')
        for camera in args.cameras:
            fits.write(np.array(throughputs[camera], np.float32), extname=camera.upper()+'THRU')
            fits.write(np.array(sky_spectra[camera], np.float32), extname=camera.upper()+'SKY')
        fits.write(np.vstack(thru_grid).astype(np.float32), extname='ETCTHRU')
        fits.write(np.vstack(sky_grid).astype(np.float32), extname='ETCSKY')
        fits.close()
        logging.info(f'Saved results to {args.save}.')


def main():
    # https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(
        description='Calculate per-exposure effective depths',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--logpath', type=str, metavar='PATH',
        help='path where logging output should be written')
    parser.add_argument('--debug', action='store_true',
        help='print traceback and enter debugger after an exception')
    parser.add_argument('--tiles', type=str,
        help='comma-separated list of tiles or a predefined name like SV1')
    parser.add_argument('--release', type=str, default='blanc',
        help='pipeline reduction release to use')
    parser.add_argument('--cameras', type=str, default='brz',
        help='spectrograph cameras to use')
    parser.add_argument('--badspec', type=int, nargs='*', default=[],
        help='ignore data from these spectrographs (0-9)')
    parser.add_argument('--desiroot', type=str, default=None,
        help='root path for locating DESI data, defaults to $DESI_ROOT')
    parser.add_argument('--etcpath', type=str, default=None,
        help='path where ETC outputs are stored, defaults to <desiroot>/ETC')
    parser.add_argument('--min-exptime', type=float, default=100,
        help='ignore exposures of duration less than this value')
    parser.add_argument('--skyref', type=str, default='fiducial_sky_eso.fits',
        help='FITS file with the fiducial zenith dark sky model to use')
    parser.add_argument('--smoothing', type=int, default=125,
        help='median filter smoothing to apply to sky spectrum in pixels')
    parser.add_argument('--save', type=str, default='etcdepth.fits',
        help='FITS file where per-exposure results are saved')
    parser.add_argument('--direct', action='store_true',
        help='database connection must be direct')
    parser.add_argument('--ngrid', type=int, default=256,
        help='size of MJD grid for interpolating within each exposure')
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
