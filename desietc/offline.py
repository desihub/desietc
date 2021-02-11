"""Utilities to support offline reprocessing of exposures using the ETC online algorithms.
"""
try:
    import DOSlib.logger as logging
except ImportError:
    # Fallback when we are not running as a DOS application.
    import logging

import numpy as np

import matplotlib.pyplot as plt

import fitsio

import desietc.gfa
import desietc.sky
import desietc.plot

def replay_exposure(ETC, path, expid, outpath, teff=1000, cutoff=10000, cosmic=500,
                    overwrite=False, dry_run=False):
    """Recreate the online ETC processing of an exposure by replaying the
    FITS files stored to disk.
    """
    logging.info(f'Replaying expid {expid} in {path}...')
    exptag = str(int(expid)).zfill(8)
    exppath = path / exptag
    if not exppath.exists():
        logging.error(f'No exposure directory found.')
        return False
    # What data is available for this exposure?
    gfa_path = exppath / f'guide-{exptag}.fits.fz'
    sky_path = exppath / f'sky-{exptag}.fits.fz'
    desi_path = exppath / f'desi-{exptag}.fits.fz'
    missing = 0
    if not gfa_path.exists():
        logging.warn(f'Missing GFA data cube: {gfa_path}.')
        missing += 1
    if not sky_path.exists():
        logging.warn(f'Missing SKY data cube: {sky_path}.')
        missing += 1
    if not desi_path.exists():
        logging.warn(f'Missing DESI exposure: {desi_path}.')
        missing += 1
    if missing > 0:
        return False
    # Get the spectrograph info for this exposure.
    desi_hdr = fitsio.read_header(str(desi_path), ext='SPEC')
    missing = 0
    for key in 'NIGHT', 'MJD-OBS', 'TILEID', 'EXPTIME':
        if key not in desi_hdr:
            logging.error('DESI exposure missing {key}.')
            missing += 1
    if missing > 0:
        return False
    night = desi_hdr['NIGHT']
    desi_mjd_obs = desi_hdr['MJD-OBS']
    desi_exptime = desi_hdr['EXPTIME']
    desi_tileid = desi_hdr['TILEID']
    # Locate the fiberassign file for this tile.
    fassign_path = exppath / f'fiberassign-{desi_tileid:06d}.fits'
    if not fassign_path.exists():
        logging.error(f'Missing fiberassign file: {fassign_path}.')
        return False
    # Read the fiberassign file.
    fassign = fitsio.read(str(fassign_path), ext='FIBERASSIGN')
    sel = (fassign['OBJTYPE'] == 'TGT')
    ntarget = np.count_nonzero(sel)
    median_Ebv = np.nanmedian(fassign[sel]['EBV'])
    logging.info(f'Tile {desi_tileid} has {ntarget} targets with median(Ebv)={median_Ebv:.5f}.')
    # If this is a dry run, stop now.
    if dry_run:
        return True
    # If the output directory exists, can we overwrite its contents?
    exppath_out = outpath / exptag
    if not overwrite and exppath_out.exists():
        logging.info(f'Will not overwrite ETC outputs for {expid}.')
        return False
    # Start the ETC tracking of this exposure.
    ETC.start_exposure(night, expid, desi_mjd_obs, median_Ebv, teff, cutoff, cosmic)
    # Get the SKY exposure info for the first available camera.
    sky_info = None
    with fitsio.FITS(str(sky_path)) as hdus:
        for camera in desietc.sky.SkyCamera.sky_names:
            if camera+'T' not in hdus:
                continue
            sky_info = hdus[camera + 'T'].read()
            break
    if sky_info is None:
        logging.error(f'Unable to find SKY exposure info.')
        return False
    num_sky_frames = len(sky_info)
    logging.info(f'Exposure has {num_sky_frames} SKY frames.')
    # Get the PM guide stars and GFA exposure info for the first available camera.
    gfa_info = None
    with fitsio.FITS(str(gfa_path)) as hdus:
        # Lookup the PlateMaker guide stars.
        if 'PMGSTARS' not in hdus:
            logging.error(f'Missing PMGSTARS HDU: no guide stars specified.')
            return False
        pm = hdus['PMGSTARS'].read()
        guide_stars = pm['GFA_LOC'], pm['ROW'], pm['COL'], pm['MAG']
        logging.info(f'Exposure has {len(guide_stars[0])} guide stars.')
        for camera in desietc.gfa.GFACamera.guide_names:
            if camera+'T' not in hdus:
                continue
            gfa_info = hdus[camera + 'T'].read()
            break
    if gfa_info is None:
        logging.error(f'Unable to find GFA exposure info.')
        return False
    num_gfa_frames = len(gfa_info)
    logging.info(f'Exposure has {num_gfa_frames} GFA frames.')
    # Determine the order in which the combined GFA+SKY frames should be fed to the ETC.
    frames = (
        [ dict(typ='gfa', num=n, when=gfa_info[n]['MJD-OBS']+gfa_info[n]['EXPTIME']/86400)
          for n in range(num_gfa_frames) ] +
        [ dict(typ='sky', num=n, when=sky_info[n]['MJD-OBS']+sky_info[n]['EXPTIME']/86400)
          for n in range(num_sky_frames) ])
    frames = sorted(frames, key=lambda frame: frame['when'])
    # Loop over frames to replay.
    for frame in frames:
        logging.debug(f'Replaying frame: {frame}')
        if frame['typ'] == 'gfa':

            continue

            data = fits_to_online(gfa_path, ETC.GFA.guide_names, frame['num'])
            if frame['num'] == 0:
                # Process the acquisition image.
                ETC.process_acquisition(data)
                # Specify the guide stars.
                ETC.set_guide_stars(*guide_stars)
            else:
                # Process the next guide frame.
                ETC.process_guide_frame(data)
        else: # SKY
            data = fits_to_online(sky_path, ETC.SKY.sky_names, frame['num'])
            ETC.process_sky(data)

    # Create the output path if necessary.
    exppath_out.mkdir(parents=False, exist_ok=True)

    # Save the ETC outputs for this exposure.
    ETC.save_exposure(exppath_out)

    mjd1 = ETC.mjd_start
    mjd2 = mjd1 + desi_exptime / ETC.SECS_PER_DAY
    teff = ETC.get_accumulated_teff(mjd1, mjd2, ETC.MW_transparency)

    desietc.plot.plot_measurements(
        ETC.sky_measurements, mjd1, mjd2, label='Sky Level')
    plt.savefig(exppath_out / f'sky-{ETC.expid}.png')

    return True


def fits_to_online(path, names, frame):
    """Read a FITS file and prepare a dictionary containing its headers and arrays
    in the same format used by the DESI online software.
    """
    online = {}
    with fitsio.FITS(str(path)) as hdus:
        online['header']= dict(hdus[0].read_header())
        for ext in names:
            if ext not in hdus or ext + 'T' not in hdus:
                continue
            data = hdus[ext][frame,:,:][0]
            table = hdus[ext + 'T'][frame]
            hdr = {key: table[key] for key in table.dtype.names}
            online[ext] = dict(header=hdr, data=data)
    return online
