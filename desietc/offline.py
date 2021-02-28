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

def fetch_exposure(path, expid, only_complete=True):
    """
    """
    exptag = str(int(expid)).zfill(8)
    exppath = path / exptag
    if not exppath.exists():
        logging.error(f'No exposure directory found.')
        return None
    # What data is available for this exposure?
    acq_path = exppath / f'guide-{exptag}-0000.fits.fz'
    gfa_path = exppath / f'guide-{exptag}.fits.fz'
    sky_path = exppath / f'sky-{exptag}.fits.fz'
    desi_path = exppath / f'desi-{exptag}.fits.fz'
    fassign_paths = list(exppath.glob('fiberassign-??????.fits*'))
    missing = 0
    if not acq_path.exists():
        logging.warn(f'Missing acquisition image: {acq_path}.')
        missing += 1
    if not gfa_path.exists():
        logging.warn(f'Missing GFA data cube: {gfa_path}.')
        missing += 1
    if not sky_path.exists():
        logging.warn(f'Missing SKY data cube: {sky_path}.')
        missing += 1
    if not desi_path.exists():
        logging.warn(f'Missing DESI exposure: {desi_path}.')
        missing += 1
    if len(fassign_paths) != 1:
        logging.warn(f'Missing fiberassign file.')
        fassign_path = exppath / '_does_not_exist_'
        missing += 1
    else:
        fassign_path = fassign_paths[0]
    if only_complete and missing > 0:
        return None
    if desi_path.exists():
        # Get the spectrograph info for this exposure.
        desi_hdr = fitsio.read_header(str(desi_path), ext='SPEC')
        missing = 0
        for key in 'NIGHT', 'MJD-OBS', 'TILEID', 'EXPTIME':
            if key not in desi_hdr:
                logging.error(f'DESI exposure missing {key}.')
                missing += 1
        if missing > 0:
            return None
        night = desi_hdr['NIGHT']
        desi_mjd_obs = desi_hdr['MJD-OBS']
        desi_exptime = desi_hdr['EXPTIME']
        desi_tileid = desi_hdr['TILEID']
    # Get the SKY exposure info for the first available camera.
    sky_info = []
    if sky_path.exists():
        try:
            with fitsio.FITS(str(sky_path)) as hdus:
                for camera in desietc.sky.SkyCamera.sky_names:
                    if camera+'T' not in hdus:
                        continue
                    sky_info = hdus[camera + 'T'].read()
                    break
        except Exception as e:
            logging.error(f'Failed to read SKY data: {e}')
    num_sky_frames = len(sky_info)
    logging.info(f'Exposure has {num_sky_frames} SKY frames.')
    # Get the PM guide stars and GFA exposure info for the first available camera.
    gfa_info, pm_info = [], None
    if gfa_path.exists():
        try:
            with fitsio.FITS(str(gfa_path)) as hdus:
                # Lookup the PlateMaker guide stars.
                if 'PMGSTARS' not in hdus:
                    logging.error(f'Missing PMGSTARS HDU: no guide stars specified.')
                    return None
                pm_info = hdus['PMGSTARS'].read()
                logging.info(f'Exposure has {len(pm_info[0])} guide stars.')
                for camera in desietc.gfa.GFACamera.guide_names:
                    if camera+'T' not in hdus:
                        continue
                    gfa_info = hdus[camera + 'T'].read()
                    break
        except Exception as e:
            logging.error(f'Failed to read GFA data: {e}')
    elif acq_path.exists():
        # We have an acq image but no subsequent guider frames.
        try:
            with fitsio.FITS(str(acq_path)) as hdus:
                for camera in desietc.gfa.GFACamera.guide_names:
                    if camera not in hdus:
                        continue
                    hdr = hdus[camera].read_header()
                    mjd_obs, exptime = hdr['MJD-OBS'], hdr['EXPTIME']
                    gfa_info = np.array(
                        [(mjd_obs, exptime)], dtype=[('MJD-OBS', float), ('EXPTIME', float)])
                    break
        except Exception as e:
            logging.error(f'Failed to read ACQ data: {e}')
    num_gfa_frames = len(gfa_info)
    logging.info(f'Exposure has {num_gfa_frames} GFA frames.')
    # Determine the order in which the combined GFA+SKY frames should be fed to the ETC.
    frames = (
        [ dict(typ='gfa', num=n, when=gfa_info[n]['MJD-OBS']+gfa_info[n]['EXPTIME'] / 86400)
          for n in range(num_gfa_frames) ] +
        [ dict(typ='sky', num=n, when=sky_info[n]['MJD-OBS']+sky_info[n]['EXPTIME'] / 86400)
          for n in range(num_sky_frames) ])
    frames = sorted(frames, key=lambda frame: frame['when'])
    if len(frames) == 0:
        logging.error(f'No GFA or SKY frames found.')
        return None
    # If we get this far, return a dictionary of the fetched results.
    return dict(
        expid=expid,
        exptag=exptag,
        desi_path=desi_path,
        desi_mjd_obs=desi_mjd_obs,
        desi_exptime=desi_exptime,
        fassign_path=fassign_path,
        pm_info=pm_info,
        acq_path=acq_path,
        gfa_path=gfa_path,
        sky_path=sky_path,
        gfa_info=gfa_info,
        sky_info=sky_info,
        frames=frames,
    )


def replay_exposure(ETC, path, expid, outpath, teff=1000, ttype='DARK', cutoff=3600, cosmic=1200,
                    maxsplit=3, splittable=True, overwrite=False, dry_run=False, only_complete=True):
    """Recreate the online ETC processing of an exposure by replaying the
    FITS files stored to disk.
    """
    logging.info(f'Replaying expid {expid} in {path}...')

    F = fetch_exposure(path, expid, only_complete)
    if F is None:
        return False

    # If this is a dry run, stop now.
    if dry_run:
        return True
    # If the output directory exists, can we overwrite its contents?
    exppath_out = outpath / F['exptag']
    if not overwrite and exppath_out.exists():
        logging.info(f'Will not overwrite ETC outputs for {expid}.')
        return False
    # Create the output exposure directory if necessary.
    exppath_out.mkdir(parents=False, exist_ok=True)
    # Save images with the per-exposure outputs.
    ETC.set_image_path(exppath_out)
    # Start the exposure processing.
    if len(F['gfa_info']) > 0:
        mjd_first_frame = F['gfa_info'][0]['MJD-OBS']
    else:
        mjd_first_frame = F['sky_info'][0]['MJD-OBS']
    timestamp = desietc.util.mjd_to_date(mjd_first_frame - 1 / ETC.SECS_PER_DAY, utc_offset=0)
    ETC.start_exposure(timestamp, expid, teff, ttype, cutoff, cosmic, maxsplit, splittable)
    # Loop over frames to replay.
    for frame in F['frames']:
        if frame['typ'] == 'gfa':
            if frame['num'] == 0 and F['acq_path'].exists():
                data = acq_to_online(F['acq_path'], desietc.gfa.GFACamera.guide_names)
            else:
                data = fits_to_online(F['gfa_path'], desietc.gfa.GFACamera.guide_names, frame['num'])
            if frame['num'] == 0:
                # Process the acquisition image.
                ETC.process_acquisition(data)
                if F['fassign_path'].exists():
                    # Read the fiber assignments for this tile.
                    ETC.read_fiberassign(F['fassign_path'])
                if F['pm_info'] is not None:
                    # Specify the guide stars.
                    ETC.set_guide_stars(F['pm_info'])
                if F['desi_path'].exists():
                    # Signal the shutter opening.
                    timestamp = desietc.util.mjd_to_date(F['desi_mjd_obs'], utc_offset=0)
                    ETC.open_shutter(timestamp)
            else:
                # Process the next guide frame.
                ETC.process_guide_frame(data)
        else: # SKY
            data = fits_to_online(F['sky_path'], ETC.SKY.sky_names, frame['num'])
            ETC.process_sky_frame(data)
    if F['desi_path'].exists():
        # Signal the shutter closing.
        timestamp = desietc.util.mjd_to_date(
            F['desi_mjd_obs'] + F['desi_exptime'] / ETC.SECS_PER_DAY, utc_offset=0)
        ETC.close_shutter(timestamp)
        # End the exposure.
        ETC.stop_exposure(timestamp)
        # Save the ETC outputs for this exposure.
        ETC.save_exposure(exppath_out)
        # Plot the signal and background measurement buffers spanning this exposure.
        mjd1 = F['desi_mjd_obs']
        mjd2 = mjd1 + F['desi_exptime'] / ETC.SECS_PER_DAY
        fig, ax = plt.subplots(2, 1, figsize=(9, 9))
        fig.suptitle(f'ETC Analysis for {ETC.night}/{ETC.exptag}')
        desietc.plot.plot_measurements(
            ETC.sky_measurements, mjd1, mjd2, label='SKYCAM Level', ax=ax[0])
        desietc.plot.plot_measurements(
            ETC.thru_measurements, mjd1, mjd2, label='GFA Throughput', ax=ax[1])
        plt.savefig(exppath_out / f'etc-measure-{ETC.exptag}.png')
        plt.close(fig)

    return True


def acq_to_online(path, names):
    online = {}
    with fitsio.FITS(str(path)) as hdus:
        online['header']= dict(hdus[0].read_header())
        online['GUIDER']= dict(header=dict(hdus['GUIDER'].read_header()))
        for ext in names:
            if ext not in hdus:
                continue
            data = hdus[ext][:,:]
            hdr = hdus[ext].read_header()
            online[ext] = dict(header=hdr, data=data)
    return online


def fits_to_online(path, names, frame):
    """Read a FITS file and prepare a dictionary containing its headers and arrays
    in the same format used by the DESI online software.
    """
    online = {}
    with fitsio.FITS(str(path)) as hdus:
        online['header']= dict(hdus[0].read_header())
        for hdrname in 'GUIDER', 'SKY':
            if hdrname in hdus:
                online[hdrname] = dict(header=dict(hdus[hdrname].read_header()))
        for ext in names:
            if ext not in hdus or ext + 'T' not in hdus:
                continue
            dims = hdus[ext].get_dims()
            if len(dims) == 2:
                if frame != 0:
                    raise ValueError(f'Requested frame {frame} when no frames present.')
                data = hdus[ext][:,:]
            elif len(dims) == 3:
                if frame >= dims[0]:
                    raise ValueError(f'Requested non-existent frame {frame}.')
                data = hdus[ext][frame,:,:][0]
            else:
                raise ValueError(f'Data has invalid dimensions: {dims}.')
            table = hdus[ext + 'T'][frame]
            hdr = {key: table[key] for key in table.dtype.names}
            online[ext] = dict(header=hdr, data=data)
    return online
