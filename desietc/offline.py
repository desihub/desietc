"""Utilities to support offline reprocessing of exposures using the ETC online algorithms.
"""
import logging

import numpy as np

import fitsio

import desietc.gfa
import desietc.sky


def replay_exposure(ETC, path, expid):
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
    if not gfa_path.exists():
        logging.error(f'Missing GFA data cube: {gfa_path}.')
        return False
    if not sky_path.exists():
        logging.error(f'Missing SKY data cube: {sky_path}.')
        return False
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
