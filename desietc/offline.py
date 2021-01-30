"""Utilities to support offline reprocessing of exposures using the ETC online algorithms.
"""
import logging

import numpy as np

import fitsio


def replay_exposure(night, expid):
    """Recreate the online ETC processing of an exposure by replaying the
    FITS files stored to disk.
    """
    logging.info(f'Processing {night}/{expid}...')


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
