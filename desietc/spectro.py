"""Utility functions for accessing DESI spectra for ETC calculations.
"""
import contextlib
import logging

import numpy as np

import scipy.ndimage

import fitsio


# Define the pipeline reduction output wavelength grid.
wmin, wmax, wdelta = 3600, 9824, 0.8
fullwave = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice = {'b': slice(0, 2751), 'r': slice(2700, 5026), 'z': slice(4900, 7781)}
cwave = {c: fullwave[cs] for (c, cs) in cslice.items()}

# Calculate the ergs/photon in each wavelength bin.
#erg_per_photon = (h * c / (fullwave * u.Angstrom)).to(u.erg).value
erg_per_photon = 1.986445857148928e-08 / fullwave

# Define the primary mirror area assumed for etendue calculations.
# https://desi.lbl.gov/svn/code/desimodel/trunk/data/desi.yaml
# geomarea = PI*(M1_diameter/2)^2 - PI*(obscuration_diameter/2)^2 - 4*trussarea
# trussarea = 0.5*(M1_diameter-obscuration_diameter) * M2_support_width
# M1_diameter: 3.797          # meters
# obscuration_diameter: 1.8   # meters
# M2_support_width: 0.03      # meters
M1_area = 8.658739421e4 # in cm2

# Define the field-averaged fiber sky area assumed for etendue calculations.
# Based on ECHO22 platescales.
fiber_solid_angle = 1.9524634 # in sq.arcsec.


class Spectrum(object):
    """Simple container of flux and ivar for a single camera or the full wavelength range.

    Dividing by a constant or vector updates both the flux and ivar.
    """
    def __init__(self, stype, flux=None, ivar=None):
        assert stype == 'full' or stype in cslice, 'invalid stype'
        self.stype = stype
        self.wave = fullwave[cslice[stype]] if stype in cslice else fullwave
        if flux is None and ivar is None:
            self._flux = np.zeros(len(self.wave))
            self.ivar = np.zeros(len(self.wave))
        elif flux is not None and ivar is not None:
            self._flux = np.asarray(flux)
            self.ivar = np.asarray(ivar)
            assert self.ivar.shape == self._flux.shape, 'flux and ivar have different shapes.'
        else:
            raise ValueError('flux and ivar must both be specified.')
    def copy(self):
        return Spectrum(self.stype, self.flux.copy(), self.ivar.copy())
    def __itruediv__(self, factor):
        np.divide(self.flux, factor, out=self._flux, where=factor != 0)
        self.ivar *= factor ** 2
        return self
    def __truediv__(self, factor):
        result = self.copy()
        result /= factor
        return result
    @property
    def flux(self):
        return self._flux


class CoAdd(Spectrum):
    """Implements += to perform ivar-weighted coaddition.
    """
    def __init__(self, stype):
        super(CoAdd, self).__init__(stype)
        self._weighted_flux_sum = np.zeros(len(self.wave))
        self._finalized = False
    def __iadd__(self, other):
        if other.stype == self.stype:
            self_slice = slice(None, None)
        elif self.stype == 'full':
            self_slice = cslice[other.stype]
        else:
            raise ValueError(f'Cannot add "{other.stype}" to "{self.stype}".')
        self._weighted_flux_sum[self_slice] += other.ivar * other.flux
        self.ivar[self_slice] += other.ivar
        self._finalized = False
        return self
    @property
    def flux(self):
        if not self._finalized:
            np.divide(self._weighted_flux_sum, self.ivar, out=self._flux, where=self.ivar > 0)
            self._finalized = True
        return self._flux


def iterspecs(path, ftypes='cframe', specs=range(10), cameras='brz', expid=None,
              openfits=True, camera_first=True, missing='warn'):
    """Iterate over all FITS files with names <ftype>-<camera><spec>-<expid>.fits*
    Yields (hdus, camera, spec) for each file, or (fname, camera, spec) if openfits is False.
    Iterates over camera then spec if camera_first is True, otherwise spec then camera.
    If ftypes is a comma-separated list, open all files within a single context-manager stack.
    If expid is None, use the enclosing directory name.
    """
    ftypes = ftypes.split(',')
    if expid is None:
        expid = path.name
    elif type(expid) is int:
        expid = expid = str(expid).zfill(8)
    if camera_first:
        outers, inners = cameras, specs
        info = lambda a,b: (a,b)
    else:
        outers, inners = specs, cameras
        info = lambda a,b: (b,a)
    for outer in outers:
        for inner in inners:
            camera, spec = info(outer, inner)
            # Build the list of file paths for this (camera,spec).
            fnames = [next(iter(path.glob(f'{ftype}-{camera}{spec}-{expid}.fits*')), None) for ftype in ftypes]
            if None in fnames:
                what = ",".join([ftype for (i,ftype) in enumerate(ftypes) if fnames[i] is None])
                if missing == 'error':
                    raise RuntimeError(f'Missing {what} for {camera}{spec} in {path}.')
                elif missing == 'warn':
                    logging.warning(f'Missing {what} for {camera}{spec} in {path}.')
                    continue
            if not openfits:
                # Return file paths without opening them.
                yield fnames, camera, spec
            else:
                # Use context managers to open each file.
                with contextlib.ExitStack() as stack:
                    hdus = [stack.enter_context(fitsio.FITS(str(fname))) for fname in fnames]
                    yield hdus, camera, spec


def get_path(release, night, expid, required=False):
    path = release / 'exposures' / str(night) / str(expid).zfill(8)
    if required and not path.exists():
        raise FileExistsError(path)
    return path


def get_sky(path, specs=range(10), cameras='brz'):
    """Calculate the mean detected sky in each camera in elec/s/Angstrom.
    """
    detected = {c:CoAdd(c) for c in cameras}
    exptime = None
    for (SKY,), camera, spec in iterspecs(path, 'sky'):
        if exptime is None:
            exptime = SKY[0].read_header()['EXPTIME']
        else:
            if SKY[0].read_header()['EXPTIME'] != exptime:
                raise RuntimeError(f'EXPTIME mismatch for sky in {path}')
        flux, ivar = SKY['SKY'].read(), SKY['IVAR'].read()
        detected[camera] += Spectrum(camera, np.median(flux, axis=0), np.median(ivar, axis=0))
    # Convert from elec/Ang to elec/Ang/sec
    for camera in cameras:
        detected[camera] /= exptime
    return detected


def get_thru(path, specs=range(10), cameras='brz'):
    """Calculate the throughput in each camera for a single exposure.
    See https://github.com/desihub/desispec/blob/master/bin/desi_average_flux_calibration
    and DESI-6043.
    The result includes the instrument throughput as well as the fiber acceptance
    loss and atmospheric extinction.
    """
    calibs = {c:CoAdd(c) for c in cameras}
    exptime = None
    primary_area = 8.659e4 # cm2
    for (FCAL,), camera, spec in iterspecs(path, 'fluxcalib'):
        if exptime is None:
            hdr = FCAL[0].read_header()
            exptime = hdr['EXPTIME']
        else:
            if FCAL[0].read_header()['EXPTIME'] != exptime:
                raise RuntimeError(f'EXPTIME mismatch for fluxcalib in {path}')
        fluxcalib, ivar = FCAL['FLUXCALIB'].read(), FCAL['IVAR'].read()
        calibs[camera] += Spectrum(camera, np.median(fluxcalib, axis=0), np.median(ivar, axis=0))
    for camera in cameras:
        # Convert from (1e17 elec cm2 s / erg) to (elec/phot)
        calibs[camera] /= (M1_area * exptime) / (1e17 * erg_per_photon[cslice[camera]])
    return calibs


def get_ebv(path, specs=range(10)):
    """Lookup the EBV value for all targets from the CFRAME fibermap.
    Return the median of all non-zero values.
    """
    ebvs = []
    for (CFRAME,), camera, spec in iterspecs(path, 'cframe', specs=specs, cameras='b'):
        ebvs.append(CFRAME['FIBERMAP'].read(columns=['EBV'])['EBV'].astype(np.float32))
    ebvs = np.stack(ebvs).reshape(-1)
    nonzero = ebvs > 0
    ebvs = ebvs[nonzero]
    return np.nanmedian(ebvs)


def get_hdr(path):
    """Return the FLUX HDU header from any CFRAME file at this path.
    """
    for (CFRAME,), camera, spec in iterspecs(path, 'cframe'):
        return CFRAME['FLUX'].read_header()
