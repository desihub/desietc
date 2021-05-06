"""Analyze images captured by the DESI GFA cameras for the online ETC.
"""
try:
    import DOSlib.logger as logging
except ImportError:
    # Fallback when we are not running as a DOS application.
    import logging

import numpy as np

import fitsio

import desietc.util


def save_calib_data(name='GFA_calib.fits', comment='GFA in-situ calibration results',
                    readnoise=None, gain=None, master_zero=None, pixel_mask=None, tempfit=None,
                    master_dark=None, overwrite=True):
    """Any elements left blank will be copied from the current default calib data.
    """
    GFA = desietc.gfa.GFACamera()
    if master_zero is None:
        print('Using default master_zero')
        master_zero = GFA.master_zero
    if master_dark is None:
        print('Using default master_dark')
        master_dark = GFA.master_dark
    if pixel_mask is None:
        print('Using default pixel_mask')
        pixel_mask = GFA.pixel_mask
    _readnoise, _gain, _tempfit = {}, {}, {}
    for gfa in GFA.gfa_names:
        _readnoise[gfa] = {}
        _gain[gfa] = {}
        _tempfit[gfa] = {}
        for amp in GFA.amp_names:
            calib = GFA.calib_data[gfa][amp]
            _readnoise[gfa][amp] = calib['RDNOISE']
            _gain[gfa][amp] = calib['GAIN']
        calib = GFA.calib_data[gfa]
        for k in 'TREF', 'IREF', 'TCOEF', 'I0', 'C0':
            _tempfit[gfa][k] = calib[k]
    if readnoise is None:
        print('Using default readnoise')
        readnoise = _readnoise
    if gain is None:
        print('Using default gain')
        gain = _gain
    if tempfit is None:
        print('Using default tempfit')
        tempfit = _tempfit
    with fitsio.FITS(name, 'rw', clobber=overwrite) as hdus:
        # Write a primary HDU with only the comment.
        hdus.write(np.zeros((1,), dtype=np.float32), header=dict(COMMENT=comment))
        # Loop over GFAs.
        for gfanum, gfa in enumerate(desietc.gfa.GFACamera.gfa_names):
            hdr = {}
            for amp in desietc.gfa.GFACamera.amp_names:
                hdr['RDNOISE_{0}'.format(amp)] = readnoise[gfa][amp]
                hdr['GAIN_{0}'.format(amp)] = gain[gfa][amp]
            # Add dark current temperature fit results.
            for k, v in tempfit[gfa].items():
                hdr[k] = v
            # Write the per-GFA image arrays.
            hdus.write(master_zero[gfa], header=hdr, extname='ZERO{}'.format(gfanum))
            hdus.write(master_dark[gfa], extname='DARK{}'.format(gfanum))
            hdus.write(pixel_mask[gfa].astype(np.uint8), extname='MASK{}'.format(gfanum))
    print('Saved GFA calib data to {0}.'.format(name))


def load_calib_data(name='GFA_calib.fits'):
    data = {}
    master_zero = {}
    master_dark = {}
    pixel_mask = {}
    with fitsio.FITS(str(name)) as hdus:
        # Loop over GFAs.
        for gfanum, gfa in enumerate(desietc.gfa.GFACamera.gfa_names):
            hdr = hdus['ZERO{0}'.format(gfanum)].read_header()
            data[gfa] = {}
            for amp in desietc.gfa.GFACamera.amp_names:
                data[gfa][amp] = {
                    'RDNOISE': hdr['RDNOISE_{0}'.format(amp)],
                    'GAIN': hdr['GAIN_{0}'.format(amp)],
                }
            for key in 'TREF', 'IREF', 'TCOEF', 'I0', 'C0':
                data[gfa][key] = hdr.get(key, -1)
            master_zero[gfa] = hdus['ZERO{0}'.format(gfanum)].read().copy()
            master_dark[gfa] = hdus['DARK{0}'.format(gfanum)].read().copy()
            pixel_mask[gfa] = hdus['MASK{0}'.format(gfanum)].read().astype(np.bool)
    logging.info('Loaded GFA calib data from {0}'.format(name))
    return data, master_zero, master_dark, pixel_mask


class GFACamera(object):

    gfa_names = [
        'GUIDE0', 'FOCUS1', 'GUIDE2', 'GUIDE3', 'FOCUS4',
        'GUIDE5', 'FOCUS6', 'GUIDE7', 'GUIDE8', 'FOCUS9']
    guide_names = [name for name in gfa_names if name.startswith('GUIDE')]
    ##guide_names = ['GUIDE0']
    amp_names = ['E', 'F', 'G', 'H']

    nampy=516
    nampx=1024
    nscan=50
    nxby2 = nampx + 2 * nscan
    quad = {
        'E': (slice(None, nampy), slice(None, nampx)), # bottom left
        'H': (slice(nampy, None), slice(None, nampx)), # top left
        'F': (slice(None, nampy), slice(nampx, None)), # bottom left
        'G': (slice(nampy, None), slice(nampx, None)), # top left
    }

    buffer_shape = (2, 2 * nampy, 2 * nampx)
    buffer_size = 2 * (2 * nampy) * (2 * nampx) * np.dtype(np.float32).itemsize

    maxdither = 1
    psf_stampsize = 45
    psf_inset = 4
    psf_stacksize = psf_stampsize - 2 * (psf_inset + maxdither)
    donut_stampsize = 65
    donut_inset = 8
    donut_stacksize = donut_stampsize - 2 * (donut_inset + maxdither)

    lab_data = None
    calib_data = None
    master_zero = None
    master_dark = None
    pixel_mask = None

    def __init__(self, nrowtrim=4, maxdelta=50, calib_name='GFA_calib.fits', buffer=None):
        """Initialize a GFA Camera analysis object.

        Parameters
        ----------
        nrowtrim : int
            Number of overscan rows to trim before calculating the bias.
        maxdelta : float
            Maximum deviation between an overscan pixel value and the median
            overscan pixel value in order for an overscan pixel to be
            considered "good".  A large number of bad pixels is probably
            an indication that the GFA is exhibiting excessive pattern noise.
        calib_name : str
            Name of the FITS file containing the GFA calibration data to use.
        buffer : object exposing buffer interface or None
            Buffer of size at least buffer_size to use for our data and
            ivar arrays.  Allocate new memory when None.
        """
        self.nrowtrim = nrowtrim
        self.maxdelta = maxdelta
        if buffer is not None:
            # Use the buffer provided, e.g. in shared memory.
            self.array = np.ndarray(self.buffer_shape, dtype=np.float32, buffer=buffer)
        else:
            # Allocate and zero new memory.
            self.array = np.zeros(self.buffer_shape, np.float32)
        assert self.array.nbytes == self.buffer_size
        self.data = self.array[0]
        self.ivar = self.array[1]
        # Load the class-level calib data if necessary.
        if GFACamera.calib_data is None:
            (GFACamera.calib_data, GFACamera.master_zero,
             GFACamera.master_dark, GFACamera.pixel_mask) = load_calib_data(calib_name)
        # We have no centering algorithms initialized yet.
        self.psf_centering = None
        self.donut_centering = None

    def setraw(self, raw, name=None, overscan_correction=True, subtract_master_zero=True, apply_gain=True):
        """Initialize using the raw GFA data provided for a single exposure.

        After calling this method the following attributes are set:

            bias : dict of arrays
                Bias values in ADU estimated from the overscan in each exposure, indexed by the amplifier name.
            amps : dict of view
                Raw array views indexed by amplifier name, including pre and post overscan regions, in row
                and column readout order.
            unit : str
                Either 'elec' or 'ADU' depending on the value of apply_gain.
            data : 2D array of float32
                Bias subtracted pixel values in elec (or ADU if apply_gain is False) of shape
                (2 * nampy, 2 * nampx) with pre and post overscan regions removed from the raw data.
            ivar : 2D array of float32
                Inverse variance estimated in units matched to the data array.

        To calculate the estimated dark current, use :meth:`get_dark_current`.  To remove the overscans
        but not apply any calibrations, set all options to False.

        Parameters:
            raw : numpy array
                An array of raw data with shape (ny, nx). The raw input is not copied or modified.
            name : str or None
                Name of the camera that produced this raw data. Must be set to one of the values in gfa_names
                in order to lookup the correct master zero and dark images, and amplifier parameters, when
                these features are used.
            overscan_correction : bool
                Subtract the per-amplifier bias estimated from each overscan region when True. Otherwise,
                these biases are still calculated and available in ``bias[amp]`` but not subtracted.
            subtract_master_zero : bool
                Subtract the master zero image for this camera after applying overscan bias correction.
            apply_gain : bool
                Convert from ADU to electrons using the gain specified for this camera.
        """
        if raw.ndim != 2:
            raise ValueError('raw data must be 2D.')
        raw_shape = (2 * self.nampy, 2 * self.nampx + 4 * self.nscan)
        if raw.shape != raw_shape:
            raise ValueError('raw data has dimensions {0} but expected {1}.'.format(raw.shape, raw_shape))
        ny, nx = raw.shape
        if name not in self.gfa_names:
            logging.warning('Not a valid GFA name: {0}.'.format(name))
        self.name = name
        # Create views (with no data copied) for each amplifier with rows and column in readout order.
        self.amps = {
            'E': raw[:self.nampy, :self.nxby2], # bottom left (using convention that raw[0,0] is bottom left)
            'H': raw[-1:-(self.nampy + 1):-1, :self.nxby2], # top left
            'F': raw[:self.nampy, -1:-(self.nxby2+1):-1], # bottom right
            'G': raw[-1:-(self.nampy + 1):-1, -1:-(self.nxby2+1):-1], # top right
        }
        # Calculate bias as mean overscan, ignoring the first nrowtrim rows
        # (in readout order) and any values > maxdelta from the per-exposure median overscan.
        # Since we use a mean rather than median, subtracting this bias changes the dtype from
        # uint32 to float32 and means that digitization noise averages out over exposures.
        self.bias = {}
        self.nbad_overscan = 0
        for amp in self.amp_names:
            overscan = self.amps[amp][self.nrowtrim:, -self.nscan:]
            delta = overscan - np.median(overscan)
            bad = np.abs(delta) > self.maxdelta
            ngood = (self.nampy - self.nrowtrim) * self.nscan
            if np.any(bad):
                nbad = np.count_nonzero(bad)
                logging.debug(f'Ignoring {nbad} bad overscan pixels for {name}-{amp}.')
                overscan = np.copy(overscan)
                overscan[bad] = 0.
                ngood -= nbad
                self.nbad_overscan += nbad
            self.bias[amp] = np.sum(overscan) / ngood
        # Assemble the real pixel data with the pre and post overscans removed.
        self.data[:self.nampy, :self.nampx] = raw[:self.nampy, self.nscan:self.nampx + self.nscan]
        self.data[:self.nampy, self.nampx:] = raw[:self.nampy, self.nxby2 + self.nscan:-self.nscan]
        self.data[self.nampy:, :self.nampx] = raw[self.nampy:, self.nscan:self.nampx + self.nscan]
        self.data[self.nampy:, self.nampx:] = raw[self.nampy:, self.nxby2 + self.nscan:-self.nscan]
        if overscan_correction:
            # Apply the overscan bias corrections.
            for amp in self.amp_names:
                self.data[self.quad[amp]] -= self.bias[amp]
        # Subtract the master zero if requested.
        if subtract_master_zero:
            self.data -= GFACamera.master_zero[name]
        # Apply the gain correction if requested.
        if apply_gain:
            calib = GFACamera.calib_data[name]
            for amp in self.amp_names:
                self.data[self.quad[amp]] *= calib[amp]['GAIN']
            # Use the calculated signal in elec as the estimate of Poisson variance.
            self.ivar = np.maximum(self.data, 0, out=self.ivar)
            # Add the per-amplifier readnoise to the variance.
            for amp in self.amp_names:
                rdnoise_in_elec = calib[amp]['RDNOISE'] * calib[amp]['GAIN']
                self.ivar[self.quad[amp]] += rdnoise_in_elec ** 2
            # Convert var to ivar in-place, avoiding divide by zero.
            self.ivar = np.divide(1, self.ivar, out=self.ivar, where=self.ivar > 0)
            # Zero ivar for any masked pixels.
            self.ivar[self.pixel_mask[name]] = 0
            self.unit = 'elec'
        else:
            self.unit = 'ADU'

    def get_dark_current(self, ccdtemp=None, exptime=None, method='linear', name=None, retval='image'):
        """Calculate the predicted dark current as a scaled master dark image.

        Parameters
        ----------
        ccdtemp : float or array or None
            The CCD temperature to subtract in degC, normally taken from the GCCDTEMP FITS
            header keyword.  If multiple exposures are loaded, can be an array or constant.
            The value None is only allowed whem method is 'decorrelate'.
        exptime : float or array or None
            The exposure time in seconds, normally taken from the EXPTIME FITS header
            keyword.  If multiple exposures are loaded, can be an array or constant.
            The value None is only allowed whem method is 'decorrelate'.
        method : 'linear' or 'exp' or 'decorrelate'
            When 'decorrelate', determine the effective integration time at 11C by setting
            the weighted correlation of the data with the master dark to zero.  This method
            does not require any input temperature or exposure time but does require that
            some raw data has already been loaded with :meth:`setraw`. Otherwise, use the
            fitted linear or exponential (Arrhenius) model to correct for temperature at the
            specified exposure time. These methods require that ``ccdtemp`` and ``exptime``
            values are provided, but do not require (or use) any previously loaded raw data.
        name : str or None
            Assume the specified camera. When None, use the name specified for the most
            recent call to :meth:`setraw`.
        retval : 'image' or 'frac'
            Returns the dark current images in electrons for each exposure as a 3D array
            for 'image', or the corresponding fractions of the master dark image when 'frac'.
            These fractions can be interpreted as the effective integration time in
            seconds for the dark current at TREF (nominally 11C).

        Returns
        -------
        array
            3D array of predicted dark current in electrons with shape (nexp, ny, nx).
        """
        # Look up the temperature model coefficients for this camera.
        name = name or self.name
        if name not in self.gfa_names:
            raise RuntimeError('Cannot subtract dark current from unknown camera: "{0}".'.format(name))
        master = self.master_dark[name]
        calib = self.calib_data[self.name]
        # Calculate the predicted and reference average dark currents in elec/s.
        if method == 'linear':
            # The IREF parameter cancels in the ratio.
            TCOEF, TREF = calib['TCOEF'], calib['TREF']
            ratio = 1 + TCOEF * (ccdtemp - TREF)
            frac = exptime * ratio
        elif method == 'exp':
            # The I0 parameter cancels in the ratio.
            C0, TREF = calib['C0'], calib['TREF']
            ratio = np.exp(-C0 / (ccdtemp + 273.15)) / np.exp(-C0 / (TREF + 273.15))
            frac = exptime * ratio
        elif method == 'decorrelate':
            # Calculate the fraction of the template to subtract in order to
            # achieve zero weighted corelation with the template.
            T = (self.ivar *  master).reshape(self.nexp, -1)
            T /= np.sum(T ** 2, axis=1, keepdims=True)
            WD = (self.data * self.ivar).reshape(self.nexp, -1)
            frac = np.sum(WD * T, axis=1)
        else:
            raise ValueError('Invalid method "{0}".'.format(method))
        if retval == 'image':
            return master * frac
        elif retval == 'frac':
            return frac
        else:
            raise ValueError('Invalid retval "{0}".'.format(retval))

    def get_psfs(self, D=None, W=None, downsampling=2, margin=16, minsnr=2.0,
                 min_snr_ratio=0.1, maxsrc=29, stack=True):
        """Find PSF candidates in our ``data`` image.

        For best results, estimate and subtract the dark current before calling this method.
        """
        stampsize, inset = self.psf_stampsize, self.psf_inset
        if self.psf_centering is None or (
            self.psf_centering.stamp_size != stampsize or self.psf_centering.inset != inset):
            self.psf_centering = desietc.util.CenteredStamp(stampsize, inset, method='fiber')
        if D is None:
            D = self.data
        if W is None:
            W = self.ivar
        ny, nx = D.shape
        SNR = desietc.util.get_significance(D, W, downsampling=downsampling)
        M = GFASourceMeasure(
            D, W, margin, ny - margin, margin, nx - margin,
            stampsize=stampsize, downsampling=downsampling, centering=self.psf_centering)
        self.psfs = desietc.util.detect_sources(
            SNR, measure=M, minsnr=minsnr, minsep=0.7 * stampsize / downsampling, maxsrc=maxsrc,
            min_snr_ratio=min_snr_ratio)
        if stack:
            self.psf_stack = desietc.util.get_stacked(self.psfs, maxdither=self.maxdither)
        else:
            self.psf_stack = None
        return len(self.psfs)

    def get_donuts(self, downsampling=2, margin=16, minsnr=1.5,
                   min_snr_ratio=0.1, maxsrc=19, column_cut=920, stack=True):
        """Find donut candidates in our ``data`` image.

        For best results, estimate and subtract the dark current before calling this method.
        """
        stampsize, inset = self.donut_stampsize, self.donut_inset
        if self.donut_centering is None or (
            self.donut_centering.stamp_size != stampsize or self.donut_centering.inset != inset):
            self.donut_centering = desietc.util.CenteredStamp(stampsize, inset, method='donut')
        D, W = self.data, self.ivar
        ny, nx = D.shape
        # Compute a single SNR image to use for both halves.
        SNR = desietc.util.get_significance(D, W, downsampling=downsampling)
        # Configure the measurements for each half.
        args = dict(stampsize=stampsize, downsampling=downsampling, centering=self.donut_centering)
        ML = GFASourceMeasure(D, W, margin, ny - margin, margin, column_cut, **args)
        MR = GFASourceMeasure(D, W, margin, ny - margin, nx - column_cut, nx - margin, **args)
        # Configure and run the source detection for each half.
        args = dict(minsnr=minsnr, minsep=0.7 * stampsize / downsampling, maxsrc=maxsrc,
                    min_snr_ratio=min_snr_ratio)
        self.donuts = (
            desietc.util.detect_sources(SNR, measure=ML, **args),
            desietc.util.detect_sources(SNR, measure=MR, **args))
        if stack:
            self.donut_stack = (
                desietc.util.get_stacked(self.donuts[0], maxdither=self.maxdither),
                desietc.util.get_stacked(self.donuts[1], maxdither=self.maxdither))
        else:
            self.donut_stack = None
        return len(self.donuts[0]), len(self.donuts[1])


class GFASourceMeasure(object):
    """Measure candidate sources in D[y1:y2, x1:x2]
    """
    def __init__(self, D, W, y1=0, y2=None, x1=0, x2=None, stampsize=45,
                 downsampling=2, maxsaturated=3, saturation=1e5, bgmargin=4, centering=None):
        assert stampsize % 2 == 1
        self.rsize = stampsize // 2
        self.downsampling = downsampling
        self.D = D
        self.W = W
        self.maxsaturated = maxsaturated
        self.saturation = saturation
        ny, nx = self.D.shape
        self.y1, self.y2 = y1, y2 or ny
        self.x1, self.x2 = x1, x2 or nx
        self.centering = centering

    def __call__(self, snrtot, xc, yc, yslice, xslice):
        # Calculate the center of the input slice.
        xc = 0.5 * (xslice.start + xslice.stop - 1)
        yc = 0.5 * (yslice.start + yslice.stop - 1)
        # Build a fixed-size stamp with this center.
        ix = int(round(self.downsampling * xc))
        if (ix < self.x1 + self.rsize) or (ix >= self.x2 - self.rsize):
            return None
        iy = int(round(self.downsampling * yc))
        if (iy < self.y1 + self.rsize) or (iy >= self.y2 - self.rsize):
            return None
        xslice = slice(ix - self.rsize, ix + self.rsize + 1)
        yslice = slice(iy - self.rsize, iy + self.rsize + 1)
        # Extract and copy the stamp data.
        d = self.D[yslice, xslice].copy()
        w = self.W[yslice, xslice].copy()
        # Count saturated pixels in this stamp.
        if self.saturation is not None:
            saturated = (d > self.saturation) & (w > 0)
            nsaturated = np.count_nonzero(saturated)
            if nsaturated > self.maxsaturated:
                return None
            w[saturated] = 0
        # Estimate and subtract the background.
        d -= desietc.util.estimate_bg(d, w)
        # Find the best centered inset stamp.
        yinset, xinset = self.centering.center(d, w)
        d, w = d[yinset, xinset], w[yinset, xinset]
        yslice = slice(yslice.start + yinset.start, yslice.start + yinset.stop)
        xslice = slice(xslice.start + xinset.start, xslice.start + xinset.stop)
        return (yslice, xslice, d, w)


# CS5 coordinates (mm) at the center of pixel (x,y)=[0,0] and the CS5 displacements
# corresponding to [1,0]-[0,0] and [0,1]-[0,0] in each GFA, determined using
# desimeter.transform.gfa2fp.gfa2fp.
CS5 = {
    'GUIDE0': np.array([
        [ 9.18450162e+01, -3.97013561e+02],
        [ 1.42573799e-02,  4.66981235e-03],
        [-4.65492774e-03,  1.41892856e-02]]),
    'GUIDE2': np.array([
        [ 4.05753952e+02, -3.52939152e+01],
        [ 4.31199162e-05,  1.50010651e-02],
        [-1.49338639e-02,  3.37050638e-05]]),
    'GUIDE3': np.array([
        [ 3.48896107e+02,  2.10237056e+02],
        [-8.86064348e-03,  1.21055081e-02],
        [-1.20477346e-02, -8.82590456e-03]]),
    'GUIDE5': np.array([
        [-9.18888483e+01,  3.96855686e+02],
        [-1.42618407e-02, -4.65682022e-03],
        [ 4.64286359e-03, -1.41935916e-02]]),
    'GUIDE7': np.array([
        [-4.05816895e+02,  3.54535663e+01],
        [-1.00846756e-05, -1.50027231e-02],
        [ 1.49362446e-02, -1.36688551e-06]]),
    'GUIDE8': np.array([
        [-3.49403148e+02, -2.09911349e+02],
        [ 8.75839914e-03, -1.21795768e-02],
        [ 1.21237342e-02,  8.72473620e-03]])
}

def gfa_to_cs5(x_gfa, y_gfa, camera):
    """Convert GFA pixel coordinates to CS5 coordinates in mm.

    Parameters
    ----------
    x_gfa : float or 1-d array
        GFA pixel coordinates where 0 refers to the center of the corner pixel.
    y_gfa : float or 1-d array
        GFA pixel coordinates where 0 refers to the center of the corner pixel.
    camera : str
        Name of the in-focus GFA to perform the conversion for.

    Returns
    -------
    array
        CS5 coordinates in mm. Will have shape (2,) if the inputs are scalars,
        or (N,2) if the inputs are 1-d arrays of length N.
    """
    scalar = np.isscalar(x_gfa)
    x_gfa = np.atleast_1d(x_gfa)
    y_gfa = np.atleast_1d(y_gfa)
    if x_gfa.shape != y_gfa.shape:
        raise ValueError('x,y arrays do not have the same shape.')
    if x_gfa.ndim != 1:
        raise ValueError('x,y arrays must be 1d.')
    XY = np.stack((np.ones_like(x_gfa), x_gfa, y_gfa), axis=1)
    result = XY.dot(CS5[camera])
    return result[0] if scalar else result
