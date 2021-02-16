"""Numerical utilities for the online exposure-time calculator.

The general guideline for things implemented here is that they
do not read/write any files or produce any logging output.
"""
import datetime
import json

import numpy as np

import scipy.ndimage


def fit_spots(data, ivar, profile, area=1):
    """Fit images of a spot to estimate the spot flux and background level.

    All inputs are nominally 2D but can have other shapes as long as
    they broadcast correctly. Input arrays with >2 dimensions are assumed
    to have the pixels indexed along their last 2 dimensions.

    Parameters
    ----------
    data : array
        Array of shape (...,ny,nx) with the data to fit.
    ivar : array
        Array of shape (...,ny,nx) with the corresponding ivars.
    profile : array
        Array of shape (...,ny,nx) with the spot profile(s) to use.
    area : scalar or array
        Area of each pixel used to predict its background level as b * area.
        Either a scalar or an array of shape (...,ny, nx).

    Returns
    -------
    tuple
        Tuple (f, b, cov) where f and b are arrays of shape (...) and
        cov has shape (...,2,2) with elements [...,0,0] = var(f),
        [...,1,1] = var(b) and [...,0,1] = [...,1,0] = cov(f,b).
    """
    # Calculate the matrix elements for the linear problem
    # [ M11 M12 ] [ f ] = [ A1 ]
    # [ M12 M22 ] [ b ]   [ A2 ]
    M11 = np.sum(ivar * profile ** 2, axis=(-2, -1))
    M12 = np.sum(ivar * area * profile, axis=(-2, -1))
    M22 = np.sum(ivar * area ** 2, axis=(-2, -1))
    A1 = np.sum(ivar * data * profile, axis=(-2, -1))
    A2 = np.sum(ivar * data * area, axis=(-2, -1))
    # Solve the linear problem.
    det = M11 * M22 - M12 ** 2
    M11 /= det
    M12 /= det
    M22 /= det
    f = (M22 * A1 - M12 * A2)
    b = (M11 * A2 - M12 * A1)
    # Calculate the covariance of (f, b).
    cov = np.stack((np.stack((M22, -M12), axis=-1), np.stack((-M12, M11), axis=-1)), axis=-1)
    return f, b, cov


def get_significance(D, W, smoothing=2.5, downsampling=2, medfiltsize=5):
    """Calculate a downsampled pixel significance image.

    This function is a quick and robust way to calculate a significance
    image suitable for thresholding to identify regions likely to contain
    a source.  There are three stages:

      - Apply a weighted Gaussian filter,
      - Perform a weighted downsampling,
      - Estimate and subtract a background image using a median filter.

    This is designed to work with :func:`detect_sources`.

    Parameters
    ----------
    D : array
        2D array of pixel values.
    W : array
        2D array of corresponding ivar weights with same shape as D.
    smoothing : float
        Gaussian smoothing sigma to apply in pixels before downsampling.
    downsampling : int
        Downsampling factor to apply. Must evenly divide both dimensions of D.
    medfiltsize : int
        Size of median filter to apply after after downsampling to estimate
        the smoothly varying background. Must be odd.

    Returns
    -------
    array
        2D array of downsampled pixel significance values. Note that the
        returned dimensions are downsampled relative to the input arrays.
    """
    # Apply weighted smoothing.
    D, W = smooth(D, W, smoothing)
    # Downsample.
    D, W = downsample_weighted(D, W, downsampling=downsampling, allow_trim=False)
    # Median filter the data to estimate background variations.
    mask = W == 0
    D[mask] = np.median(D[~mask])
    Dm = scipy.ndimage.median_filter(D, medfiltsize)
    # Subtract the median-filtered image.
    D -= Dm
    # Estimate the significance of each (downsampled) pixel.
    return D * np.sqrt(W)


def detect_sources(snr, minsnr=4, minsize=8, maxsize=32, minsep=0,
                   min_snr_ratio=0.1, maxsrc=20, measure=None):
    """Detect and measure sources in a significance image.

    A source is defined as a connected and isolated region of pixels above
    some threshold that fits within a square bounding box with a size in
    the range ``minsize`` to ``maxsize`` pixels.

    When ``measure`` is None, the ``maxsrc`` sources with the
    highest total SNR are returned with their total SNR and centroid
    coordinates measured.  When a callable ``measure`` is supplied, it
    is passed the total SNR and centroid coordinates and can either return
    None to reject a source, or return an updated set of measurements.

    Parameters
    ----------
    snr : array
        2D image of pixel significances, e.g., from :func:`get_significance`.
    minsnr : float
        All pixels above this threshold will be assigned to a potential
        source.
    minsize : int
        Minimum square bounding box size for a source.
    maxsize : int
        Maximum square bounding box size for a source.
    minsep : float
        Minimum distance between any pair of detected sources in pixels.
        Distances are measured between the SNR ** 2 weighted centers of
        gravity of each source candidate.
    maxsrc : int
        Maximum number of measured sources to return.
    measure : callable or None
        Optional function that is passed the total SNR and centroid coordinates
        of a candidate source and either returns None to reject the source or
        an updated set of measurements.

    Returns
    -------
    list
        A list of the measurements for each detected source.
    """
    if minsize > maxsize:
        raise ValueError('Expected minsize <= maxsize.')
    ny, nx = snr.shape
    # Label all non-overlapping regions above SNRmin in the inset image.
    labeled, nlabels = scipy.ndimage.label(snr > minsnr)
    if nlabels == 0:
        return []
    labels = np.arange(1, nlabels + 1)
    # Calculate bounding boxes for each candidate source.
    bboxes = scipy.ndimage.find_objects(labeled)
    # Estimate the quadrature summed SNR for each candidate source.
    snrtot = scipy.ndimage.labeled_comprehension(
        snr, labeled, labels, out_dtype=float, default=-1,
        func=lambda X: np.sqrt(np.sum(X ** 2)))
    maxsnrtot = None
    # Rank sources by snrtot.
    ranks = np.argsort(snrtot)[::-1]
    # Build the final list of detected sources.
    sources = []
    snrsq = snr ** 2
    minsepsq = minsep ** 2
    centroids = np.empty((maxsrc, 2))
    for idx in range(nlabels):
        label = labels[ranks[idx]]
        srcsnrtot = snrtot[label - 1]
        if maxsnrtot is not None and srcsnrtot < min_snr_ratio * maxsnrtot:
            break
        # Lookup this source's bounding box.
        yslice, xslice = bboxes[label - 1]
        size = max(yslice.stop - yslice.start, xslice.stop - xslice.start)
        if size < minsize or size > maxsize:
            continue
        # Calculate the SNR**2 weighted center of mass for this source.
        yc, xc = scipy.ndimage.center_of_mass(snrsq, labeled, label)
        nsrc = len(sources)
        if nsrc > 0 and minsep > 0:
            # Calculate the distance to each previous source.
            rsq = np.sum((centroids[:nsrc] - np.array([xc, yc])) ** 2, axis=1)
            if np.any(rsq < minsepsq):
                continue
        params = (srcsnrtot, xc, yc, yslice, xslice)
        if measure is not None:
            params = measure(*params)
            if params is None:
                continue
        centroids[nsrc] = (xc, yc)
        if maxsnrtot is None:
            maxsnrtot = srcsnrtot
        sources.append(params)
        if len(sources) == maxsrc:
            break
    return sources


def estimate_bg(D, W, margin=4, maxchisq=2, minbgfrac=0.2):
    """Estimate the background level from the margins of an image.

    Parameters
    ----------
    D : array
        2D array of pixel values.
    W : array
        2D array of corresponding inverse variances.
    margin : int
        Size of margin around the outside of the image to use to
        estimate the background.
    maxchisq : float
        Maximum pixel chi-square value to consider a margin
        pixel as background like.
    minbgfrac : float
        Minimum fraction of background-like margin pixels required
        to use a weighted mean value estimate.  Otherwise, a
        noisier but more robust median of unmasked margin pixel
        values is returned.

    Returns
    -------
    float
        Estimate of the background level. Will be zero if all
        margin pixels are masked.
    """
    mask = np.zeros(D.shape, bool)
    mask[:margin] = mask[-margin:] = True
    mask[:, :margin] = mask[:, -margin:] = True
    # Find the median unmasked pixel value in the margin.
    d = D[margin]
    w = W[margin]
    if not np.any(w > 0):
        # There are no unmasked margin pixels.
        return 0
    med = np.median(d[w > 0])
    # Find the median unmasked ivar in the margin.
    sig = 1 / np.sqrt(np.median(w[w > 0]))
    # Select bg-like pixels in the margin.
    chisq = w * (d - med) ** 2
    bg = (chisq < maxchisq) & (w > 0)
    if np.count_nonzero(bg) < minbgfrac * d.size:
        # Return the median when there are not enough bg pixels.
        return med
    else:
        # Calculate a weighted mean of the bg pixels.
        return np.sum(w[bg] * d[bg]) / np.sum(w[bg])


def make_template(size, profile, dx=0, dy=0, oversampling=10, normalized=True):
    """Build a square template for an arbitrary profile.

    Parameters
    ----------
    size : int
        Output 2D array will have shape (size, size).
    profile : callable
        Function of (x,y) that evaluates the profile to use, where x and y are arrays
        of pixel coordinates relative to the template center. This function is called
        once, instead of iterating over pixels, so should broadcast over x and y.
    dx : float
        Offset values of x passed to the profile by this amount (in pixels).
    dy : float
        Offset values of y passed to the profile by this amount (in pixels).
    oversampling : int
        Integrate over the template pixels by working on a finer grid with this
        oversampling factor, then downsample to the output pixels.
    normalized : bool
        When True, the sum of output pixels is normalized to one.

    Returns
    -------
    array
        2D numpy array of template pixel values with shape (size, size).
    """
    xy = (np.arange(size * oversampling) - 0.5 * (size * oversampling - 1)) / oversampling
    z = profile(xy - dx, (xy - dy).reshape(-1, 1))
    T = downsample(z, oversampling, np.mean)
    if normalized:
        T /= T.sum()
    return T


def downsample(data, downsampling, summary=np.sum, allow_trim=False):
    """Downsample a 2D array.

    Parameters
    ----------
    data : array
        Two dimensional array of values to downsample.
    downsampling : int
        Downsampling factor to use along both dimensions. Must evenly divide the
        data dimensions when allow_trim is False.
    summary : callable
        The summary function to use that will be applied to each block of shape
        (dowsampling, downsampling) to obtain the output downsampled values.
        Must support broadcasting and an axis parameter. Useful choices are
        np.sum, np.mean, np.min, np.max, np.median, np.var but any ufunc
        should work.
    allow_trim : bool
        When False, the input dimensions (ny, nx) must both exactly divide
        the downsampling value.  Otherwise, any extra rows and columns are
        silently trimmed before apply the summary function.

    Returns
    -------
    array
        A two dimensional array of shape (ny // downsampling, nx // downsampling)
        where the input data shape is (ny, nx).
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError('Data must be 2 dimensional.')
    ny, nx = data.shape
    if not allow_trim and ((nx % downsampling) or (ny % downsampling)):
        raise ValueError('Data shape {0} does not evenly divide downsampling={1} and allow_trim is False.'
                         .format((ny, nx), downsampling))
    ny //= downsampling
    nx //= downsampling
    shape = (ny, nx, downsampling, downsampling)
    strides = (downsampling * data.strides[0], downsampling * data.strides[1]) + data.strides
    blocks = np.lib.stride_tricks.as_strided(
        data[:downsampling * ny, :downsampling * nx], shape=shape, strides=strides)
    return summary(blocks, axis=(2, 3))


def downsample_weighted(D, W, downsampling=4, allow_trim=True):
    """Downsample 2D data D with weights W.
    """
    if D.shape != W.shape:
        raise ValueError('Arrays D, W must have the same shape.')
    if D.ndim != 2:
        raise ValueError('Arrays D, W must be 2D.')
    if np.any(W < 0):
        raise ValueError('Array W contains negative values.')
    WD = downsample(D * W, downsampling=downsampling, summary=np.sum, allow_trim=allow_trim)
    W = downsample(W, downsampling=downsampling, summary=np.sum, allow_trim=allow_trim)
    D = np.divide(WD, W, out=np.zeros_like(WD), where=W > 0)
    return D, W


def smooth(D, W, smoothing):
    """Apply a weighted Gaussian smoothing.
    """
    WD = scipy.ndimage.gaussian_filter(W * D, smoothing)
    W = scipy.ndimage.gaussian_filter(W, smoothing)
    D = np.divide(WD, W, out=np.zeros_like(D), where=W > 0)
    return D, W


def get_stacked(stamps, smoothing=1, maxdither=1, maxdist=3, min_stack=2):
    """Calculate a stack of detected sources ignoring outliers.
    """
    # Extract and normalize stamps.
    nstamps = len(stamps)
    if nstamps == 0:
        return None, None
    stamps = [normalize_stamp(*S[2:4]) for S in stamps]
    ny, nx = (stamps[0][0]).shape
    # Calculate distance matrix and record best dithers and scales.
    dist = np.zeros((nstamps, nstamps))
    dither = np.zeros((nstamps, nstamps, 2), int)
    fscale = np.ones((nstamps, nstamps))
    for j in range(nstamps):
        D1, W1 = stamps[j]
        for i in range(j + 1, nstamps):
            D2, W2 = stamps[i]
            dist_ji, dither_ji, fscale_ji, _ = get_stamp_distance(
                D1, W1, D2, W2, maxdither=maxdither, smoothing=smoothing)
            dist[i, j] = dist[j, i] = dist_ji
            dither[j, i] = dither_ji
            dither[i, j] = -dither_ji
            fscale[j, i] = fscale_ji
            fscale[i, j] = 1 / fscale_ji
    # Find the medioid stamp.
    totdist = dist.sum(axis=1)
    imed = np.argmin(totdist)
    # How many other stamps are close enough to stack?
    stack_idx = np.where(dist[imed] < maxdist)[0]
    if len(stack_idx) < min_stack:
        # Calculate and return the weighted average stamp.
        DWsum = np.sum(np.stack([D * W for D, W in stamps]), axis=0)
        Wavg = np.sum(np.stack([W for D, W in stamps]), axis=0)
        Davg = np.divide(DWsum, Wavg, out=np.zeros_like(DWsum), where=Wavg > 0)
        inset = slice(maxdither, ny - maxdither), slice(maxdither, nx - maxdither)
        return normalize_stamp(Davg[inset], Wavg[inset])
    # Calculate the final stack.
    ndither = 2 * maxdither + 1
    DWstack = np.zeros((ny - 2 * maxdither, nx - 2 * maxdither))
    Wstack = np.zeros_like(DWstack)
    for j in stack_idx:
        D, W = stamps[j]
        dy, dx = dither[imed, j]
        f = fscale[imed, j]
        inset_j = slice(maxdither + dy, ny - maxdither + dy), slice(maxdither + dx, nx - maxdither + dx)
        Dj, Wj = f * D[inset_j], W[inset_j] / f ** 2
        DWstack += Dj * Wj
        Wstack += Wj
    Dstack = np.divide(DWstack, Wstack, out=np.zeros_like(DWstack), where=Wstack > 0)
    return normalize_stamp(Dstack, Wstack)


def get_stamp_distance(D1, W1, D2, W2, maxdither=3, smoothing=1, fscale=np.linspace(0.85, 1.15, 11)):
    """Calculate the minimum chisq distance between two stamps allowing for some dither.
    """
    ny, nx = D1.shape
    assert D1.shape == D2.shape == W1.shape == W2.shape
    nscale = len(fscale)
    fvec = fscale.reshape(-1, 1, 1)
    # Smooth both stamps.
    D1, W1 = smooth(D1, W1, smoothing)
    D2, W2 = smooth(D2, W2, smoothing)
    # Inset the first stamp by the dither size.
    inset = slice(maxdither, ny - maxdither), slice(maxdither, nx - maxdither)
    D1inset = D1[inset]
    W1inset = W1[inset]
    # Loop over dithers of the second stamp.
    ndither = 2 * maxdither + 1
    pull = np.zeros((ndither, ndither, nscale, ny - 2 * maxdither, nx - 2 * maxdither))
    dxy = np.arange(-maxdither, maxdither + 1)
    for iy, dy in enumerate(dxy):
        for ix, dx in enumerate(dxy):
            # Dither the second stamp.
            D2inset = D2[maxdither + dy:ny - maxdither + dy, maxdither + dx:nx - maxdither + dx]
            W2inset = W2[maxdither + dy:ny - maxdither + dy, maxdither + dx:nx - maxdither + dx]
            # Calculate the chi-square distance between the inset stamps with scale factors of
            # 1/fvec and fvec applied to (D1,W1) and (D2,W2) respectively.
            num = np.sqrt(W1inset * W2inset) * (D1inset / fvec - D2inset * fvec)
            denom = np.sqrt(W1inset * fvec ** 2 + W2inset * fvec ** -2)
            # Could also use where=(num > 0) here.
            pull[iy, ix] = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    # Find the dither with the smallest chisq.
    chisq = np.sum(pull ** 2, axis=(3, 4))
    iy, ix, iscale = np.unravel_index(np.argmin(chisq.reshape(-1)), (ndither, ndither, nscale))
    assert chisq.min() == chisq[iy, ix, iscale]
    # Return the smallest distance, the corresponding dither and scale, and the best pull image.
    return (chisq[iy, ix, iscale] / D1inset.size, np.array((dxy[iy], dxy[ix]), int),
            fscale[iscale], pull[iy, ix, iscale].copy())


def normalize_stamp(D, W, smoothing=2.5):
    """Normalize a stamp to its weighted mean value.
    Should generally subtract a background estimate first.
    """
    smoothed, _ = smooth(D, W, smoothing)
    norm = smoothed.sum()
    if norm != 0:
        return D / np.abs(norm), W * norm ** 2


def diskgrid(n, radius=1, alpha=2):
    """Distribute points over a disk with increasing density towards the center.

    Points are locally uniformly distributed according to the sunflower pattern
    https://demonstrations.wolfram.com/SunflowerSeedArrangements/

    A non-linear transformation of the radial coordinate controlled by alpha
    increases the density of points towards the center. Use alpha=0 for
    uniform density.

    Parameters
    ----------
    n : int
        Total number of points to use in the grid.
    radius : float
        Radius of the disk to fill.
    alpha : float
        Parameter controlling the increase of density towards the center of
        the disk, with alpha=0 corresponding to no increase.

    Returns
    -------
    tuple
        Tuple (x, y) of 2D points covering the disk.
    """
    # Golden ratio.
    phi = 0.5 * (np.sqrt(5) + 1)
    # Calculate coordinates of each point to uniformly fill the unit disk.
    k = np.arange(1, n + 1)
    theta = 2 * np.pi * k / phi ** 2
    r = np.sqrt((k - 0.5) / (n - 0.5))
    # Transform r to increase the density towards the center.
    if alpha > 0:
        r = (np.exp(alpha * r) - 1) / (np.exp(alpha) - 1)
    r *= radius
    return r * np.cos(theta), r * np.sin(theta)


def make_template(size, profile, dx=0, dy=0, oversampling=10, normalized=True):
    """Build a square template for an arbitrary profile.

    Parameters
    ----------
    size : int
        Output 2D array will have shape (size, size).
    profile : callable
        Function of (x,y) that evaluates the profile to use, where x and y are arrays
        of pixel coordinates relative to the template center. This function is called
        once, instead of iterating over pixels, so should broadcast over x and y.
    dx : float
        Offset values of x passed to the profile by this amount (in pixels).
    dy : float
        Offset values of y passed to the profile by this amount (in pixels).
    oversampling : int
        Integrate over the template pixels by working on a finer grid with this
        oversampling factor, then downsample to the output pixels.
    normalized : bool
        When True, the sum of output pixels is normalized to one.

    Returns
    -------
    array
        2D numpy array of template pixel values with shape (size, size).
    """
    xy = (np.arange(size * oversampling) - 0.5 * (size * oversampling - 1)) / oversampling
    z = profile(xy - dx, (xy - dy).reshape(-1, 1))
    T = downsample(z, oversampling, np.mean)
    if normalized:
        T /= T.sum()
    return T


def preprocess(D, W, nsig_lo=10, nsig_hi=30, vmin=None, vmax=None):
    """Preprocess weighted 2D array data for display.
    """
    masked = W == 0
    # Calculate the median unmasked pixel value.
    median_value = np.median(D[~masked])
    # Calculate the median non-zero inverse variance.
    median_ivar = np.median(W[~masked])
    # Calculate the corresponding pixel sigma.
    sigma = 1 / np.sqrt(median_ivar)
    if vmin is None:
        vmin = median_value - nsig_lo * sigma
    if vmax is None:
        vmax = median_value + nsig_hi * sigma
    # Clip values to [vmin, vmax].
    D = np.clip(D, vmin, vmax)
    # Set masked pixel values to nan so they are not plotted.
    D[masked] = np.nan
    return D


def ADCangles(EL, HA, DEC, LAT=31.963972222):
    """Calculate the parallactic angle in degrees W of N. Inputs in degrees."""
    Z, HA, coDEC, coLAT = np.deg2rad([90 - EL, HA, 90 - DEC, 90 - LAT])
    if Z == 0:
        return np.zeros(3)
    sinZ = np.sin(Z)
    sinP = np.sin(HA) * np.sin(coLAT) / sinZ
    cosP = (np.cos(coLAT) - np.cos(coDEC) * np.cos(Z)) / (np.sin(coDEC) * sinZ)
    P = np.arctan2(sinP, cosP)
    # Formulas from DESI-4957
    tanZ = np.tan(Z)
    HORIZON = P + 0.5 * np.pi
    ADC1 = HORIZON + (0.0353 + tanZ * (0.2620 + tanZ * 0.3563))
    ADC2 = HORIZON - (0.0404 + tanZ * (0.2565 + tanZ * 0.3576))
    return np.rad2deg([P, ADC1, ADC2])


class PSFMeasure(object):

    def __init__(self, stamp_size, fiber_diam_um=107, pixel_size_um=15, plate_scales=(70., 76.),
                 max_offset_pix=3.5, noffset=15, nangbins=20):
        self.stamp_size = stamp_size
        self.pixel_size_um = pixel_size_um
        self.plate_scales = plate_scales
        # Tabulate fiber templates for each (x,y) offset in the x >= 0 and y >= 0 quadrant.
        self.offset_template = np.empty((noffset, noffset, stamp_size, stamp_size), np.float32)
        max_rsq = (0.5 * fiber_diam_um / pixel_size_um) ** 2
        profile = lambda x, y: 1.0 * (x ** 2 + y ** 2 < max_rsq)
        delta = np.linspace(0, max_offset_pix, noffset)
        for iy in range(noffset):
            for ix in range(noffset):
                self.offset_template[iy, ix] = make_template(
                    stamp_size, profile, dx=delta[ix], dy=delta[iy], normalized=False)
        self.xyoffset = np.linspace(-max_offset_pix, +max_offset_pix, 2 * noffset - 1)
        dxy = np.arange(self.stamp_size) - 0.5 * (self.stamp_size - 1)
        self.xgrid, self.ygrid = np.meshgrid(dxy, dxy, sparse=False)
        rmax = dxy[-1] * self.pixel_size_um / max(self.plate_scales)
        self.angbins = np.linspace(0., rmax, nangbins + 1)
        self.rang = 0.5 * (self.angbins[1:] + self.angbins[:-1])

    def measure(self, P, W):
        assert P.shape == W.shape == (self.stamp_size, self.stamp_size)
        Psum = np.sum(P)
        # Prepare the array of fiber fractions for offsets in all 4 quadrants.
        nquad = len(self.offset_template)
        nfull = 2 * nquad - 1
        fiberfrac = np.zeros((nfull, nfull), np.float32)
        # Loop over offsets in the x >= 0 and y >= 0 quadrant.
        origin = nquad - 1
        reverse = slice(None, None, -1)
        for iy in range(nquad):
            for ix in range(nquad):
                T = self.offset_template[iy, ix]
                fiberfrac[origin + iy, origin + ix] = np.sum(P * T) / Psum
                if iy > 0:
                    # Calculate in the x >= 0 and y < 0 quadrant.
                    fiberfrac[origin - iy, origin + ix] = np.sum(P * T[reverse, :]) / Psum
                if ix > 0:
                    # Calculate in the x < 0 and y >= 0 quadrant.
                    fiberfrac[origin + iy, origin - ix] = np.sum(P * T[:, reverse]) / Psum
                if iy > 0 and ix > 0:
                    # Calculate in the x < 0 and y < 0 quadrant.
                    fiberfrac[origin - iy, origin - ix] = np.sum(P * T[reverse, reverse]) / Psum
        # Locate the best centered offset.
        iy, ix = np.unravel_index(np.argmax(fiberfrac), fiberfrac.shape)
        xc = self.xyoffset[ix]
        yc = self.xyoffset[iy]
        # Calculate the radius of each pixel in arcsecs relative to this center.
        radius = np.hypot((self.xgrid - xc) * self.pixel_size_um / self.plate_scales[0],
                          (self.ygrid - yc) * self.pixel_size_um / self.plate_scales[1]).reshape(-1)
        # Fill ivar-weighted histograms of flux versus angular radius.
        WZ, _ = np.histogram(radius, bins=self.angbins, weights=(P * W).reshape(-1))
        W, _ = np.histogram(radius, bins=self.angbins, weights=W.reshape(-1))
        # Calculate the circularized profile, normalized to 1 at (xc, yc).
        Z = np.divide(WZ, W, out=np.zeros_like(W), where=W > 0)
        fwhm = -1
        if Z[0] > 0:
            Z /= Z[0]
            # Find the first bin where Z <= 0.5.
            k = np.argmax(Z <= 0.5)
            if k > 0:
                # Use linear interpolation over this bin to estimate FWHM.
                s = (Z[k] - 0.5) / (Z[k] - Z[k - 1])
                fwhm = 2 * ((1 - s) * self.rang[k] + s * self.rang[k - 1])
        self.Z = Z
        self.xcbest = xc
        self.ycbest = yc
        return fwhm, np.max(fiberfrac)


class CenteredStamp(object):

    def __init__(self, stamp_size, inset, method='fiber'):
        self.inset = inset
        self.stamp_size = stamp_size
        self.inset_size = stamp_size - 2 * inset
        # Calculate the range of offsets to explore.
        self.dxy = np.arange(-inset, inset + 1)
        noffset = len(self.dxy)
        # Allocate memory for the templates to use.
        self.template = np.zeros((noffset, noffset, stamp_size, stamp_size))
        # Define the template profile.
        rfibersq = (0.5 * 107 / 15) ** 2
        if method == 'fiber':
            profile = lambda x, y: 1.0 * (x ** 2 + y ** 2 < rfibersq)
        elif method == 'donut':
            xymax = 0.5 * (stamp_size - 2 * inset)
            def profile(x, y):
                rsq = x ** 2 + y ** 2
                return rsq * np.exp(-0.5 * rsq / (3 * rfibersq)) * (np.maximum(np.abs(x), np.abs(y)) < xymax)
        else:
            raise ValueError('Unsupported method "{0}".'.format(method))
        # Build the profiles.
        for iy in range(noffset):
            for ix in range(noffset):
                self.template[iy, ix] = make_template(
                    stamp_size, profile, dx=self.dxy[ix], dy=self.dxy[iy], normalized=True)

    def center(self, D, W):
        assert D.shape == (self.stamp_size, self.stamp_size) and W.shape == D.shape
        S = slice(self.inset, self.inset + self.inset_size)
        # Calculate the weighted mean template flux for each offset.
        WDsum = np.sum(W * D * self.template, axis=(2, 3))
        Wsum = np.sum(W * self.template, axis=(2, 3))
        meanflux = np.divide(WDsum, Wsum, out=np.zeros(self.template.shape[:2]), where=Wsum > 0)
        # Calculate the best-centered offset
        iy, ix = np.unravel_index(np.argmax(meanflux.reshape(-1)), meanflux.shape)
        yslice = slice(iy, iy + self.inset_size)
        xslice = slice(ix, ix + self.inset_size)
        return yslice, xslice


class MeasurementBuffer(object):
    """Manage a circular buffer of measurements consisting of a time interval, value and error.

    Auxiliary data can optionally be attached to each measurement.
    """
    SECS_PER_DAY = 86400

    def __init__(self, maxlen, default_value, resolution=1, padding=300, recent=300, aux_dtype=[]):
        self.oldest = None
        self.len = 0
        self.full = False
        dtype = [
            ('mjd1', np.float64), ('mjd2', np.float64), ('value', np.float32), ('error', np.float32)
        ] + aux_dtype
        self._entries = np.empty(shape=maxlen, dtype=dtype)
        self.default_value = default_value
        self.resolution = resolution / self.SECS_PER_DAY
        self.padding = padding / self.SECS_PER_DAY
        self.recent = recent / self.SECS_PER_DAY

    @property
    def entries(self):
        return self._entries[:self.len]

    def __str__(self):
        return f'{self.__class__.__name__}(len={self.len}, full={self.full}, oldest={self.oldest})'

    def add(self, mjd1, mjd2, value, error, aux_data=()):
        """Add a single measurement.

        We make no assumption that measurements are non-overlapping or added in time order.
        """
        assert mjd1 < mjd2 and error > 0
        is_oldest = (self.oldest is None) or (mjd1 < self._entries[self.oldest]['mjd1'])
        entry = (mjd1, mjd2, value, error) + aux_data
        if self.full:
            assert self.oldest is not None
            if is_oldest:
                # Ignore this since it is older than all existing entries.
                return
            self._entries[self.oldest] = entry
            # Update the index of the oldest entry, which might be us.
            self.oldest = np.argmin(self.entries['mjd1'])
        else:
            idx = self.len
            if is_oldest:
                # This is now the oldest entry.
                self.oldest = idx
            self.len += 1
            self.full = (self.len == self._entries.size)
            self._entries[idx] = entry

    def inside(self, mjd1, mjd2):
        """Return a mask for entries whose intervals overlap [mjd1, mjd2].

        Use mjd2=None to select all entries after mjd1.
        """
        mask = self.entries['mjd2'] > mjd1
        if mjd2 is not None:
            mask &= self.entries['mjd1'] < mjd2
        return mask

    def sample(self, mjd1, mjd2):
        """Sample measurements on a grid covering (mjd1, mjd2) using linear interpolation.

        The grid spacing will be approximately resolution seconds.
        Use measurements that lie outside the grid up to self.padding seconds.
        Return default_value when no measurements are available.
        Use constant extrapolation of the first/last measurement if necessary.
        """
        assert (mjd2 is not None) and (mjd1 < mjd2)
        # Construct the grid to use.
        ngrid = int(np.ceil((mjd2 - mjd1) / self.resolution))
        mjd_grid = mjd1 + (np.arange(ngrid) + 0.5) * (mjd2 - mjd1) / ngrid
        # Select measurements that span the padded input grid.
        sel = self.inside(mjd1 - self.padding, mjd2 + self.padding)
        if not np.any(sel):
            return mjd_grid, np.full_like(mjd_grid, self.default_value)
        mjd_sel = 0.5 * (self.entries[sel]['mjd1'] + self.entries[sel]['mjd2'])
        value_sel = self.entries[sel]['value']
        iorder = np.argsort(mjd_sel)
        # Use linear interpolation with constant extrapolation beyond the endpoints.
        return mjd_grid, np.interp(mjd_grid, mjd_sel[iorder], value_sel[iorder])

    def trend(self, mjd):
        """Return the linear trend in values over (mjd - recent, mjd).
        For now, this returns a weighted average with zero slope.
        """
        sel = self.inside(mjd - self.recent, mjd)
        if not np.any(sel):
            return self.default_value, 0
        wgt = self.entries[sel]['error'] ** -0.5
        val = self.entries[sel]['value']
        return np.sum(wgt * val) / np.sum(wgt), 0

    def save(self, mjd1, mjd2):
        """Return a json suitable serialization of our entries spanning (mjd1, mjd2).

        Use mjd2=None to use all entries after mjd1.
        If this buffer has auxiliary data, that will saved also.
        Note that we return numpy data types, which are not JSON serializable by
        default, so this assumes that the caller uses :class:`NumpyEncoder` or
        something equivalent.
        """
        sel = self.inside(mjd1, mjd2)
        names = self._entries.dtype.names
        output = []
        for entry in self.entries[sel]:
            row = {}
            for name in names:
                if self._entries.dtype[name].shape != ():
                    row[name] = entry[name].tolist()
                else:
                    row[name] = entry[name]
            output.append(row)
        return output


def mjd_to_date(mjd, utc_offset=-7):
    """Convert an MJD value to a datetime using the specified UTC offset in hours.

    The default utc_offset of -7 corresponds to local time at Kitt Peak.
    Use :func:`date_to_mjd` to invert this calculation.
    """
    return datetime.datetime(2019, 1, 1) + datetime.timedelta(days=mjd - 58484.0, hours=utc_offset)


def date_to_mjd(date, utc_offset=-7):
    """Convert a datetime using the specified UTC offset in hours to an MJD value.

    The default utc_offset of -7 corresponds to local time at Kitt Peak.
    Use :func:`mjd_to_date` to invert this calculation.
    """
    delta = date - datetime.datetime(2019, 1, 1) - datetime.timedelta(hours=utc_offset)
    return 58484 + delta.days + (delta.seconds + 1e-6 * delta.microseconds) / 86400


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder to use with numpy data.
    """
    def default(self, obj):
        if isinstance(obj, np.float32):
            # TODO: round output to 6 decimals
            return float(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


def load_guider_centroids(path, expid):
    """Attempt to read the centroids json file produced by the guider.

    Extracts numbers from the json file into numpy arrays. Note that
    the json file uses "x" for rows and "y" for columns, which we map
    to indices 0 and 1, respectively.

    Returns
    -------
    tuple
        Tuple (expected, combined, centroid) where expected gives the
        expected position of each star with shape (nstars, 2), combined
        gives the combined guider move after each frame with shape (2, nframes),
        and centroid gives the centroid of each star for each frame with
        shape (nstars, 2, nframes). If a star is not measured in a frame,
        the centroid values are np.nan.
    """
    cameras = ('GUIDE0', 'GUIDE2', 'GUIDE3', 'GUIDE5', 'GUIDE7', 'GUIDE8')
    # Read the json file of guider outputs.
    jsonpath = path / 'centroids-{0}.json'.format(expid)
    if not jsonpath.exists():
        raise ValueError('Non-existent path: {0}.'.format(jsonpath))
    with open(jsonpath) as f:
        D = json.load(f)
        assert D['expid'] == int(expid)
        nframes = D['summary']['frames']
    # Use the first frame to lookup the guide stars for each camera.
    frame0 = D['frames']['1']
    stars = {G: len([K for K in frame0.keys() if K.startswith(G)]) for G in cameras}
    expected = {G: np.zeros((stars[G], 2)) for G in cameras}
    combined = {G: np.zeros((2, nframes)) for G in cameras}
    centroid = {G: np.zeros((stars[G], 2, nframes)) for G in cameras}
    for camera in cameras:
        # Get the expected position for each guide star.
        for istar in range(stars[camera]):
            S = frame0.get(camera + f'_{istar}')
            expected[camera][istar, 0] = S['y_expected']
            expected[camera][istar, 1] = S['x_expected']
        # Get the combined centroid sent to the telescope for each frame.
        for iframe in range(nframes):
            F = D['frames'].get(str(iframe + 1))
            if F is None:
                logging.warning('Missing frame {0}/{1} in {2}'.format(iframe + 1, nframes, jsonpath))
                continue
            combined[camera][0, iframe] = F['combined_y']
            combined[camera][1, iframe] = F['combined_x']
            # Get the measured centroids for each guide star in this frame.
            for istar in range(stars[camera]):
                S = F.get(camera + '_{0}'.format(istar))
                centroid[camera][istar, 0, iframe] = S.get('y_centroid', np.nan)
                centroid[camera][istar, 1, iframe] = S.get('x_centroid', np.nan)
    return expected, combined, centroid
