"""Utilities for the online exposure-time calculator.
"""
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
