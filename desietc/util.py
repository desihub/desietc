"""Numerical utilities for the online exposure-time calculator.

The general guideline for things implemented here is that they
do not read/write any files or produce any logging output.
"""

import datetime
import json
import pathlib
import subprocess
import os

import numpy as np

import scipy.ndimage
import scipy.interpolate
from scipy.interpolate import RegularGridInterpolator
import scipy.linalg
import scipy.signal


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
        Tuple (f, b, cov, None) where f and b are arrays of shape (...) and
        cov has shape (...,2,2) with elements [...,0,0] = var(f),
        [...,1,1] = var(b) and [...,0,1] = [...,1,0] = cov(f,b).
        The final None is for compatibility with fit_spots_flux_and_pos().
    """
    # Calculate the matrix elements for the linear problem
    # [ M11 M12 ] [ f ] = [ A1 ]
    # [ M12 M22 ] [ b ]   [ A2 ]
    M11 = np.sum(ivar * profile**2, axis=(-2, -1))
    M12 = np.sum(ivar * area * profile, axis=(-2, -1))
    M22 = np.sum(ivar * area**2, axis=(-2, -1))
    A1 = np.sum(ivar * data * profile, axis=(-2, -1))
    A2 = np.sum(ivar * data * area, axis=(-2, -1))
    # Solve the linear problem.
    det = M11 * M22 - M12**2
    M11 /= det
    M12 /= det
    M22 /= det
    f = M22 * A1 - M12 * A2
    b = M11 * A2 - M12 * A1
    # Calculate the covariance of (f, b).
    cov = np.stack(
        (np.stack((M22, -M12), axis=-1), np.stack((-M12, M11), axis=-1)), axis=-1
    )
    return f, b, cov, None


def shifted_profile(profile, dx, dy):
    """Shift a reference profile with translations along the x and y axis.

    Parameters
    ----------
    profile : array
        Array of shape (nf,ny,nx) with the spot profile(s) to use.
    dx : array
        Array of shape (nf)
        Position shift of the input profile to be applied in the x direction.
    dy : array
        Array of shape (nf)
        Position shift of the input profile to be applied in the y direction.

    Returns
    -------
    array
        Array (Shifted_profile) has shape (nf,ny,nx) and is the profile whose position have been translated by dx
        in the x direction and dy in the y direction.
    """
    nf = profile.shape[0]
    nx = profile.shape[1]
    ny = profile.shape[2]
    shifted_profile = np.zeros(profile.shape)
    x, y = np.meshgrid(
        np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny), indexing="ij"
    )
    for i in range(nf):
        interp = scipy.interpolate.RegularGridInterpolator(
            (np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny)),
            profile[i],
            bounds_error=False,
            fill_value=0,
        )
        shifted_profile[i] = interp((x - dx[i], y - dy[i]))
        # Add a normalisation step as the interpolation might not be flux preserving
        try:
            normalization = np.sum(shifted_profile[i])
            shifted_profile[i] = shifted_profile[i] / normalization
        except:
            pass
    return shifted_profile


def get_chi2(data, ivar, profile, flux, background, area=1):
    """Compute the chi2 residual given the data and the model parameters.

    Parameters
    ----------
    data : array
        Array of shape (ny,nx) with the data to fit.
    ivar : array
        Array of shape (ny,nx) with the corresponding ivars.
    profile : array
        Array of shape (ny,nx) with the spot profile to use.
    flux : scalar
        Value of the flux in the model fitting the data (profile*flux + background*area)
    background :
        Value of the background in the model fitting the data (profile*flux + background*area)
    area : scalar
        Area of each pixel used to predict its background level as b * area.


    Returns
    -------
    scalar
        Value of the chi2 residual given the data and the model.
    """
    return np.sum(ivar * (data - profile * flux - area * background) ** 2)


def fit_spots_flux_and_pos_fast(
    data, ivar, offset_spots, offset_dx, offset_dy, area=1, nfine=26, return_grids=False
):
    """Fit spot images to simultaneously determine the flux, background level, and centroid offsets.

    Similar to fit_spots_flux_and_pos() but runs much faster using precomputed offset spot images.
    The fit is performed in two stages: first the flux, background and resulting chi-square are
    calculated on a grid of centroid offsets. These calculations are fast since the flux and background
    can be calculated using linear algebra. The minimum chi-square over the offset grid then provides an
    initial fit over (flux, background, dx, dy) on a fixed coarse grid of (dx,dy).

    If nfine > 0, the second stage uses a 2D bicubic spline over the 5x5 subgrid of centroid offsets
    centered on the minimum chi-square from the first stage. This provides a refined estimate of (dx,dy)
    where the chi-square is minimized. The flux and background are then interpolated to this refined
    grid point on the same 5x5 subgrid. This second stage is not run if the initial minimum is too close
    to the edge of the offset grid, preventing the 5x5 subgrid from being defined.

    Parameters
    ----------
    data : array
        2D array of spot images with shape (nspots, ny, nx).
    ivar : array
        2D array of inverse variance of data with shape (nspots, ny, nx).
    offset_spots : array
        Array of offset spot images with shape (nspots, ngridy, ngridx, ny, nx).
    offset_dx : array
        Array of centroid offsets in units of pixels with shape (ngridx,).
        offset_spots[n,j,i] is the 2D spot image with shape (ny, nx) at centroid offset
        x = offset_dx[i], y = offset_dy[j].
    offset_dy : array
        Array of centroid offsets in units of pixels with shape (ngridy,).
        offset_spots[n,j,i] is the 2D spot image with shape (ny, nx) at centroid offset
        x = offset_dx[i], y = offset_dy[j].
    area : scalar or array
        Area of each pixel used to predict its background level as b * area.
    nfine : int
        Number of points to use along dx and dy in the refined 2D bicubic spline fit.
        Set to zero to prevent this second stage, resulting in faster but less
        accurate results.
    return_grids : bool
        If True, return the chi-square grid and the refined chi-square grid, which
        are useful for displaying diagnostic plots.

    Returns
    -------
    tuple
        Tuple of (fit_flux, fit_bg, fit_cov, fit_offsets) where fit_flux and fit_bg are the
        fitted flux and background for each spot with shapes (nspot,), fit_cov is the
        covariance matrix of (fit_flux, fit_bg) for each spot with shape (nspots, 2, 2),
        and fit_offsets is the fitted centroid offsets with shape (nspots, 2).
        If return_grids is True, the tuple also includes (chisq, finegrid, finedx, finedy)
        where chisq is the initial chi-square grid, finegrid is the refined chi-square grid,
        and finedx and finedy are the dx and dy offsets used in the refined grid.
    """
    # Tabulate chi-square over grid of centroid offsets (applied simultaneously to all spots).
    ndata = len(data)
    ngridx = len(offset_dx)
    ngridy = len(offset_dy)
    chisq = np.zeros((ngridy, ngridx))
    flux = np.zeros((ngridy, ngridx, ndata))
    bg = np.zeros((ngridy, ngridx, ndata))
    cov = np.zeros((ngridy, ngridx, ndata, 2, 2))
    for j in range(ngridy):
        for i in range(ngridx):
            flux[j, i], bg[j, i], cov[j, i], _ = fit_spots(
                data, ivar, offset_spots[:, j, i], area=area
            )
            model = (
                bg[j, i].reshape(-1, 1, 1)
                + flux[j, i].reshape(-1, 1, 1) * offset_spots[:, j, i]
            )
            chisq[j, i] = np.sum((data - model) ** 2 * ivar)
    # Find grid point with minimum chi-square.
    j, i = np.unravel_index(np.argmin(chisq), chisq.shape)
    fit_dx = offset_dx[i]
    fit_dy = offset_dy[j]
    fit_flux = flux[j, i]
    fit_bg = bg[j, i]
    fit_cov = cov[j, i]
    finegrid = None
    finedx = None
    finedy = None
    if nfine > 0:
        if j < 2 or i < 2 or j >= ngridy - 2 or i >= ngridx - 2:
            # The minimum is too close to the edge of the offset grid, so assume this is a bad fit
            # and instead center on the middle of the grid.
            i = ngridx // 2 + 1
            j = ngridy // 2 + 1
            fit_cov = cov[j, i]
        # Use a 2D bicubic spline of a 5x5 subgrid to refine the location of the minimum chisq.
        subgrid = chisq[j - 2 : j + 3, i - 2 : i + 3]
        subdx = offset_dx[i - 2 : i + 3]
        subdy = offset_dy[j - 2 : j + 3]
        finedx = np.linspace(subdx[0], subdx[-1], nfine)
        finedy = np.linspace(subdy[0], subdy[-1], nfine)
        pts = np.stack(np.meshgrid(finedy, finedx, indexing="ij"), axis=-1)
        spline = scipy.interpolate.RegularGridInterpolator(
            (subdy, subdx), subgrid, method="cubic"
        )
        finegrid = spline(pts)
        j2, i2 = np.unravel_index(np.argmin(finegrid), finegrid.shape)
        fit_dx = finedx[i2]
        fit_dy = finedy[j2]
        # Evalute the flux and background at the refined grid point.
        fit_flux = scipy.interpolate.RegularGridInterpolator(
            (subdy, subdx), flux[j - 2 : j + 3, i - 2 : i + 3], method="cubic"
        )((fit_dy, fit_dx))
        fit_bg = scipy.interpolate.RegularGridInterpolator(
            (subdy, subdx), bg[j - 2 : j + 3, i - 2 : i + 3], method="cubic"
        )((fit_dy, fit_dx))

    fit_offsets = np.stack((fit_dx, fit_dy), axis=-1)
    retval = (fit_flux, fit_bg, fit_cov, fit_offsets)
    if return_grids:
        retval += (chisq, finegrid, finedx, finedy)
    return retval


def fit_spots_flux_and_pos(data, ivar, profile, area=1):
    """Fit images of a spot to estimate the spot flux and background level as well as the position offset
    from the reference profile.

    Parameters
    ----------
    data : array
        Array of shape (nf,ny,nx) with the data to fit.
    ivar : array
        Array of shape (nf,ny,nx) with the corresponding ivars.
    profile : array
        Array of shape (nf,ny,nx) with the spot profile(s) to use.
    area : scalar or array
        Area of each pixel used to predict its background level as b * area.
        Either a scalar or an array of shape (nf,ny, nx).

    Returns
    -------
    tuple
        Tuple (f, b, cov, offsets) where f and b are arrays of shape (nf),
        cov has shape (nf,2,2) with elements [...,0,0] = var(f),
        [...,1,1] = var(b) and [...,0,1] = [...,1,0] = cov(f,b) and offsets has shape (nf,2) with elements
        [...,0] = position_offset(x direction) and [...,1] = position_offset(y direction)
    """
    npar = 4  # flux,bkg,dx,dy
    nf = profile.shape[
        0
    ]  # Number of sky monitoring fibers (10 for SkyCam0 and 7 for SkyCam1)
    nx = profile.shape[1]
    ny = profile.shape[2]

    flux = np.zeros(nf)
    bkg = np.zeros(nf)
    dx = np.zeros(nf)
    dy = np.zeros(nf)
    eps = 0.1

    # Set the matrices that will verify: M [delta_params] = A, and other variable used in the minimisation
    M = np.zeros((npar, npar))
    A = np.zeros(npar)
    der = np.zeros((npar, nf, nx, ny))
    cov = np.zeros((nf, 2, 2))
    prev_chi2 = 0  # Residual of the previous step (set to 0)
    spots_to_fit = list(range(nf))
    for step in range(10):
        der[0] = shifted_profile(profile, dx, dy)
        der[1] = np.ones((nf, nx, ny)) * area
        for i in spots_to_fit:
            if step == 0:
                Model = der[0][i] * flux[i] + area * bkg[i]
                for p in range(npar):
                    for q in range(p, npar):
                        M[q, p] = M[p, q] = np.sum(ivar[i] * der[p][i] * der[q][i])
                    A[p] = np.sum(ivar[i] * der[p][i] * (data[i] - Model))
                # Setting the diagonal to one everywhere it is null (ie in the dx, dy related blocks)
                M[2:, 2:] = np.diag(np.ones(2))

            else:
                der[2][i] = flux[i] * (
                    (shifted_profile(profile, dx + eps, dy)[i] - der[0][i]) / eps
                )
                der[3][i] = flux[i] * (
                    (shifted_profile(profile, dx, dy + eps)[i] - der[0][i]) / eps
                )
                Model = der[0][i] * flux[i] + area * bkg[i]
                for p in range(npar):
                    for q in range(p, npar):
                        M[q, p] = M[p, q] = np.sum(ivar[i] * der[p][i] * der[q][i])
                    A[p] = np.sum(ivar[i] * der[p][i] * (data[i] - Model))

            # Solve the matrix system
            if np.linalg.det(M) != 0:
                M_inv = np.linalg.inv(M)
                Sol = np.dot(M_inv, A)
                # Search for a missed minimum of the chi2 function
                alpha_opt = 1
                new_profile = shifted_profile(profile, dx + Sol[2], dy + Sol[3])[i]
                new_chi2 = get_chi2(
                    data[i],
                    ivar[i],
                    new_profile,
                    flux[i] + Sol[0],
                    bkg[i] + Sol[1],
                    area,
                )
                try:
                    for alpha in np.linspace(
                        0, 1, int(10 * max(abs(Sol[2]), abs(Sol[3])))
                    ):
                        if alpha != 0:
                            new_profile = shifted_profile(
                                profile, dx + alpha * Sol[2], dy + alpha * Sol[3]
                            )[i]
                            chi2 = get_chi2(
                                data[i],
                                ivar[i],
                                new_profile,
                                flux[i] + alpha * Sol[0],
                                bkg[i] + alpha * Sol[1],
                                area,
                            )
                            if chi2 < new_chi2:
                                alpha_opt = alpha
                                new_chi2 = chi2
                except:
                    pass
                # Add to each parameter their "optimal" increment
                flux[i] += alpha_opt * Sol[0]
                bkg[i] += alpha_opt * Sol[1]
                dx[i] += alpha_opt * Sol[2]
                dy[i] += alpha_opt * Sol[3]
                # Break i-th spot loop if its minimisation is satisfying or gives unrealistic dx and/or dy
                if (
                    (abs(prev_chi2 - new_chi2) < 0.1)
                    or (abs(dx[i]) > 6)
                    or (abs(dy[i]) > 6)
                ):
                    spots_to_fit.remove(i)
                # Store the residual at this step
                prev_chi2 = new_chi2
                # Calculate the covariance of (f, b).
                cov[i] = M_inv[:2, :2]

    offsets = np.stack((dx, dy), axis=-1)
    return flux, bkg, cov, offsets


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


def detect_sources(
    snr,
    minsnr=4,
    minsize=8,
    maxsize=32,
    minsep=0,
    min_snr_ratio=0.1,
    maxsrc=20,
    measure=None,
):
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
        raise ValueError("Expected minsize <= maxsize.")
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
        snr,
        labeled,
        labels,
        out_dtype=float,
        default=-1,
        func=lambda X: np.sqrt(np.sum(X**2)),
    )
    maxsnrtot = None
    # Rank sources by snrtot.
    ranks = np.argsort(snrtot)[::-1]
    # Build the final list of detected sources.
    sources = []
    snrsq = snr**2
    minsepsq = minsep**2
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
    xy = (
        np.arange(size * oversampling) - 0.5 * (size * oversampling - 1)
    ) / oversampling
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
        raise ValueError("Data must be 2 dimensional.")
    ny, nx = data.shape
    if not allow_trim and ((nx % downsampling) or (ny % downsampling)):
        raise ValueError(
            "Data shape {0} does not evenly divide downsampling={1} and allow_trim is False.".format(
                (ny, nx), downsampling
            )
        )
    ny //= downsampling
    nx //= downsampling
    shape = (ny, nx, downsampling, downsampling)
    strides = (
        downsampling * data.strides[0],
        downsampling * data.strides[1],
    ) + data.strides
    blocks = np.lib.stride_tricks.as_strided(
        data[: downsampling * ny, : downsampling * nx], shape=shape, strides=strides
    )
    return summary(blocks, axis=(2, 3))


def downsample_weighted(D, W, downsampling=4, allow_trim=True):
    """Downsample 2D data D with weights W."""
    if D.shape != W.shape:
        raise ValueError("Arrays D, W must have the same shape.")
    if D.ndim != 2:
        raise ValueError("Arrays D, W must be 2D.")
    if np.any(W < 0):
        raise ValueError("Array W contains negative values.")
    WD = downsample(
        D * W, downsampling=downsampling, summary=np.sum, allow_trim=allow_trim
    )
    W = downsample(W, downsampling=downsampling, summary=np.sum, allow_trim=allow_trim)
    D = np.divide(WD, W, out=np.zeros_like(WD), where=W > 0)
    return D, W


def smooth(D, W, smoothing):
    """Apply a weighted Gaussian smoothing."""
    WD = scipy.ndimage.gaussian_filter(W * D, smoothing)
    W = scipy.ndimage.gaussian_filter(W, smoothing)
    D = np.divide(WD, W, out=np.zeros_like(D), where=W > 0)
    return D, W


def get_stacked(stamps, smoothing=1, maxdither=1, maxdist=3, min_stack=2):
    """Calculate a stack of detected sources ignoring outliers."""
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
                D1, W1, D2, W2, maxdither=maxdither, smoothing=smoothing
            )
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
        inset_j = slice(maxdither + dy, ny - maxdither + dy), slice(
            maxdither + dx, nx - maxdither + dx
        )
        Dj, Wj = f * D[inset_j], W[inset_j] / f**2
        DWstack += Dj * Wj
        Wstack += Wj
    Dstack = np.divide(DWstack, Wstack, out=np.zeros_like(DWstack), where=Wstack > 0)
    return normalize_stamp(Dstack, Wstack)


def get_stamp_distance(
    D1, W1, D2, W2, maxdither=3, smoothing=1, fscale=np.linspace(0.85, 1.15, 11)
):
    """Calculate the minimum chisq distance between two stamps allowing for some dither."""
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
            D2inset = D2[
                maxdither + dy : ny - maxdither + dy,
                maxdither + dx : nx - maxdither + dx,
            ]
            W2inset = W2[
                maxdither + dy : ny - maxdither + dy,
                maxdither + dx : nx - maxdither + dx,
            ]
            # Calculate the chi-square distance between the inset stamps with scale factors of
            # 1/fvec and fvec applied to (D1,W1) and (D2,W2) respectively.
            num = np.sqrt(W1inset * W2inset) * (D1inset / fvec - D2inset * fvec)
            denom = np.sqrt(W1inset * fvec**2 + W2inset * fvec**-2)
            # Could also use where=(num > 0) here.
            pull[iy, ix] = np.divide(
                num, denom, out=np.zeros_like(num), where=denom > 0
            )
    # Find the dither with the smallest chisq.
    chisq = np.sum(pull**2, axis=(3, 4))
    iy, ix, iscale = np.unravel_index(
        np.argmin(chisq.reshape(-1)), (ndither, ndither, nscale)
    )
    assert chisq.min() == chisq[iy, ix, iscale]
    # Return the smallest distance, the corresponding dither and scale, and the best pull image.
    return (
        chisq[iy, ix, iscale] / D1inset.size,
        np.array((dxy[iy], dxy[ix]), int),
        fscale[iscale],
        pull[iy, ix, iscale].copy(),
    )


def normalize_stamp(D, W, smoothing=2.5):
    """Normalize a stamp to its weighted mean value.
    Should generally subtract a background estimate first.
    """
    smoothed, _ = smooth(D, W, smoothing)
    norm = smoothed.sum()
    if norm != 0:
        return D / np.abs(norm), W * norm**2


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
    theta = 2 * np.pi * k / phi**2
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
    xy = (
        np.arange(size * oversampling) - 0.5 * (size * oversampling - 1)
    ) / oversampling
    z = profile(xy - dx, (xy - dy).reshape(-1, 1))
    T = downsample(z, oversampling, np.mean)
    if normalized:
        T /= T.sum()
    return T


def preprocess(D, W, nsig_lo=10, nsig_hi=30, vmin=None, vmax=None):
    """Preprocess weighted 2D array data for display."""
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

    def __init__(
        self,
        stamp_size,
        fiber_diam_um=107,
        pixel_size_um=15,
        plate_scales=(70.0, 76.0),
        max_offset_pix=3.5,
        noffset=15,
        nangbins=20,
    ):
        self.stamp_size = stamp_size
        self.pixel_size_um = pixel_size_um
        self.plate_scales = plate_scales
        # Tabulate fiber templates for each (x,y) offset in the x >= 0 and y >= 0 quadrant.
        self.offset_template = np.empty(
            (noffset, noffset, stamp_size, stamp_size), np.float32
        )
        max_rsq = (0.5 * fiber_diam_um / pixel_size_um) ** 2
        profile = lambda x, y: 1.0 * (x**2 + y**2 < max_rsq)
        delta = np.linspace(0, max_offset_pix, noffset)
        for iy in range(noffset):
            for ix in range(noffset):
                self.offset_template[iy, ix] = make_template(
                    stamp_size, profile, dx=delta[ix], dy=delta[iy], normalized=False
                )
        self.xyoffset = np.linspace(-max_offset_pix, +max_offset_pix, 2 * noffset - 1)
        dxy = np.arange(self.stamp_size) - 0.5 * (self.stamp_size - 1)
        self.xgrid, self.ygrid = np.meshgrid(dxy, dxy, sparse=False)
        rmax = dxy[-1] * self.pixel_size_um / max(self.plate_scales)
        self.angbins = np.linspace(0.0, rmax, nangbins + 1)
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
                    fiberfrac[origin - iy, origin + ix] = (
                        np.sum(P * T[reverse, :]) / Psum
                    )
                if ix > 0:
                    # Calculate in the x < 0 and y >= 0 quadrant.
                    fiberfrac[origin + iy, origin - ix] = (
                        np.sum(P * T[:, reverse]) / Psum
                    )
                if iy > 0 and ix > 0:
                    # Calculate in the x < 0 and y < 0 quadrant.
                    fiberfrac[origin - iy, origin - ix] = (
                        np.sum(P * T[reverse, reverse]) / Psum
                    )
        # Locate the best centered offset.
        iy, ix = np.unravel_index(np.argmax(fiberfrac), fiberfrac.shape)
        xc = self.xyoffset[ix]
        yc = self.xyoffset[iy]
        # Calculate the radius of each pixel in arcsecs relative to this center.
        radius = np.hypot(
            (self.xgrid - xc) * self.pixel_size_um / self.plate_scales[0],
            (self.ygrid - yc) * self.pixel_size_um / self.plate_scales[1],
        ).reshape(-1)
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

    def __init__(self, stamp_size, inset, method="fiber"):
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
        if method == "fiber":
            profile = lambda x, y: 1.0 * (x**2 + y**2 < rfibersq)
        elif method == "donut":
            xymax = 0.5 * (stamp_size - 2 * inset)

            def profile(x, y):
                rsq = x**2 + y**2
                return (
                    rsq
                    * np.exp(-0.5 * rsq / (3 * rfibersq))
                    * (np.maximum(np.abs(x), np.abs(y)) < xymax)
                )

        else:
            raise ValueError('Unsupported method "{0}".'.format(method))
        # Build the profiles.
        for iy in range(noffset):
            for ix in range(noffset):
                self.template[iy, ix] = make_template(
                    stamp_size,
                    profile,
                    dx=self.dxy[ix],
                    dy=self.dxy[iy],
                    normalized=True,
                )

    def center(self, D, W):
        assert D.shape == (self.stamp_size, self.stamp_size) and W.shape == D.shape
        S = slice(self.inset, self.inset + self.inset_size)
        # Calculate the weighted mean template flux for each offset.
        WDsum = np.sum(W * D * self.template, axis=(2, 3))
        Wsum = np.sum(W * self.template, axis=(2, 3))
        meanflux = np.divide(
            WDsum, Wsum, out=np.zeros(self.template.shape[:2]), where=Wsum > 0
        )
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

    def __init__(self, maxlen, default_value, padding=300, recent=300, aux_dtype=[]):
        self.oldest = None
        self.len = 0
        self.full = False
        self.last = None
        dtype = [
            ("mjd1", np.float64),
            ("mjd2", np.float64),
            ("value", np.float32),
            ("error", np.float32),
        ] + aux_dtype
        self._entries = np.zeros(shape=maxlen, dtype=dtype)
        self.names = self._entries.dtype.names
        self.default_value = default_value
        self.padding = padding / self.SECS_PER_DAY
        self.recent = recent / self.SECS_PER_DAY

    @property
    def entries(self):
        return self._entries[: self.len]

    def __str__(self):
        return f"{self.__class__.__name__}(len={self.len}, full={self.full}, oldest={self.oldest})"

    def add(self, mjd1, mjd2, value, error, aux_data=()):
        """Add a single measurement.

        We make no assumption that measurements are non-overlapping or added in time order.
        """
        assert mjd1 < mjd2 and error > 0
        is_oldest = (self.oldest is None) or (mjd1 < self._entries[self.oldest]["mjd1"])
        entry = (mjd1, mjd2, value, error) + aux_data
        if self.full:
            assert self.oldest is not None
            if is_oldest:
                # Ignore this since it is older than all existing entries.
                return
            self.last = self.oldest
            self._entries[self.last] = entry
            # Update the index of the oldest entry, which might be us.
            self.oldest = np.argmin(self.entries["mjd1"])
        else:
            self.last = self.len
            if is_oldest:
                # This is now the oldest entry.
                self.oldest = self.last
            self.len += 1
            self.full = self.len == self._entries.size
            self._entries[self.last] = entry

    def set_last(self, **kwargs):
        """Set values of the most recently added measurement."""
        if self.last is not None:
            for name, value in kwargs.items():
                if name in self.names:
                    self._entries[self.last][name] = value

    def inside(self, mjd1, mjd2):
        """Return a mask for entries whose intervals overlap [mjd1, mjd2].

        Use mjd2=None to select all entries after mjd1.
        """
        mask = self.entries["mjd2"] > mjd1
        if mjd2 is not None:
            mask &= self.entries["mjd1"] < mjd2
        return mask

    def average(self, mjd, interval_secs, min_values, field="value"):
        """Return the average of values recorded up to inteval_secs before mjd,
        or None if less than min_values have been recorded.
        """
        sel = self.inside(mjd - interval_secs / self.SECS_PER_DAY, mjd)
        nsel = np.count_nonzero(sel)
        return np.mean(self.entries[sel][field]) if nsel >= min_values else None

    def sample_grid(self, mjd_grid, field="value"):
        """Sample measurements on a the specified MJD grid.

        Use measurements that lie outside the grid up to self.padding seconds.
        Return default_value when no measurements are available.
        Use constant extrapolation of the first/last measurement if necessary.
        """
        mjd1, mjd2 = mjd_grid[0], mjd_grid[-1]
        # Select measurements that span the padded input grid.
        sel = self.inside(mjd1 - self.padding, mjd2 + self.padding)
        if not np.any(sel):
            return np.full_like(mjd_grid, self.default_value)
        mjd_sel = 0.5 * (self.entries[sel]["mjd1"] + self.entries[sel]["mjd2"])
        dmjd_sel = self.entries[sel]["mjd2"] - self.entries[sel]["mjd1"]
        value_sel = self.entries[sel][field]
        # Values might not be recorded in time order so fix that now.
        iorder = np.argsort(mjd_sel)
        mjd_sel = mjd_sel[iorder]
        dmjd_sel = dmjd_sel[iorder]
        value_sel = value_sel[iorder]
        # The measurements are integrals over each exposure with some deadtime between them.
        # Correct for this deadtime by calculating a piece-wise linear approximation to
        # the instantaneous value that matches the measured integrals.
        try:
            value_sel_corrected = pwlinear_solve(
                mjd_sel, dmjd_sel, value_sel * dmjd_sel
            )
        except Exception as e:
            print(f"pwlinear_solve failed: {e}")
            value_sel_corrected = value_sel
        # Use linear interpolation with constant extrapolation beyond the endpoints.
        return np.interp(mjd_grid, mjd_sel, value_sel_corrected)

    def trend(self, mjd):
        """Return the linear trend in values over (mjd - recent, mjd).
        For now, this returns a weighted average with zero slope.
        """
        sel = self.inside(mjd - self.recent, mjd)
        if not np.any(sel):
            return self.default_value, 0
        wgt = self.entries[sel]["error"] ** -0.5
        val = self.entries[sel]["value"]
        return np.sum(wgt * val) / np.sum(wgt), 0

    def forecast_grid(self, mjd_grid):
        """Forecast our trend on the specified MJD grid."""
        mjd1, mjd2 = mjd_grid[0], mjd_grid[-1]
        # Calculate the trend at mjd1.
        offset, slope = self.trend(mjd1)
        # Evaluate the linear trend on our grid.
        return offset + slope * (mjd_grid - mjd1)

    def save(self, mjd1, mjd2=None):
        """Return a json suitable serialization of our entries spanning (mjd1, mjd2).

        Use mjd2=None to use all entries after mjd1.
        If this buffer has auxiliary data, that will saved also.
        Note that we return numpy data types, which are not JSON serializable by
        default, so this assumes that the caller uses :class:`NumpyEncoder` or
        something equivalent.
        """
        sel = self.inside(mjd1, mjd2)
        E = self.entries[sel]
        if len(E) == 0:
            return {}
        # Sort based on mjd1.
        isort = np.argsort(E["mjd1"])
        E = E[isort]
        # Convert to a dictionary of fields, excluding mjd1,2.
        D = {name: E[name] for name in E.dtype.fields if name not in ("mjd1", "mjd2")}
        # Lookup the earliest MJD.
        mjd0 = E["mjd1"][0]
        D["mjd0"] = mjd0
        # Replace mjd1,2 with offsets dt1,2 from mjd0 in seconds.
        # Use float32 so that JSON output will be rounded.
        D["dt1"] = np.float32((E["mjd1"] - mjd0) * self.SECS_PER_DAY)
        D["dt2"] = np.float32((E["mjd2"] - mjd0) * self.SECS_PER_DAY)
        return D


def mjd_to_date(mjd, utc_offset):
    """Convert an MJD value to a datetime using the specified UTC offset in hours.

    Use utc_offset of -7 for local time at Kitt Peak.
    Use :func:`date_to_mjd` to invert this calculation.
    """
    return datetime.datetime(2019, 1, 1) + datetime.timedelta(
        days=mjd - 58484.0, hours=utc_offset
    )


def date_to_mjd(date, utc_offset):
    """Convert a datetime using the specified UTC offset in hours to an MJD value.

    Use utc_offset of -7 for local time at Kitt Peak.
    Use :func:`mjd_to_date` to invert this calculation.
    """
    delta = date - datetime.datetime(2019, 1, 1) - datetime.timedelta(hours=utc_offset)
    return 58484 + delta.days + (delta.seconds + 1e-6 * delta.microseconds) / 86400


def mjd_to_night(mjd):
    """Convert MJD to NIGHT for KPNO in the format YYYYMMDD.

    Uses the convention that the night rollover occurs at local (UTC-7) noon.
    """
    date = mjd_to_date(mjd, utc_offset=-7)
    if date.hour < 12:
        date -= datetime.timedelta(days=1)
    return int(date.strftime("%Y%m%d"))


def night_to_midnight(night, utc_offset):
    """Convert YYYYMMDD into a datetime representing midnight. Use utc_offset=0 for a
    result with midnight.hour==0 or utc_offset=-7 for the KPNO local time.
    """
    night = str(night)
    if len(night) != 8:
        raise ValueError("night_to_midnight: expected an integer of the form YYYYMMDD.")
    year, month, day = int(night[0:4]), int(night[4:6]), int(night[6:8])
    return datetime.datetime(year, month, day, 12) + datetime.timedelta(
        hours=12 + utc_offset
    )


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder to use with numpy data with rounding of float32 values."""

    FLOAT32_DECIMALS = 6

    def default(self, obj):
        if isinstance(obj, np.float32):
            # Convert to 64-bit float before rounding.
            return float(np.round(np.float64(obj), self.FLOAT32_DECIMALS))
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            if obj.dtype.fields is not None:
                # convert a recarray to a dictionary.
                new_obj = {}
                for name, (dtype, size) in obj.dtype.fields.items():
                    if dtype.base == np.float32:
                        new_obj[name] = np.round(obj[name], self.FLOAT32_DECIMALS)
                    else:
                        new_obj[name] = obj[name]
                return new_obj
            else:
                if obj.dtype == np.float32:
                    # tolist converts to 64-bit native float so apply rounding first.
                    obj = np.round(obj.astype(np.float64), self.FLOAT32_DECIMALS)
                return obj.tolist()
        else:
            return super().default(obj)


def is_datetime(time, oldest=datetime.datetime(2019, 1, 1)):
    """Test for a valid datetime after oldest."""
    try:
        delta = (time - oldest).days
        return delta > 0
    except Exception as e:
        return False


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
    cameras = ("GUIDE0", "GUIDE2", "GUIDE3", "GUIDE5", "GUIDE7", "GUIDE8")
    # Read the json file of guider outputs.
    jsonpath = path / "centroids-{0}.json".format(expid)
    if not jsonpath.exists():
        raise ValueError("Non-existent path: {0}.".format(jsonpath))
    with open(jsonpath) as f:
        D = json.load(f)
        assert D["expid"] == int(expid)
        nframes = D["summary"]["frames"]
    # Use the first frame to lookup the guide stars for each camera.
    frame0 = D["frames"]["1"]
    stars = {G: len([K for K in frame0.keys() if K.startswith(G)]) for G in cameras}
    expected = {G: np.zeros((stars[G], 2)) for G in cameras}
    combined = {G: np.zeros((2, nframes)) for G in cameras}
    centroid = {G: np.zeros((stars[G], 2, nframes)) for G in cameras}
    for camera in cameras:
        # Get the expected position for each guide star.
        for istar in range(stars[camera]):
            S = frame0.get(camera + f"_{istar}")
            expected[camera][istar, 0] = S["y_expected"]
            expected[camera][istar, 1] = S["x_expected"]
        # Get the combined centroid sent to the telescope for each frame.
        for iframe in range(nframes):
            F = D["frames"].get(str(iframe + 1))
            if F is None:
                logging.warning(
                    "Missing frame {0}/{1} in {2}".format(iframe + 1, nframes, jsonpath)
                )
                continue
            combined[camera][0, iframe] = F["combined_y"]
            combined[camera][1, iframe] = F["combined_x"]
            # Get the measured centroids for each guide star in this frame.
            for istar in range(stars[camera]):
                S = F.get(camera + "_{0}".format(istar))
                centroid[camera][istar, 0, iframe] = S.get("y_centroid", np.nan)
                centroid[camera][istar, 1, iframe] = S.get("x_centroid", np.nan)
    return expected, combined, centroid


def git_describe():
    """Return a string describing the git origin of the package where this function is defined.

    The result is usually <tag>-<n>-g<hash> where <tag> is the last tag, <n> is the number of
    subsequent commits, and <hash> is the current git hash.  When n=0, only <hash> is returned.
    Details on git-describe are at https://git-scm.com/docs/git-describe
    """
    try:
        path = pathlib.Path(__file__).parent
        process = subprocess.Popen(
            ["git", "describe", "--tags", "--always"],
            cwd=path,
            shell=False,
            stdout=subprocess.PIPE,
        )
        return process.communicate()[0].strip().decode()
    except Exception as e:
        return None


def cos_zenith_to_airmass(cosZ):
    """Convert a zenith angle to an airmass.
    Uses the Rozenberg 1966 interpolation formula, which gives reasonable
    results for high zenith angles, with a horizon air mass of 40.
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Interpolative_formulas
    Rozenberg, G. V. 1966. "Twilight: A Study in Atmospheric Optics."
    New York: Plenum Press, 160.
    The value of cosZ is clipped to [0,1], so observations below the horizon
    return the horizon value (~40).
    Parameters
    ----------
    cosZ : float or array
        Cosine of angle(s) to convert.
    Returns
    -------
    float or array
        Airmass value(s) >= 1.
    """
    cosZ = np.clip(np.asarray(cosZ), 0.0, 1.0)
    return np.clip(1.0 / (cosZ + 0.025 * np.exp(-11 * cosZ)), 1.0, None)


# Arrays of (r,scale_r,scale_az) in units of (mm, um/arcsec, um/arcsec)
# downsampled from $DESIMODEL/data/focalplane/platescale.txt
platescale_data = np.array(
    [
        [
            0.0,
            9.55338614,
            19.10823877,
            28.66602485,
            38.22821229,
            47.79627044,
            57.37167061,
            66.95583925,
            76.5503467,
            86.15653808,
            95.77607221,
            105.40998724,
            115.06050339,
            124.72876316,
            134.41629762,
            144.12445419,
            153.85507949,
            163.60955313,
            173.38942287,
            183.19603606,
            193.03141186,
            202.89690121,
            212.79409317,
            222.72462014,
            232.69009941,
            242.69217521,
            252.73250831,
            262.81277626,
            272.93467342,
            283.09991068,
            293.31021495,
            303.56735178,
            313.87303193,
            324.22913743,
            334.637255,
            345.09926806,
            355.61697659,
            366.19242695,
            376.8270098,
            387.52276409,
            398.28177809,
            409.10421235,
            419.98856998,
        ],
        [
            67.48456941,
            67.4897893,
            67.50549147,
            67.53167143,
            67.56832223,
            67.61543507,
            67.67300035,
            67.74100878,
            67.81945285,
            67.9083283,
            68.00763583,
            68.11738281,
            68.23758501,
            68.36826828,
            68.50947014,
            68.66124117,
            68.8236462,
            68.99676513,
            69.18069348,
            69.3755424,
            69.58143832,
            69.7985219,
            70.0269465,
            70.26687594,
            70.51848152,
            70.78193842,
            71.05742124,
            71.3450988,
            71.64512835,
            71.95764893,
            72.2827744,
            72.62058611,
            72.97112576,
            73.33438887,
            73.71031989,
            74.09881018,
            74.49970068,
            74.91279208,
            75.33786611,
            75.77472346,
            76.22324542,
            76.6834893,
            77.15583098,
        ],
        [
            67.48456948,
            67.4863062,
            67.49151606,
            67.50019814,
            67.51235097,
            67.52797265,
            67.54706099,
            67.56961364,
            67.59562835,
            67.62510318,
            67.65803674,
            67.69442847,
            67.73427893,
            67.77759008,
            67.82436552,
            67.87461092,
            67.92833383,
            67.98554449,
            68.04625563,
            68.11048271,
            68.17824403,
            68.24956072,
            68.32445674,
            68.40295874,
            68.48509592,
            68.57089967,
            68.66040331,
            68.75364155,
            68.85064994,
            68.95146421,
            69.05611955,
            69.16464973,
            69.27708619,
            69.39345711,
            69.51378646,
            69.63809304,
            69.76638985,
            69.89868353,
            70.03497444,
            70.17525741,
            70.31952364,
            70.46776416,
            70.61997566,
        ],
    ]
)

"""Return the radial and azimuthal platescales in um/arcsec given the
CS5 focal-plane radius in mm, using linear interpolation.
"""
get_platescales = scipy.interpolate.interp1d(
    platescale_data[0], platescale_data[1:], copy=False, assume_sorted=True
)


def robust_median(X, axis=0, zcut=3.5):
    """Calculate a robust median of X along the specified axis."""
    X = np.asarray(X, np.float64)
    X0 = np.nanmedian(X, axis=axis, keepdims=True)
    MAD = np.nanmedian(np.abs(X - X0), axis=axis, keepdims=True)
    Z = 0.6745 * (X - X0)
    good = np.abs(Z) <= zcut * MAD
    X[~good] = np.nan
    return np.nanmedian(X, axis=axis)


def pwlinear_solve(t, dt, yint):
    """Calculate the piecewise linear function y(t) passing through points (t[i], y[i])
    with integrals yint[i] = Integrate[y[t], {t,t[i]-dt[i]/2,t[i]+dt[i]/2}].

    Parameters
    ----------
    t : array
        1D array of points at the center of each integral.
    dt : array
        1D array of widths for each integral.
    yint : array
        1D array of integral values.

    Returns
    -------
    array
        1D array y[i] such that yint[i] = Integrate[y[t], {t,t[i]-dt[i]/2,t[i]+dt[i]/2}]
        when y(t) is piecewise linear through (t[i],y[i]) and constant below and above
        the endpoints.
    """
    dt = np.asarray(dt)
    if len(dt) == 1:
        return yint / dt
    assert len(dt) == len(t)
    assert len(dt) == len(yint)
    Dt = np.diff(t)
    if np.any(Dt <= 0):
        return yint / dt
    dt2 = dt**2 / 8
    hi = dt2[1:] / Dt
    lo = dt2[:-1] / Dt
    banded = np.zeros((3, len(dt)))
    banded[1] = dt
    banded[1, :-1] -= hi
    banded[1, 1:] -= lo
    banded[0, 1:] = hi
    banded[2, :-1] = lo
    return scipy.linalg.solve_banded((1, 1), banded, yint)


_blur_kernel = np.array(
    [
        [1.8584491e-07, 4.3132287e-04, 1.8588798e-07],
        [4.3132226e-04, 9.9827397e-01, 4.3132226e-04],
        [1.8588798e-07, 4.3132287e-04, 1.8584491e-07],
    ]
)


def blur(D, W):
    """Apply a weighted 0.15-pixel Gaussian blur to reduce the impact of any
    isolated pixels with large ivar"""
    DWconv = scipy.signal.convolve(D * W, _blur_kernel, mode="same")
    Wconv = scipy.signal.convolve(W, _blur_kernel, mode="same")
    return np.divide(DWconv, Wconv, out=np.zeros_like(D), where=Wconv > 0), Wconv


# 11x11 convolution kernel for the nominal ELG profile which is round Sersic n=1 with 0.45" half-light radius.
# This kernel is stretched by 1.084 along y and squeezed by 0.922 along x to account for the different plate scales.
# To apply this profile, use scipy.signal.convolve(GFA_data, _ELG_kernel, mode='same')
_ELG_kernel = np.array(
    [
        [
            0.0004233430081512779,
            0.0007364505436271429,
            0.001188363297842443,
            0.0017316826852038503,
            0.0022099444177001715,
            0.002399129094555974,
            0.0022099444177001715,
            0.0017316826852038503,
            0.001188363297842443,
            0.0007364505436271429,
            0.0004233430081512779,
        ],
        [
            0.0006760309333913028,
            0.0012480159057304263,
            0.0021513437386602163,
            0.003348887199535966,
            0.0045022256672382355,
            0.0049937451258301735,
            0.0045022256672382355,
            0.003348887199535966,
            0.0021513437386602163,
            0.0012480159057304263,
            0.0006760309333913028,
        ],
        [
            0.0010057131294161081,
            0.001976283034309745,
            0.0036765243858098984,
            0.006234399974346161,
            0.00904802419245243,
            0.010399244725704193,
            0.00904802419245243,
            0.006234399974346161,
            0.0036765243858098984,
            0.001976282801479101,
            0.0010057131294161081,
        ],
        [
            0.0013665164588019252,
            0.0028473353013396263,
            0.005736198276281357,
            0.010810506530106068,
            0.017641831189393997,
            0.021501323208212852,
            0.017641831189393997,
            0.010810506530106068,
            0.005736198276281357,
            0.0028473353013396263,
            0.0013665164588019252,
        ],
        [
            0.001660904148593545,
            0.0036149118095636368,
            0.0077792382799088955,
            0.016309885308146477,
            0.03155653551220894,
            0.04382346570491791,
            0.03155653551220894,
            0.016309885308146477,
            0.0077792382799088955,
            0.0036149118095636368,
            0.001660904148593545,
        ],
        [
            0.0017684746999293566,
            0.003920542076230049,
            0.00870299432426691,
            0.019154787063598633,
            0.04165779799222946,
            0.06993736326694489,
            0.04165779799222946,
            0.019154787063598633,
            0.00870299432426691,
            0.003920542076230049,
            0.0017684746999293566,
        ],
        [
            0.001660904148593545,
            0.0036149118095636368,
            0.0077792382799088955,
            0.016309885308146477,
            0.03155653551220894,
            0.04382346570491791,
            0.03155653551220894,
            0.016309885308146477,
            0.0077792382799088955,
            0.0036149118095636368,
            0.001660904148593545,
        ],
        [
            0.0013665164588019252,
            0.0028473353013396263,
            0.005736198276281357,
            0.010810506530106068,
            0.017641831189393997,
            0.021501323208212852,
            0.017641831189393997,
            0.010810506530106068,
            0.005736198276281357,
            0.0028473353013396263,
            0.0013665164588019252,
        ],
        [
            0.0010057131294161081,
            0.001976282801479101,
            0.0036765243858098984,
            0.006234399974346161,
            0.00904802419245243,
            0.010399244725704193,
            0.00904802419245243,
            0.006234399974346161,
            0.0036765243858098984,
            0.001976283034309745,
            0.0010057131294161081,
        ],
        [
            0.0006760309333913028,
            0.0012480159057304263,
            0.0021513437386602163,
            0.003348887199535966,
            0.0045022256672382355,
            0.0049937451258301735,
            0.0045022256672382355,
            0.003348887199535966,
            0.0021513437386602163,
            0.0012480159057304263,
            0.0006760309333913028,
        ],
        [
            0.0004233430081512779,
            0.0007364505436271429,
            0.001188363297842443,
            0.0017316826852038503,
            0.0022099444177001715,
            0.002399129094555974,
            0.0022099444177001715,
            0.0017316826852038503,
            0.001188363297842443,
            0.0007364505436271429,
            0.0004233430081512779,
        ],
    ]
)

# 11x11 convolution kernel for the nominal BGS profile which is round Sersic n=4 with 1.5" half-light radius.
# This kernel is stretched by 1.084 along y and squeezed by 0.922 along x to account for the different plate scales.
# To apply this profile, use scipy.signal.convolve(GFA_data, _ELG_kernel, mode='same')
_BGS_kernel = np.array(
    [
        [
            0.0008819324430078268,
            0.0010629459284245968,
            0.0012624080991372466,
            0.0014565379824489355,
            0.00160425144713372,
            0.0016609467566013336,
            0.00160425144713372,
            0.0014565379824489355,
            0.0012624080991372466,
            0.0010629459284245968,
            0.0008819324430078268,
        ],
        [
            0.0010629459284245968,
            0.001333612366579473,
            0.0016604403499513865,
            0.0020129659678786993,
            0.002308456925675273,
            0.002431850880384445,
            0.002308456925675273,
            0.0020129659678786993,
            0.0016604403499513865,
            0.0013336124829947948,
            0.0010629459284245968,
        ],
        [
            0.0012624080991372466,
            0.0016604403499513865,
            0.002200713846832514,
            0.002877740887925029,
            0.003545331070199609,
            0.0038440146017819643,
            0.003545331070199609,
            0.002877740887925029,
            0.002200713846832514,
            0.0016604403499513865,
            0.0012624080991372466,
        ],
        [
            0.0014565379824489355,
            0.0020129659678786993,
            0.002877740887925029,
            0.004214275162667036,
            0.0059745050966739655,
            0.006993815768510103,
            0.0059745050966739655,
            0.004214275162667036,
            0.002877740887925029,
            0.0020129659678786993,
            0.0014565379824489355,
        ],
        [
            0.00160425144713372,
            0.002308456925675273,
            0.003545331070199609,
            0.0059745050966739655,
            0.011260008439421654,
            0.017317788675427437,
            0.011260008439421654,
            0.0059745050966739655,
            0.003545331070199609,
            0.002308456925675273,
            0.00160425144713372,
        ],
        [
            0.0016609467566013336,
            0.002431850880384445,
            0.0038440146017819643,
            0.006993815768510103,
            0.017317788675427437,
            0.053177446126937866,
            0.017317788675427437,
            0.006993815768510103,
            0.0038440146017819643,
            0.002431850880384445,
            0.0016609467566013336,
        ],
        [
            0.00160425144713372,
            0.002308456925675273,
            0.003545331070199609,
            0.0059745050966739655,
            0.011260008439421654,
            0.017317788675427437,
            0.011260008439421654,
            0.0059745050966739655,
            0.003545331070199609,
            0.002308456925675273,
            0.00160425144713372,
        ],
        [
            0.0014565379824489355,
            0.0020129659678786993,
            0.002877740887925029,
            0.004214275162667036,
            0.0059745050966739655,
            0.006993815768510103,
            0.0059745050966739655,
            0.004214275162667036,
            0.002877740887925029,
            0.0020129659678786993,
            0.0014565379824489355,
        ],
        [
            0.0012624080991372466,
            0.0016604403499513865,
            0.002200713846832514,
            0.002877740887925029,
            0.003545331070199609,
            0.0038440146017819643,
            0.003545331070199609,
            0.002877740887925029,
            0.002200713846832514,
            0.0016604403499513865,
            0.0012624080991372466,
        ],
        [
            0.0010629459284245968,
            0.0013336124829947948,
            0.0016604403499513865,
            0.0020129659678786993,
            0.002308456925675273,
            0.002431850880384445,
            0.002308456925675273,
            0.0020129659678786993,
            0.0016604403499513865,
            0.001333612366579473,
            0.0010629459284245968,
        ],
        [
            0.0008819324430078268,
            0.0010629459284245968,
            0.0012624080991372466,
            0.0014565379824489355,
            0.00160425144713372,
            0.0016609467566013336,
            0.00160425144713372,
            0.0014565379824489355,
            0.0012624080991372466,
            0.0010629459284245968,
            0.0008819324430078268,
        ],
    ]
)


def get_fiber_fractions(PSF, FIBER):
    """Given a PSF postage stamp observed in a GFA and a fiber profile, return the (PSF,ELG,BGS) fiber fractions."""
    PSFsum = np.sum(PSF)
    ELG = scipy.signal.convolve(PSF, _ELG_kernel, mode="same")
    BGS = scipy.signal.convolve(PSF, _BGS_kernel, mode="same")
    return (
        np.sum(PSF * FIBER) / PSFsum,
        np.sum(ELG * FIBER) / PSFsum,
        np.sum(BGS * FIBER) / PSFsum,
    )


class ETCExposure(object):
    """Utility class for reading the per-exposure json file written by the ETC."""

    class TimeSeries:
        pass

    utc_offset = -7
    mjd_epoch = np.datetime64("1858-11-17", "ms") + utc_offset * np.timedelta64(
        3600000, "ms"
    )
    onesec = np.timedelta64(1000, "ms")

    @staticmethod
    def get_timestamps(mjd0, dt):
        dt = np.asarray(dt)
        return (
            ETCExposure.mjd_epoch
            + mjd0 * np.timedelta64(86400000, "ms")
            + dt * ETCExposure.onesec
        )

    @staticmethod
    def load(night, expid, datadir=None):
        datadir = datadir or os.getenv("DESI_SPECTRO_DATA") or "."
        path = pathlib.Path(datadir)
        if not path.exists():
            raise RuntimeError(f"Invalid datadir: {datadir}")
        path = path / str(night)
        if not path.exists():
            raise RuntimeError(f"Invalid night: {night}")
        path = path / str(expid).zfill(8)
        if not path.exists():
            raise RuntimeError(f"Invalid expid: {expid} for {night}")

    def __init__(self, filename):
        """Initialize the ETC exposure data from a saved json file"""
        with open(filename) as f:
            self.data = json.load(f)
        for block in self.data.keys():
            if block in ("thru", "sky", "accum"):
                # Expand time series data into an attribute object.
                setattr(self, block, ETCExposure.TimeSeries())
                timeseries = getattr(self, block)
                mjd0 = self.data[block]["mjd0"]
                for k in self.data[block].keys():
                    if k == "mjd0":
                        continue
                    if k.startswith("dt"):
                        name = "t" + k[2:]
                        # Convert deltas into numpy 64-bit local timestamps.
                        setattr(
                            timeseries,
                            name,
                            ETCExposure.get_timestamps(mjd0, self.data[block][k]),
                        )
                    # Convert all timeseries values to float32 numpy arrays (including dt* arrays)
                    setattr(timeseries, k, np.array(self.data[block][k], np.float32))
            else:
                # Attach sub-dict as attribute of this object.
                setattr(self, block, self.data[block])
