"""Shared utilities for the online exposure-time calculator.
"""
import numpy as np


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
