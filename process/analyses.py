"""
Complex analyses transforming data into other data.
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import leastsq
from .math_ import f_gaussian, f_sine, xr_time_step, xr_unique

__all__ = ['principal_modes',
           'gaussian_smoothing',
           'lsq_curve_fit',
           'pd_regression_statistics',
           'xr_cross_correlate',
           'xr_time_aht']

def principal_modes(args):
    """
    Perform eigenvalue decomposition using numpy.

    Given a dataset described by `n` vectors of length
    `m`, the principal modes of variation can be extracted
    using eigenvalue decomposition. The `n` vectors are
    first detrended by subtracting the mean. A covariance
    matrix `R` is then calculated as,

    .. math::

       \\mathbf{R} = \\mathbf{F}\\cdot\\mathbf{F'}

    where :math:`F` is a matrix whose columns are formed from
    the `n` detrended input vectors. This matrix is then
    passed to `numpy.linalg.eig` whose outputs `w` and `v`
    are returned.

    In addition, the matrix :math:`\\mathbf{v}` is
    inverted and returned as :math:`\\mathbf{B}` to provide
    a basis change matrix into principal mode space. The
    orginal coordinates may then be rotated into principal
    mode space by arranging them into the `n` by `m` matrix
    :math:`O` by multiplication,

    .. math::

       \\mathbf{C} = \\mathbf{B}\\cdot\\mathbf{O}

    we get the matrix `n` by `m` matrix :math:`c` whose rows
    are the rotated vectors.

    Parameters
    ----------
    args : tuple
        Vectors of length m.

    Returns
    -------
    vals : array (n)
        Eigenvalues of the covariance matrix.
    vecs : ndarray (n by n)
        Eigenvectors (columns).
    B : ndarray (n by n)
        Basis change matrix.

    """
    F = np.array(args).T

    # Detrend
    F -= np.nanmean(F, axis=0)

    # Covariance matrix
    R = F.T @ F

    # Eigenvalues/vectors
    vals, vecs = np.linalg.eig(R)

    # Basis change matrix
    B = np.linalg.inv(vecs)

    return vals, vecs, B


def gaussian_smoothing(xs, ys, zs, xg, yg, XR=1., YR=1.):
    """
    Interpolates scattered data to grid via Gaussian smoothing.

    Simplified version of the Barnes smoothing algorithm. This
    routine computes a value for each grid point by averaging
    all scattered data are weighted by their distance to the grid
    point. The weighing function for scatter point `i` is,

    .. math::

       w_i = \\text{exp}\\left[-\\left(\\frac{xg-xs_i}{XR}\\right)^2 +
                        \\left(\\frac{yg-ys_i}{YR}\\right)^2\\right]

    Parameters
    ----------
    xs, ys : 1D array
        Coordinates of the scattered data.
    zs : 1D array
        Values of the scattered data.
    xg, yg : 2D array
        Grid coordinates.
    XR, YR : float
        Horizontal and vertical search radius.

    Returns
    -------
    zg : 2D array
        Scattered data interpolated to grid.

    """
    zg = np.zeros_like(xg)

    # Interpolate on grid
    if xg.size < xs.size:
        # If xg.size << than xs.size this is faster
        for i in range(xv.size):
            for j in range(yv.size):
                dx = (xg[j, i] - xs) / XR
                dy = (yg[j, i] - ys) / YR
                w = np.exp(-(dx ** 2 + dy ** 2))
                zg[j, i] = np.matmul(w, zs) / w.sum()
    else:
        # If xg.size >> than xs.size this is faster
        z_sum = np.zeros_like(zg)
        w_sum = np.zeros_like(zg)
        for i in range(xs.size):
            dx = (xg - xs[i]) / XR
            dy = (yg - ys[i]) / YR
            w = np.exp(-(dx ** 2 + dy ** 2))
            z_sum += w * zs[i]
            w_sum += w
        zg = z_sum / w_sum

    return zg


def lsq_curve_fit(model, x, y, *parameters):
    """
    Least squares fitting using scipy.optimize.

    Formatting input for scipy.optimize.leastsq can
    be confusing. This wrapper serves the purpose of
    not having to remember how it works.

    Let :math:`y=f(x)` and `g` be a model for `f` 
    dependent on a variable number of parameters
    :math:`\\mathbf{a}`. Then this routine requires
    requires `model` to be a function object taking
    as arguments `x`, :math:`a_1, a_2, ..., a_n` in this
    order, and `*parameters` to be ordered initial guesses
    for :math:`\\mathbf{a}`.

    The following predefined models can serve as examples
    and can be called by setting `model` to the function
    name.

    * mxtoolbox.process.math.f_gaussian
    * mxtoolbox.process.math.f_sine


    Parameters
    ----------
    model : function object or str
        Must be of form `y = model(x, a_1, a_2, ...)`
    x, y : 1D array
        Dependent and independent variables.
    parameters : tuple of floats
        Ordered initial guesses for :math:`\\mathbf{a}`

    Returns
    -------
    fit_values : 1D array
        model(`x`, a_1, a_2, ...) using optimized parameters.
    fit_parameters : 1D array
        Optimized model parameters.
    """
    # Switch between predefined models
    if model == 'gaussian':
        _model = f_gaussian
    elif model == 'sine':
        _model = f_sine
    else:
        _model = model

    # Formatted for leastsq
    def residuals(parameters, y, x):
        return y - _model(x, *parameters)

    # Perform fit
    fit_parameters, _ = leastsq(_residuals, parameters, args=(y, x))

    # Calculate fit y values
    fit_values = _model(x, *fit_parameters)

    return fit_values, fit_parameters


def pd_regression_statistics(dataframe, xname, yname):
    """
    Print analysis of variance on linear regression.

    Print analysis of variance (ANOVA) table for a linear regression
    drawn for yname from xname, two columns of dataframe. The independent
    variable is considered to be xname.

    Parameters
    ----------
    dataframe : pandas.Dataframe
        The dataframe to operate on.
    xname : str
        Column name of the independent variable.
    yname : str
        Column name of the dependent variable.

    References
    ----------

    Van Emden, Helmut (2008)
    Statistics for terrified biologists,
    Blackwell publishing
    p. 262
    """
    x = dataframe.get([xname])
    y = dataframe.get([yname])
    n = x.size

    # Regression coefficient
    b = np.polyfit(x.values.flatten(), y.values.flatten(), 1)[0]

    # Deviation from regression line
    sum_of_sq_mean_x = b ** 2 * ((x - x.mean()) ** 2).sum()

    # Deviation from mean y
    sum_of_sq_mean_y = (y ** 2).sum() - (y.sum()) ** 2 / n

    # Residual variation
    sum_of_sq_mean_residual = sum_of_sq_mean_y.values - sum_of_sq_mean_x.values

    # Degrees of freedom
    df_b = 1
    df_r = n - df_b - 1

    # Mean squares
    mean_sq_b = sum_of_sq_mean_x / df_b
    mean_sq_res = sum_of_sq_mean_residual / df_r

    # Variation ratio
    f_statistic = mean_sq_b / mean_sq_res

    # Readout
    print('\nAnalysis of variance table\n')
    print('Independent variable: %s' % xname)
    print('Dependent   variable: %s\n' % yname)
    print("%15s\t%10s\t%10s\t%10s\t%10s" % ('',
                                            'd.f.',
                                            'sum. sq.',
                                            'mean sq.',
                                            'var. ratio'))
    print("%s\t%10d\t%10.2f\t%10.2f\t%10.3f" % ('Regression (b):'.ljust(15),
                                                df_b,
                                                sum_of_sq_mean_x,
                                                sum_of_sq_mean_x / df_b,
                                                f_statistic))
    print("%s\t%10d\t%10.2f\t%10.2f\t%10s" % ('Residual:'.ljust(15),
                                              df_r,
                                              sum_of_sq_mean_residual,
                                              mean_sq_res,
                                              'xx'))
    print("%s\t%10d\t%10.2f\t%10s\t%10s" % ('Total:'.ljust(15),
                                            n - 1,
                                            sum_of_sq_mean_y,
                                            'xx',
                                            'xx'))


def xr_cross_correlate(dataarray_a, dataarray_b, coord='time'):
    """
    Compute cross correlation between DataArrays dataarray_a and dataarray_b.
    
    The two input arrays can contain NaN values and be of different
    lengths but they must have the same coordinate resolution
    and coordinates must match on overlap. Resample prior to
    using if necessary.

    Parameters
    ----------
    dataarray_a, dataarray_b : xarray.DataArray
        Arrays to correlate.
    coord : str
        Name of dimension along which to operate.

    Returns
    -------
    tau : float
        dataarray_a shifted forward by `tau` matches dataarray_b
    r2 : float
        Variance explained at maximum cross correlation.
    dataarray_c : xarray.DataArray
        dataarray_b shifted to maximum correlation.
    cc : 1D array
        Cross correlation vector.

    """
    a = dataarray_a.values.copy() 
    b = dataarray_b.values.copy()
    t = dataarray_b[coord].values.copy()
    
    # Missing values to zero
    amask, bmask = np.isnan(a), np.isnan(b)
    a[amask] = 0
    b[bmask] = 0

    # Calculate cross-correlation
    cc = np.correlate(a, b, 'full')
    dn = (b.size - cc.argmax() - 1)
    tau = dn * np.diff(t).mean()

    # Pad with zeros to same size and calculate correlation after lag
    a_pad = np.hstack((np.zeros(b.size - 1), a, np.zeros(b.size - 1)))
    b_pad = np.roll(np.hstack((b, np.zeros(a.size + b.size - 2))), cc.argmax())
    r2 = np.corrcoef(a_pad, b_pad)[0, 1] ** 2

    # Mask values with missing values in either input
    a_pad_mask = np.hstack((np.zeros(b.size -1, dtype='bool'),
                            amask,
                            np.zeros(b.size -1, dtype='bool')))
    b_pad_mask = np.hstack((bmask,
                            np.zeros(a.size + b.size - 2, dtype='bool')))
    b_pad_mask = np.roll(b_pad_mask, cc.argmax())
    b_pad[a_pad_mask | b_pad_mask] = np.nan

    # Remove pads
    c = np.roll(b_pad, -cc.argmax())[:b.size]

    # Format shifted array to dataarray
    dataarray_c = xr.DataArray(c, dims=[coord], coords={coord: t - tau})

    return tau, r2, dataarray_c, cc


def xr_time_aht(sl, h='h', period=12.40):
    '''
    Function xr_t_aht summary:

    A much simplified version of libmx.physics.ocn.t_aht_ast

    Serves as validation of the aforementioned code and as a fast
    way to get time aht from a libmx.physics.xri.sealev xarray
    object.

    Parameters:

    sl: sealev object
    h: either 'h' or 'hp' to use measure or predicted sea level
    period: defaults to 12.40 hours (semidiurnal). This adjusts
    the required distance between high tides as well as the filtering
    frequency.
    '''
    # regularise time grid
    sl = xr_unique(sl, 'time')
    sl['hp'] = sl.hp.interpolate_na(dim='time')
    sl['h'] = sl.h.where(np.isfinite(sl.h), sl.hp)
    dt = np.median(np.diff(sl.time.values.tolist()) / 10**9)
    t_grid = pd.date_range(start=sl.time.values[0],
                           end=sl.time.values[-1], freq='%ds' % dt)
    sl = sl.interp(time=t_grid)

    # Filter and find high tide indices
    hflt = xrflt(sl.time, sl[h] - sl[h].mean(), 10
                 / (period * 3600), btype='low')
    I_pks, _ = find_peaks(hflt, distance=int(0.8*period*3600/dt))

    # Remove peaks within 1 period of dataset borders
    d_period = int(period * 3600 / dt)
    I = np.flatnonzero(np.logical_and(I_pks > d_period/2,
                                      I_pks < sl.time.size - d_period/2))
    I_pks = I_pks[I]

    # Numerical values of time after first high tide
    aht = (np.array((sl.time-sl.time[I_pks[0]]).values.tolist())
           / (3600 * 10**9))
    sl['aht'] = ('time', aht)

    # Subtract time of the previous high tide
    for (start, end) in zip(I_pks[0:-1], I_pks[1:]):
        sl.aht[start: end] = sl.aht[start: end] - sl.aht[start]

    # Manage data before first high tide
    sl.aht[0: I_pks[0]] = (sl.aht[0: I_pks[0]] + period) % period

    # Manage data after last high tide
    sl.aht[I_pks[-1]:] = sl.aht[I_pks[-1]:] - sl.aht[I_pks[-1]]

    return sl, I_pks
