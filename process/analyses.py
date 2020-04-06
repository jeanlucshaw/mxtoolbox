"""
Complex analyses transforming data into other data.
"""
import numpy as np
import pandas as pd
import xarray as xr
from .math import xr_time_step, xr_unique

__all__ = ['eigenvalues',
           'gaussian_smoothing',
           'fit_lsq',
           'pd_regression_statistics',
           'xr_cross_correlate',
           'xr_time_aht']

def eigenvalues(args):
    """

    Function eigDec summary:

        Find the principal directions of variation of an ND
        data scatter and return the basis change matrix.

    Usage:

        vals,vecs,B =   eigDec(args)

    Input:

        args:   tuple of same length vectors to decompose

    Output:

        vals:   sorted eigenvalues of the covariance matrix

        vecs:   sorted eigenvectors (columns)

        B   :   basis change matrix from prior to principal
                directions of variation.
    """
    F = np.array(args).T

    # Detrend
#    F   =   F-np.tile(np.nanmean(F,axis=-1),(F.shape[-1],1)).T

    # Covariance matrix
    R = F.T @ F

    # Eigenvalues/vectors
    vals, vecs = np.linalg.eig(R)
    I = np.argsort(vals)[::-1]
    vecs = vecs[:,I]
    vals = vals[I]

    # Basis change matrix
    B = np.linalg.inv(vecs)

    return vals, vecs, B


def gaussian_smoothing(xs, ys, zs, xg, yg, XR=1, YR=1, ITERS=1):
    """
    Gaussian smoothing filter of scattered data onto
    gridded coordinates. Simplified version of the 
    Barnes smoothing algorithm.
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


def fit_lsq(function, parameters, y, x = None):
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    if x is None: x = np.arange(y.shape[0])
    p = [param() for param in parameters]
    return optimize.leastsq(f, p)


def pd_regression_statistics(dataframe, xname, yname):
    """
    Print analysis of variance (ANOVA) table for a linear regression
    drawn for yname from xname, two columns of dataframe. The independent
    variable is considered to be xname.

    This is based on an example from:

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


def xr_cross_correlate(da_a, da_b, coord='time'):
    """
    Cross correlation between dataarrays da_a and da_b. The
    two input arrays can contain NaN values and be of different
    lengths but they must have the same coordinate resolution
    and coordinates must match on overlap. Resample prior to
    using if necessary.

    Outputs
        tau: by how much da_a must be shifted forward
             to match da_b.

        r2: variance explained at maximum cross correlation

        da_c: da_b shifted to max correlation

        cc: cross correlation vector
    """
    a, b, t = da_a.values.copy(), da_b.values.copy(), da_b[coord].values.copy()

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
    da_c = xr.DataArray(c, dims=[coord], coords={coord: t - tau})

    return tau, r2, da_c, cc


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
