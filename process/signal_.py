"""
Analyses of 1D signal type datasets or multidimensional
arrays processed along one dimension. This includes
filtering and binning functions.
"""
import numpy as np
import xarray as xr
import scipy.signal as signal
from scipy.stats import mode
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from .math_ import xr_time_step
from .convert import binc2edge, bine2center
from pandas.tseries.frequencies import to_offset
import warnings

__all__ = ['ctd_flag_inversions',
           'ctd_flag_spikes',
           'lowess',
           'pd_bin',
           'pd_bin_2',
           'pd_at_var_stat',
           'regularize_xy',
           'running_mean',
           'xr_at_var_max',
           'xr_bin',
           'xr_bin_where',
           'xr_create_lagged',
           'xr_filtfilt',
           'xr_godin',
           'xr_peaks']


# Flag density inversions
def ctd_flag_inversions(dataset, density, threshold=0):
    """
    Flag water denser than deeper water in CTD cast.

    Parameters
    ----------
    dataset: xarray.dataset
        Of a single CTD cast.
    density: str
        Name of the density variable.
    threshold: float
        Amount by which data must be denser to get flagged.

    Returns
    -------
    1D array of bool
        Inverted density data marked as True.

    Note
    ----

       Data are expected to be arranged in order of monotonically
       increasing depth.

    """
    densities = dataset[density].values
    inverted = [(d_ > densities[i_:] + threshold).any()
                for (i_, d_)
                in enumerate(densities)]

    return np.array(inverted)


def ctd_flag_spikes(dataset,
                    salinity,
                    window=5,
                    window_size_type='pc',
                    n_sigma=1,
                    min_spike_size=0.5,
                    return_lowpass=False):
    """
    De-spike profile data using a two pass median filter test.

    Parameters
    ----------
    dataset : xarray.Dataset
        Containing a single CTD cast.
    salinity : str
        Name of the variable to de-spike.
    window : int or float
        Size of window.
    window_size_type : str
        Either `pc` (percentage of size) or `n` (number of data).
    n_sigma : float
        Fraction of local std to use as a spike threshold. (see Note)
    min_spike_size : float
        Keep only spikes larger than this value.
    return_lowpass : bool
        Return the low passed profile (2nd pass).

    Returns
    -------
    tuple : (1D array of bool) or (1D array of bool, 1D array of float)
        Values marked as spikes (and lowpassed profile).

    Note
    ----
       The algorithm is the following:

          * Low pass the profile with a median filter (LP)
          * Calculate the difference between data and LP (residual)
          * Calculate the local std for each window (local_std)
          * Flag  abs(residuals) > 1 * n_sigma * local_std
          * Replace flagged values with LP in a copy of the profile
          * Low pass the copy with a median filter (LP)
          * Calculate the difference between the copy and LP (residual)
          * Calculate the local std of the copy for each window (local_std)
          * Calculate the difference between the profile and LP (error)
          * Flag  abs(error) > 2 * n_sigma * local_std
          * Keep only flags where  abs(error) > min_spike_size

    """
    # Get dimension name
    if len(dataset[salinity].shape) == 1:
        dim = dataset[salinity].dims[0]
    else:
        raise ValueError('Profile variable must be 1D.')

    # Percentage of cast size converted to window size
    if window_size_type == 'pc':
        window = int(dataset[salinity].size * window / 100)

        # Unless this means the window is 1 data wide
        if window <= 1:
            window = 3

    # Working copy of data
    data = dataset[salinity].copy()

    """ First pass """

    # Moving average filter the profile
    kw_rolling = {dim: window, 'center': True, 'min_periods': np.floor(window / 2)}
    lowpass_profile = data.rolling(**kw_rolling).median()

    # Get difference of data to the moving average
    residual_profile = data - lowpass_profile

    # Get std of each window
    residual_std = residual_profile.rolling(**kw_rolling).std()

    # Identify first pass spikes
    spike_pass_1 = np.array(abs(residual_profile) > (n_sigma * residual_std))

    # Remove pass 1 spikes from working copy
    data.values[spike_pass_1] = lowpass_profile.values[spike_pass_1]

    """ Second pass """

    # Moving average filter the profile
    lowpass_profile = data.rolling(**kw_rolling).median()

    # Get difference of data to the moving average
    residual_profile = data - lowpass_profile

    # Get std of each window
    residual_std = residual_profile.rolling(**kw_rolling).std()

    # Error profile
    error_profile = dataset[salinity] - lowpass_profile

    # Identify first pass spikes
    spike_pass_2 = abs(error_profile) > (2 * n_sigma * residual_std)
    spike_pass_2 = spike_pass_2 & (abs(error_profile) > min_spike_size)

    # Return lowpassed salinity or not
    if return_lowpass:
        output = np.array(spike_pass_2), lowpass_profile
    else:
        output = np.array(spike_pass_2)

    return output


def lowess(*args, **kwargs):
    """
    Wrapper for lowess from statsmodels.

    This is lazy, but the import call for this function is
    hard to remember.

    Parameters
    ----------
    endog: 1D array
        Independent variable.
    exog: 1D array
        Dependent variable.
    frac: float
        Between 0 and 1. The fraction of the data used when
        estimating each y-value.

    Returns
    -------
    1D array
        The smoothed independent variable.
    """
    return lowess(*args, **{'return_sorted': False, **kwargs})


def pd_bin(dataframe, dim, binc, func=np.nanmean):
    """
    Bin values in pandas dataframe.

    Wrapper around xr_bin to allow bin averaging in pandas
    dataframes.

    Parameters
    ----------
    dataframe : pandas.Dataframe
        Dataframe to operate on.
    dim : str
        Dimension to operate along
    binc : array_like
        Bin centers.
    func : Object
        Function used to reduce bin groups.

    Returns
    -------
    pandas.Dataframe
        Input dataframe binned at `binc`.

    """
    index_names = dataframe.index.names
    dataset = dataframe.reset_index().set_index(dim).to_xarray()
    dataset = xr_bin(dataset, dim, binc, func=func)
    try:
        return dataset.to_dataframe().reset_index().set_index(index_names)
    except:
        return dataset.to_dataframe().reset_index()


def pd_bin_2(dataframe,
             bins,
             on,
             centers=True,
             func=np.mean):
    """
    Bin values in pandas dataframe using `cut`.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Data to bin.
    bins: 1D array
        Bin definitions.
    on: str
        Column to use as index.
    centers: bool
        `bins` is the bin centers (or edges).
    func: callable
        Apply to groupy object for aggregation.

    Returns
    -------
    pandas.Dataframe:
        Binned dataframe.

    """
    # Isolate series to bin along
    series = dataframe[on]

    # Manage bin edges/labels
    if centers:
        labels = bins
        bins = binc2edge(bins)
    else:
        labels = bine2center(bins)

    # Bin and format output
    dataframe.loc[:, 'bin'] = pd.cut(series, bins=bins, labels=labels)
    dataframe = dataframe.groupby('bin').apply(func)
    dataframe.loc[:, on] = labels
    dataframe = dataframe.reset_index()

    return dataframe


def pd_at_var_stat(dataframe, varname, func, interval, drop=True):
    """
    Find lines of `dataframe` where `varname` is equal to `func(varname)`.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Must have a time index.
    varname: str
        Column name on which to apply `func`.
    func: str or callable
        Passed to pandas.resample.agg, the bin reduction function.
    interval: str
        Bin time interval. For now only multiple of hours, days and months are
        supported. Weekly bins can be taken by entering `7D`. Daily bins start at
        midnight. Hourly bins start at 00:00:00 and increase by `interval`. Monthly
        bins start on the first day of each month.
    drop: bool
        Drop duplicate lines for equal values of func(var) in a given bin.

    Returns
    -------
    pandas.DataFrame:
        Input `dataframe` with all lines removed but those where the selection
        criterium is met. A `bin_starts` column is associate values with time
        intervals.
    """
    # Group dataframe by time intervals
    resampler = dataframe.resample(interval, label='left')

    # Aggregate groups using func
    df_varstat = resampler.agg(func)

    # This is needed for monthly intervals because the origin keyword seems to be missing
    if 'M' in interval:
        df_varstat.index = df_varstat.index + to_offset('1D')

    # Format aggregated variable for comparison to input dataframe
    df_varstat = df_varstat[varname].reset_index()
    df_varstat = df_varstat.rename(columns={'time': 'bin_starts', varname: 'varstat'}).set_index('bin_starts')

    # Create a coarse time index column for dataframe
    dataframe.loc[:, 'bin_starts'] = dataframe.index.copy()
    for value, indices in resampler.indices.items():
        dataframe.iloc[indices, -1] = value

    # This is needed for monthly intervals because the origin keyword seems to be missing
    if 'M' in interval:
        dataframe['bin_starts'] = dataframe['bin_starts'] + to_offset('1D')

    # Add variable statistic column
    df_match = dataframe.reset_index().set_index('bin_starts')
    df_match = df_match.join(df_varstat)

    # Make time the index once again
    df_match = df_match.reset_index().set_index('time')

    # Query for lines where var matches varstat
    df_match =  df_match.query('%s == varstat' % varname)

    # Optionnally drop duplicated
    if drop:
        df_match = df_match.drop_duplicates(subset='bin_starts')

    return df_match


def regularize_xy(x, y):
    """
    Force regular intervals of the coordinate in x,y data.

    Parameters
    ----------
    x, y: 1D array
        Independent and dependent variables.

    Returns
    -------
    x_reg, y_reg: 1D array
        Interpolated regular interval data.
    dx: type( diff(x) )
        The mode of `diff` applied to `x` used as the interval
        for `x_reg`.

    """
    # Find best regular interval
    dx, _ = mode(np.diff(x))
    dx = dx[0]

    # Construct regular coordinate
    x_r = np.arange(x[0], x[-1] + dx, dx)

    # Interpolate via xarray
    da = xr.DataArray(y, dims=['x'], coords={'x': x})
    da = da.interp(x=x_r)

    return da.x.values, da.values, dx


def running_mean(x, ws, *args, **kwargs):
    """
    Running mean using convolve.

    Parameters
    ----------
    x: 1D array
        The variable to smooth.
    ws: int
        The window size in data points.
    *args, **kwargs: arguments, keyword arguments
        Passed to `numpy.convolve`.

    Returns
    -------
    1D array:
        Smoothed variable.

    """
    return np.convolve(x, np.ones(ws) / ws, *args, **kwargs)


def xr_at_var_max(dataset, variable, dim=None):
    """
    Return values of all others at variable max.

    Parameters
    ----------
    dataset : xarray.Dataset
        On which to operate.
    variable : str
        Name of the selector variable.
    dim : str
        Dimension along which to get maximum.

    Returns
    -------
    xarray.Dataset
        Reduced input dataset.

    """
    # Avoid modifying original dataset
    output = dataset.copy()

    # Reduce to specified dimension
    if dim:
        # Remove any labels where variable is all NaN
        for d in output[variable].dims:
            output = output.dropna(dim=d, how='all', subset=[variable])

        # Index max of variable along dim 
        index = output[variable].argmax(dim=dim)

        # Reduce dataset to values at max for relevant dimsensions
        for key, value in output.data_vars.items():
            if dim in output[key].dims:
                output[key] = output[key].isel({dim: index})

    # Reduce to maximum point value
    else:
        # Run successively over all dimensions
        for d in dataset[variable].dims:
            output = xr_at_var_max(output, variable, dim=d)
    return output


def xr_bin(dataset, dim, bins, centers=True, func=np.nanmean):
    '''
    Bin dataset along `dim`.

    Convenience wrapper for the groupby_bins xarray method. Meant for
    simply binning xarray `dataset` to the values of dimension `dim`, and
    return values at bin centers (or edges) `bins`.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        Dataset to operate on.
    dim: str
        Name of dimension along which to bin.
    bins: array_like
        Bin centers or edges if `centers` is False.
    centers: bool
        Parameter `bins` is the centers, otherwise it is the edges.
    func: Object
        Function used to reduce bin groups.

    Returns
    -------
    xarray.Dataset
        Dataset binned at `binc` along `dim`.
    '''
    # Bin type management
    if centers:
        edge = binc2edge(bins)
        labels = bins
    else:
        edge = bins
        labels = bine2center(bins)

    # Skip for compatibility with DataArray
    if isinstance(dataset, xr.core.dataset.Dataset):
        # Save dimension orders for each variable
        dim_dict = dict()
        for key in dataset.keys():
            dim_dict[key] = dataset[key].dims

        # Save dataset attributes
        attributes = dataset.attrs

        # Save variable attributes
        var_attributes = dict()
        for v in dataset.data_vars:
            var_attributes[v] = dataset[v].attrs

        # Save variable attributes
        coord_attributes = dict()
        for c in dataset.coords:
            coord_attributes[c] = dataset[c].attrs

    # Avoids printing mean of empty slice warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

        # Bin reduction 
        output = (dataset.groupby_bins(dataset[dim], bins=edge, labels=labels)
                   .reduce(func, dim=dim)
                   .rename({dim+'_bins': dim}))

    # Skip for compatibility with DataArray
    if isinstance(dataset, xr.core.dataset.Dataset):
        # Restore dataset attributes
        output.attrs = attributes

        # Restore variable
        for v in output.data_vars:
            output[v].attrs = var_attributes[v]

        # Restore variable
        for c in output.coords:
            output[c].attrs = coord_attributes[c]

        # Restore dimension order to each variable
        for key, dim_tuple in dim_dict.items():
            if dim not in dim_tuple:
                output[key] = dataset[key]
            else:
                output[key] = output[key].transpose(*dim_tuple)

    return output


def xr_bin_where(dataset, field, selector, dim, binc, fill_na=-99999):
    """
    Bin select values in all variables based on one variable.

    Obtain the values of `field` chosen by `selector` and the
    associated values of all other variables at the same
    coordinates. For example, setting `selector` to `np.argmax`
    will return the binned maximum of `field` and the values of
    the other variables at the same coordinates (not necessarily
    their maximums).

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to operate on.
    field : str
        Name of variable off of which to base selection.
    selector : callable
        Returns index of selected value in each bin.
    dim : str
        Name of dimension along which to bin.
    binc : 1D array
        Bin centers.
    fill_na : numeric
        Replace missing data in `field` with this value
        prior to applying `selector`. Good values are function
        dependent. For example you could use -99999 for
        numpy.argmax and 99999 for numpy.argmin to not affect
        the outcome.

    Returns
    -------
    xarray.Dataset
        The bin selected dataset.

    Note
    ----

       The selector function is applied to the GroupBy objects
       after binning. It is expected to return the index of
       the selected value and not the selected value itself (e.g.
       `numpy.argmax` or `numpy.argmin`).
    """
    def _reductor(DATASET, FIELD=None, SELECTOR=None, DIM=None):
        return DATASET.isel(**{DIM: DATASET[FIELD].reduce(SELECTOR, dim=DIM)})

    # Save original dimension orders for each variable
    dim_dict = dict()
    for key in dataset.keys():
        dim_dict[key] = dataset[key].dims

    # Remove nan values
    output = dataset.fillna(fill_na)

    # Binning
    output = (output.groupby_bins(output[dim], binc2edge(binc), labels=binc)
              .apply(_reductor, FIELD=field, SELECTOR=selector, DIM=dim)
              .reset_coords(dim)
              .drop_vars(dim)
              .rename({dim+'_bins': dim}))

    # Restore nan values
    output = output.where(output != fill_na)

    # Transpose to original dimensions
    for key, dim_tuple in dim_dict.items():
        output[key] = output[key].transpose(*dim_tuple)

    # Pass on attributes
    output.attrs = dataset.attrs

    return output


def xr_create_lagged(dataset, variable, dim='time', sign_td=1, args_td=None, name=None):
    """
    Add lagged version of variable to xarray Dataset

    Parameters
    ----------
    dataset: xarray.Dataset
        On which to operate.
    variable: str
        Name of the variable to lag.
    dim: str
        Name of the time coordinate.
    sign_td: int
        Delay (1) of rush (-1) the signal.
    args_td: tuple
        Delay parameters. Unpacked as input to numpy.timedelta64.
    name: str
        Name of output lagged variable.

    Returns
    -------
    xarray.Dataset
        With the lagged variable added.

    """
    # Zero lag by default
    if args_td is None:
        args_td = (0, 'D')

    # Set default lagged variable name
    if name is None:
        name = '%s_lagged' % variable

    # Create shifted time
    shifted_dim = dataset[dim].values + sign_td * np.timedelta64(*args_td)

    # Create a shifted dataarray
    shifted_var = xr.DataArray(dataset[variable].values, dims=[dim], coords={dim: shifted_dim})

    # Reinterpolated to original time
    dataset[name] = shifted_var.interp({dim: dataset[dim]})

    # Save as attribute what the added lag is
    dataset[name].attrs['lag'] = 'Variable %s lagged by %d * %d %s' % (variable, sign_td, *args_td)

    return dataset


def xr_filtfilt(dataset, dim, cutoff, btype='low', order=2, vars_=None):
    """
    Pass xarray data through variable state forward-backward filter.

    Data is first gridded to a regular time vector going from the
    first element of `dataset[dim]` to its last element. The resolution
    used is the median time step of the input time coordinate. Forward-
    backward filtering doubles the window's order, so the input value
    is integer divided by two. Odd order values will result in filtering
    of order `int(order / 2)`.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input data.
    dim : str
        Dimension along which to filter.
    cutoff : float
        Filter cutoff specified in Hz.
    btype : str
        Filter type passed to scipy.signal.butter.
    order : int
        Filter order passed to scipy.signal.butter.
    vars_ : list of str
        Variables to filter, defaults to all.

    Returns
    -------
    xarray.Dataset
        The filtered input dataset
    """
    # Save the original time vector
    time_ungrid = dataset[dim].copy()

    # Grid the data
    time_step = xr_time_step(dataset, dim, 's')
    time_grid = np.arange(dataset[dim].values[0],
                          dataset[dim].values[-1],
                          np.timedelta64(int(time_step), 's'),
                          dtype=dataset[dim].dtype)
    dataset = dataset.interp({dim: time_grid})

    # Parameters
    fs = 1 / time_step
    fn = fs / 2

    # Create the filter function 
    b, a = sl.butter(int(order / 2), cutoff / fn, btype=btype, output="ba")
    filtfilt = lambda x : sl.filtfilt(b, a, x)

    # apply_ufunc interface
    if vars_ is None:
        vars_ = dataset.keys()

    output = dataset.copy()
    for var in vars_:
        output[var] = xr.apply_ufunc(filtfilt,
                                     dataset[var],
                                     input_core_dims=[[dim]],
                                     output_core_dims=[[dim]])

    # Regrid to original time vector
    output = output.interp({dim: time_ungrid})

    return output


def xr_godin(dataarray, tname):
    """
    Apply Godin filter to `dataarray` along time dimension.

    Godin filtering is meant to remove the semi-diurnal and
    diurnal components of tide. It consists of iteratively
    time averaging over 24, 24, and 25 hours, and then
    downsampling to the daily scale.

    Parameters
    ----------
    dataarray : xarray.DataArray
        DataArray on which to operate.
    tname : str
        Name of the time dimension.

    Returns
    -------
    xarray.DataArray
        Godin filtered DataArray.

    """
    # Save original dimension orders for each variable
    if type(dataarray) == 'xarray.core.dataset.Dataset':
        dim_dict = dict()
        for key in dataarray.keys():
            dim_dict[key] = dataarray[key].dims

    # Perform averaging
    time_step = xr_time_step(dataarray, tname, 'second')
    # Avoids printing mean of empty slice warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

        flt_godin = (dataarray
                     .rolling({tname: int(24 * 3600 / time_step)}, center=True).mean()
                     .rolling({tname: int(24 * 3600 / time_step)}, center=True).mean()
                     .rolling({tname: int(25 * 3600 / time_step)}, center=True).mean()
                     .resample({tname: '1D'}).mean())

    # Transpose to original dimensions
    if type(dataarray) == 'xarray.core.dataset.Dataset':
        for key, dim_tuple in dim_dict.items():
            flt_godin[key] = flt_godin[key].transpose(*dim_tuple)

    return flt_godin


# def xr_peaks(array, th_array, tname, two_sided=True, fp_kwargs=None):
#     """
#     Find peaks in xarray dataset.

#     !!!The UI here needs to be reworked and tested!!!

#     Wrapper of scipy.signal.find_peaks that takes as input the array in which
#     to find peaks, the variable threshold array 'th_array' and their matching
#     x coordinate name 'tname'. If two_sided is True, positive and negative peaks
#     above the threshold will be returned.

#     out is a tuple formed of DataArrays for high peaks, low peaks (two_sided=True), and
#     index of peaks in original array.
#     """

#     hpksi, props = signal.find_peaks(array.values, fp_kwargs)
#     hpks = xr.DataArray(array.values[hpksi],
#                         coords=[array[tname].values[hpksi]],
#                         dims=tname)
#     hpks = hpks.where(hpks > th_array.values[hpksi], drop=True)
#     out = (hpks,)
#     ipks = hpksi

#     if two_sided:
#         th_array = -th_array
#         lpksi, props = signal.find_peaks(-array, fp_kwargs)
#         lpks = xr.DataArray(array.values[lpksi],
#                             coords=[array[tname].values[lpksi]],
#                             dims=tname)
#         lpks = lpks.where(lpks < th_array.values[lpksi],
#                           drop=True)
#         out = out + (lpks,)
#         ipks = np.sort(np.hstack((ipks, lpksi)))

#     out = out + (ipks,)

#     return out

def xr_peaks(dataset, variable, troughs=False, **kwargs_fp):
    """
    Convenience xarray wrapper for scipy.signal.find_peaks.

    Parameters
    ----------
    dataset: xarray.Dataset
        In which to operate.
    variable: str
        Name of variable to find peaks in.
    **kwargs_fp: dict
        Keyword arguments passed to `find_peaks`.

    Returns
    -------
    x_p, y_p: 1D array
        Position and value of peaks.

    """
    # Get arrays of the coordinate and data variables
    dim = dataset[variable].dims
    if len(dim) == 1:
        dim = dim[0]
    else:
        raise TypeError('Variable must be one dimensional.')
    coord = dataset[dim].values
    x = dataset[variable].values

    # Function call
    peaks, _ = signal.find_peaks(x, **kwargs_fp)

    if troughs:
        troughs, _ = signal.find_peaks(-x, **kwargs_fp)
        peaks = np.sort([*peaks, *troughs])

    return coord[peaks], x[peaks]
