"""
Analyses of 1D signal type datasets or multidimensional
arrays processed along one dimension. This includes 
filtering and binning functions. 
"""
import numpy as np
import xarray as xr
import scipy.signal as signal
from .math import xr_time_step
from .convert import binc2edge

__all__ = ['pd_bin',
           'xr_bin',
           'xr_filter',
           'xr_godin',
           'xr_peaks']


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
    return dataset.to_dataframe().reset_index().set_index(index_names)


def xr_bin(dataset, dim, binc, func=np.nanmean):
    '''
    Bin dataset along `dim`.

    Convenience wrapper for the groupby_bins xarray method. Meant for
    simply binning xarray `dataset` to the values of dimension `dim`, and
    return values at bin centers `binc`.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to operate on.
    dim: str
        Name of dimension along which to bin.
    binc: array_like
        Bin centers.
    func: Object
        Function used to reduce bin groups.

    Returns
    -------
    xarray.Dataset
        Dataset binned at `binc` along `dim`.
    '''
    edge = binc2edge(binc)
    dimorder = tuple(dataset.coords)
    attributes = dataset.attrs
    dataset = (dataset.groupby_bins(dataset[dim], bins=edge, labels=binc)
          .reduce(func, dim=dim)
          .rename({dim+'_bins': dim})
          .transpose(*dimorder))
    dataset.attrs = attributes

    return dataset


def xr_filter(coord, val, fc, btype='high', axis=-1, order=1):
    """
    Apply zero phase shift butterworth filter to xarray.

    !!!This needs to be tested and the UI needs a rework!!!

    Wrapper of scipy.signal.filtfilt that takes for input a coordinate
    DataArray and a val DataArray. Returns a DataArray of the filtered
    val values along the initial coordinate. Cutoff frequency should be
    supplied in seconds -1.
    """

    # Replace nan by interpolated values
    val = val.interpolate_na(coord.name)

    # Drop any label along coord where 
    val = val.where(np.isfinite(val), drop=True)
    crd = coord.where(np.isfinite(val), drop=True)

    # Parameters
    dt = np.diff(coord).mean().tolist() / 10**9
    fs = 1/dt
    fn = fs/2

    # Filtering
    b, a = signal.butter(order, fc/fn, btype=btype, output="ba")
    flt = signal.filtfilt(b, a, val.values, axis=axis)

    # Interpolate to original grid
    flt = xr.DataArray(flt, dims=[coord.name], coords=[crd.values])
    flt = flt.interp({coord.name: coord.values})

    return flt


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
    time_step = ps.xr_time_step(dataarray, tname, 'second')
    flt_godin = (dataarray
                 .rolling({tname: int(24 * 3600 / time_step)}, center=True).mean()
                 .rolling({tname: int(24 * 3600 / time_step)}, center=True).mean()
                 .rolling({tname: int(25 * 3600 / time_step)}, center=True).mean()
                 .resample({tname: '1D'}).mean())

    return flt_godin


def xr_peaks(array, th_array, tname, two_sided=True, fp_kwargs=None):
    """
    Find peaks in xarray dataset.

    !!!The UI here needs to be reworked and tested!!!

    Wrapper of scipy.signal.find_peaks that takes as input the array in which
    to find peaks, the variable threshold array 'th_array' and their matching
    x coordinate name 'tname'. If two_sided is True, positive and negative peaks
    above the threshold will be returned.

    out is a tuple formed of DataArrays for high peaks, low peaks (two_sided=True), and
    index of peaks in original array.
    """

    hpksi, props = signal.find_peaks(array.values, fp_kwargs)
    hpks = xr.DataArray(array.values[hpksi],
                        coords=[array[tname].values[hpksi]],
                        dims=tname)
    hpks = hpks.where(hpks > th_array.values[hpksi], drop=True)
    out = (hpks,)
    ipks = hpksi

    if two_sided:
        th_array = -th_array
        lpksi, props = signal.find_peaks(-array, fp_kwargs)
        lpks = xr.DataArray(array.values[lpksi],
                            coords=[array[tname].values[lpksi]],
                            dims=tname)
        lpks = lpks.where(lpks < th_array.values[lpksi],
                          drop=True)
        out = out + (lpks,)
        ipks = np.sort(np.hstack((ipks, lpksi)))

    out = out + (ipks,)

    return out
