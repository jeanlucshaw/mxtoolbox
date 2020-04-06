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


def pd_bin(dataframe, dimension, grid, func=np.nanmean):
    """
    Wrapper around xr_bin to allow bin averaging in pandas
    dataframes.

    dataframe [pandas dataframe]  : input data
    dimesion  [string]            : index along which to bin
    grid      [array-like]        : bin centers
    func      [callable]          : function to pass xrutils.xr_bin
    """
    index_names = dataframe.index.names
    dataset = dataframe.reset_index().set_index(dimension).to_xarray()
    dataset = xr_bin(dataset, dimension, grid, func=func)
    return dataset.to_dataframe().reset_index().set_index(index_names)


def xr_bin(ds, dim, binc, func=np.nanmean):
    '''
    Function xr_bin usage:  ds = xr_bin(ds, dim, binc, func=numpy.nanmean)

    Wrapper function for the groupby_bins xarray method. Meant for
    simply binning xarray ds to the values of dimension dim, and
    return values at bin centers binc.

    ds: input Dataset of DataArray
    dim: string: name of dimension along which to bin
    binc: array: bin centers
    func: callable: reducing function, defaults to numpy.nanmean
    '''
    edge = binc2edge(binc)
    dimorder = tuple(ds.coords)
    attributes = ds.attrs
    ds = (ds.groupby_bins(ds[dim], bins=edge, labels=binc)
          .reduce(func, dim=dim)
          .rename({dim+'_bins': dim})
          .transpose(*dimorder))
    ds.attrs = attributes

    return ds


def xr_filter(coord, val, fc, btype='high', axis=-1, order=1):
    """
    Function xrflt usage:   da  =  xrflt(coord,val,fc,btype='high',axis=-1,order=1)

    Summary:

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
    Tide killing Godin filter of dataarray along dimension 'tname'.
    """
    time_step = xrtimestep(dataarray, tname, 'second')
    flt_godin = (dataarray
                 .rolling({tname: int(24 * 3600 / time_step)}, center=True).mean()
                 .rolling({tname: int(24 * 3600 / time_step)}, center=True).mean()
                 .rolling({tname: int(25 * 3600 / time_step)}, center=True).mean()
                 .resample({tname: '1D'}).mean())

    return flt_godin


def xr_peaks(array, th_array, tname, two_sided=True, fp_kwargs=None):
    """
    Function xrPks usage:   out  =  xrPks(array,th_array,tname,two_sided=True,fp_kwargs=None)

    Summary:

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
