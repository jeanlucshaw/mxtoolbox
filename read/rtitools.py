import datetime
import glob
import os
import xarray as xr
import numpy as np
from scipy.stats import circmean
from tqdm import tqdm
from rti_python.Codecs.BinaryCodec import BinaryCodec
from rti_python.Ensemble.EnsembleData import *
from mxtoolbox.read.adcp import adcp_init

__all__ = ['index_rtb_data',
           'load_rtb_binary',
           'read_rtb_ensemble',
           'read_rtb_file']


# Constants
DELIMITER = b'\x80' * 16


def index_rtb_data(file_path):
    """
    Read binary as byte stream. Find ensemble locations and sizes.

    Parameters
    ----------
    file_path : str
        File path and name.

    Returns
    -------
    1D array
        Ensemble start index values.
    1D array
        Ensemble data lengths.
    1D array
        Data as byte stream.

    """
    # Open binary
    with open(file_path, 'rb') as df:

        # Read data file
        data = df.read()
        ensemble_starts = []
        ensemble_lengths = []

        # Get payload size of first ensemble
        payloadsize = int.from_bytes(data[24: 27],'little')

        # Get individual ensemble starts and lengths
        ii = 0
        while ii < len(data) - payloadsize - 32 - 4:
            if data[ii: ii + 16] == DELIMITER:
                ensemble_starts.append(ii)
                ensemble_lengths.append(payloadsize + 32 + 4)

                # Increment by payload size, plus header plus checksum
                ii  +=  payloadsize +   32  +   4
            else:
                print("Data format bad")
                break

            # Get payload size of next ensemble
            payloadsize =   int.from_bytes(data[ii+24: ii+27],'little')

    return ensemble_starts, ensemble_lengths, data


def load_rtb_binary(files):
    """
    Read Rowetech RTB binary ADCP data to xarray.

    Paramters
    ---------
    files : str or list of str
        File name or expression designating .ENS files, or list or file names.

    Returns
    -------
    xarray.Dataset
        ADCP data as organized by mxtoolbox.read.adcp.adcp_init .

    """

    # Make list of files to read
    if isinstance(files, str):
        adcp_files = glob.glob(files)
    else:
        adcp_files = files

    # Make xarray from file list
    if len(adcp_files) > 1:
        xarray_datasets = [read_rtb_file(f) for f in adcp_files if f[-3:]=='ENS']
        ds = xr.concat(xarray_datasets, dim='time')
    else:
        ds = read_rtb_file(*adcp_files)

    return ds

def read_rtb_file(file_path):
    """
    Read data from one RTB .ENS file into xarray.

    Parameters
    ----------
    file_path : str
        File name and path.

    Returns
    -------
    xarray.Dataset
        As organized by mxtoolbox.read.adcp.adcp_init .

    """
    # Index the ensemble starts
    idx, enl, data = index_rtb_data(file_path)

    chunk = data[idx[0]: idx[1]]
    if BinaryCodec.verify_ens_data(chunk):
        ens     =   BinaryCodec.decode_data_sets(chunk)

    # Init Dataset
    Nbins = ens.EnsembleData.NumBins
    Nbeam = ens.EnsembleData.NumBeams
    bin_sz = ens.AncillaryData.BinSize
    Nens = len(idx)
    ds = adcp_init(Nbins,Nens,Nbeam)
    time = np.empty(Nens, dtype='datetime64[ns]')

    # Read and store ensembles
    with tqdm(total=len(idx)-1,desc="Processing "+file_path,unit=' ensembles') as pbar:
        for ii in range( len(idx) ):

            # Get data binary chunck for one ensemble
            chunk   =   data[ idx[ii]:idx[ii] + enl[ii] ]

            # Check that chunk looks ok
            if BinaryCodec.verify_ens_data(chunk):

                # Decode data variables
                ens = BinaryCodec.decode_data_sets(chunk)

                CORR = np.array(ens.Correlation.Correlation)
                AMP = np.array(ens.Amplitude.Amplitude)
                PG = np.array(ens.GoodEarth.GoodEarth)

                time[ii] = ens.EnsembleData.datetime()
                ds.u[:, ii] = np.array(ens.EarthVelocity.Velocities)[:, 0]
                ds.v[:, ii] = np.array(ens.EarthVelocity.Velocities)[:, 1]
                ds.w[:, ii] = np.array(ens.EarthVelocity.Velocities)[:, 2]
                ds.e[:, ii] = np.array(ens.EarthVelocity.Velocities)[:, 3]
                ds.temp[ii] = ens.AncillaryData.WaterTemp
                ds.dep[ii] = ens.AncillaryData.TransducerDepth
                ds.heading[ii] = ens.AncillaryData.Heading
                ds.pitch[ii] = ens.AncillaryData.Pitch
                ds.Roll[ii] = ens.AncillaryData.Roll
                ds.corr[:, ii] = np.nanmean(CORR, axis=-1)
                ds.amp[:, ii] = np.nanmean(AMP, axis=-1)
                ds.pg[:, ii] = PG[:, 3]

                # Bottom track data
                if ens.IsBottomTrack:
                    ds.u_bt[ii] = np.array(ens.BottomTrack.EarthVelocity)[0]
                    ds.v_bt[ii] = np.array(ens.BottomTrack.EarthVelocity)[1]
                    ds.w_bt[ii] = np.array(ens.BottomTrack.EarthVelocity)[2]
                    ds.e_bt[ii] = np.array(ens.BottomTrack.EarthVelocity)[3]
                    ds.pg_bt[:,ii] = np.asarray(ens.BottomTrack.BeamGood)
                    ds.corr_bt[:,ii] = np.asarray(ens.BottomTrack.Correlation)
                    ds.range_bt[:,ii] = np.asarray(ens.BottomTrack.Range)
            pbar.update(1)

    # Determine up/down configuration
    # These are saved as strings because netcdf4 does not support booleans
    mroll = np.abs(180 * circmean(np.pi * ds.Roll.values / 180) / np.pi)
    if mroll >= 0 and mroll < 30:
        ds.attrs['looking'] = 'up'
    else:
        ds.attrs['looking'] = 'down'

    # Determine bin depths
    # xarray concatenation fails if z is too different so rounding to cm
    if ds.attrs['looking']=='up':
        z = np.asarray(dep.mean()
                       - ens.AncillaryData.FirstBinRange
                       - np.arange(0, Nbins * bin_sz, bin_sz)).round(2)
    else:
        z = np.asarray(ens.AncillaryData.FirstBinRange
                       + np.arange(0, Nbins * bin_sz, bin_sz)).round(2)

    # Get beam angle
    if ens.EnsembleData.SerialNumber[1] in '12345678DEFGbcdefghi':
        ds.attrs['angle'] = 20
    elif ens.EnsembleData.SerialNumber[1] in 'OPQRST':
        ds.attrs['angle'] = 15
    elif ens.EnsembleData.SerialNumber[1] in 'IJKLMNjklmnopqrstuvwxy':
        ds.attrs['angle'] = 30
    elif ens.EnsembleData.SerialNumber[1] in '9ABCUVWXYZ':
        ds.attrs['angle'] = 0
    else:
        raise ValueError("Could not determine beam angle.")

    # Manage coordinates and remaining attributes
    ds = ds.assign_coords(z=z, time=time, Nbeam=np.arange(Nbeam))
    ds.attrs['binSize'] = bin_sz
    ds.attrs['serial'] = ens.EnsembleData.SerialNumber
    ds.attrs['freqHz'] = ens.SystemSetup.WpSystemFreqHz

    return ds


def read_rtb_ensemble(file_path ,N=0):
    """
    Read one ensemble from a RTB .ENS file.

    Parameters
    ----------
    file_path : str
        Name and path of the RTB file.
    N : int
        Index value of the ensemble to read.

    Returns
    -------
    rti_python.Ensemble.Ensemble
        Ensemble data object.

    """
    ensemble_starts, ensemble_lengths, data = index_rtb_data(file_path)

    chunk = data[ensemble_starts[N]: ensemble_starts[N] + ensemble_lengths[N]]
    if BinaryCodec.verify_ens_data(chunk):
        ens = BinaryCodec.decode_data_sets(chunk)
    else:
        ens = []

    return ens
