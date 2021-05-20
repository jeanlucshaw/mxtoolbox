"""
Command line utility to collect cnv files into a CF netCDF file.

TODO:

* Add user grid command line option
* Implement binning to user grid
"""
import argparse
import xarray as xr
import mxtoolbox.read as rd
import mxtoolbox.process as ps
import numpy as np
import os
from datetime import datetime

# Parameters
max_depth = 200

# Long, standard, units
seabird_std_names = {'tv290C': 'sea_water_temperature',
                     't090C': 'sea_water_temperature',
                     'prdM': 'sea_water_pressure',
                     'sal00': 'sea_water_practical_salinity',
                     'c0S/m': 'sea_water_electrical_conductivity',
                     'sigma-t00': 'sea_water_density',
                     'flTC7': 'mass_concentration_of_chlorophyll_in_sea_water',
                     'ox': 'mass_concentration_of_chlorophyll_in_sea_water',
                     'AroFTox': 'volume_fraction_of_oxygen_in_sea_water',
                     'flag': 'flags',
                     'scan': 'sample_id',
                     'timeK': 'time'}


def time_series_init(time, series_metadata=None):
    """
    Return empty xarray shell in standardized format.

    Initializes many attributes required or recommended by
    CF-1.8 conventions. If used by others that the author,
    change the default values of the `source` and `contact`
    attributes.

    Parameters
    ----------
    time : int or 1D array
        Number of time steps or time vector.
    series_metadata : dict of list
        Contains the keys `names`, `seabird_names` and `units`. This
        determines the profile variables which will be initialized.

    Returns
    -------
    xarray.Dataset
        An empty dataset ready for data input.

    See Also
    --------

       * mxtoolbox.read.text.read_cnv_metadata
       * mxtoolbox.read.text.pd_read_cnv

    """
    # Take inputs as coordinate sizes or vectors
    if isinstance(depth, int):
        z = np.arange(depth)
    else:
        z = depth
    size_depth = z.size
    if isinstance(time, int):
        t = np.empty(time, dtype='datetime64[ns]')
    else:
        t = time 
    size_time = t.size

    # Initialize 1D and 2D blank arrays
    blank_timeseries = np.nan * np.ones(size_time)

    # Select variables to keep
    if series_metadata is None:
        series_vars = ['Temperature', 'Salinity']
        series_units = ['[deg C]', '[PSU]']
        series_sb_names = ['tv290', 'sal00']
    else:
        series_vars = series_metadata['names']
        series_units = series_metadata['units']
        series_sb_names = series_metadata['seabird_names']

    series_blanks = dict()
    for v_, u_ in zip(series_vars, series_units):
        series_blanks[v_] = (['time'], blank_timeseries.copy())

    ds = xr.Dataset(
        data_vars = {**series_blanks,
                     'lon': (['time'], blank_timeseries.copy()),
                     'lat': (['time'], blank_timeseries.copy()),
                     'uship': (['time'], blank_timeseries.copy()),
                     'vship': (['time'], blank_timeseries.copy())},
        coords = {'time': t},
        attrs = {'Conventions': 'CF-1.8',
                 'title': '',
                 'institution': '',
                 'source': 'CTD data, processed with https://github.com/jeanlucshaw/mxtoolbox',
                 'description': '',
                 'history': '',
                 'platform': '',
                 'bin_size': '',
                 'looking': '',
                 'instrument_serial': '',
                 'contact': 'Jean-Luc.Shaw@dfo-mpo.gc.ca'})

    # Add metadata
    for v_, u_, sb_ in zip(series_vars, series_units, series_sb_names):
        ds[v_].attrs['units'] = u_
        ds[v_].attrs['long_name'] = v_

        if sb_ in seabird_std_names:
            ds[v_].attrs['standard_name'] = seabird_std_names[sb_]
        else:
            print('No CF name for variable: %s, %s' % (v_, sb_))

    return ds


def profile_time_series_init(depth,
                             time,
                             profile_metadata=None):
    """
    Return empty xarray shell in standardized format.

    Initializes many attributes required or recommended by
    CF-1.8 conventions. If used by others that the author,
    change the default values of the `source` and `contact`
    attributes.

    Parameters
    ----------
    depth : int or 1D array
        Number of vertical bins or vertical bin vector.
    time : int or 1D array
        Number of time steps or time vector.
    profile_metadata : dict of list
        Contains the keys `names`, `seabird_names` and `units`. This
        determines the profile variables which will be initialized.

    Returns
    -------
    xarray.Dataset
        An empty dataset ready for data input.

    See Also
    --------

       * mxtoolbox.read.text.read_cnv_metadata
       * mxtoolbox.read.text.pd_read_cnv

    """
    # Take inputs as coordinate sizes or vectors
    if isinstance(depth, int):
        z = np.arange(depth)
    else:
        z = depth
    size_depth = z.size
    if isinstance(time, int):
        t = np.empty(time, dtype='datetime64[ns]')
    else:
        t = time 
    size_time = t.size

    # Initialize 1D and 2D blank arrays
    blank_profiles = np.nan * np.ones((size_depth, size_time))
    blank_timeseries = np.nan * np.ones(size_time)

    # Select variables to keep
    if profile_metadata is None:
        profile_vars = ['Temperature', 'Salinity']
        profile_units = ['[deg C]', '[PSU]']
        profile_sb_names = ['tv290', 'sal00']
    else:
        profile_vars = profile_metadata['names']
        profile_units = profile_metadata['units']
        profile_sb_names = profile_metadata['seabird_names']

    profile_blanks = dict()
    for v_, u_ in zip(profile_vars, profile_units):
        profile_blanks[v_] = (['z', 'time'], blank_profiles.copy())
    
    ds = xr.Dataset(
        data_vars = {**profile_blanks,
                     'lon': (['time'], blank_timeseries.copy()),
                     'lat': (['time'], blank_timeseries.copy()),
                     'uship': (['time'], blank_timeseries.copy()),
                     'vship': (['time'], blank_timeseries.copy())},
        coords = {'z' : z,
                  'time': t},
        attrs = {'Conventions': 'CF-1.8',
                 'title': '',
                 'institution': '',
                 'source': 'CTD data, processed with https://github.com/jeanlucshaw/mxtoolbox',
                 'description': '',
                 'history': '',
                 'platform': '',
                 'bin_size': '',
                 'looking': '',
                 'instrument_serial': '',
                 'contact': 'Jean-Luc.Shaw@dfo-mpo.gc.ca'})

    # Add metadata
    for v_, u_, sb_ in zip(profile_vars, profile_units, profile_sb_names):
        ds[v_].attrs['units'] = u_
        ds[v_].attrs['long_name'] = v_

        if sb_ in seabird_std_names:
            ds[v_].attrs['standard_name'] = seabird_std_names[sb_]
        else:
            print('No CF name for variable: %s, %s' % (v_, sb_))

    return ds


# Command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=__doc__)

    # identifies files
    parser.add_argument('files',
                        metavar='1 - files',
                        help='Expression identifying adcp files',
                        nargs='+')
    # deployment nickname
    parser.add_argument('name',
                        metavar='2 - name',
                        help='''Mission, mooring, or station name to
                        include to the output file name.''')
    # optional save destination
    parser.add_argument('-o', '--output',
                        metavar='output destination',
                        help='''Path (relative or absolute) to destination of
                        the ouput netCDF file.''')
    args = parser.parse_args()

    # Manage path
    if args.output:
        # Output path is absolute
        if args.output[0] == '/':
            path = args.output

        # Output path is relative
        else:
            path = "%s/%s" % (os.getcwd(), args.output)

    # output path is not provided
    else:
        path = os.getcwd()

    if path[-1] != '/':
        path += '/'

    # Initialize time and geo coordinates
    time = np.empty(len(args.files), dtype='datetime64[ns]')
    lon = np.empty(len(args.files))
    lat = np.empty(len(args.files))

    # Get time series data
    for i_, f_ in enumerate(args.files):

        # Read metadata
        md = rd.read_cnv_metadata(f_)

        # Manage coordinates and time
        time[i_] = md['date']
        lon[i_] = md['lon']
        lat[i_] = md['lat']

    # Differentiate two columns named identically
    md['names'] = rd.mangle_list_duplicates(md['names'])

    # Initialize xarray
    ds = profile_time_series_init(np.arange(max_depth), time, profile_metadata=md)

    # Get profile data
    for i_, f_ in enumerate(args.files):

        # Read profile
        df = rd.pd_read_cnv(f_)

        # Add profile to dataset
        crd = dict(z=df.Pressure.values, time=df.date[0].to_datetime64())
        for v_ in ds.data_vars:
            if (len(ds[v_].dims) == 2) and (v_ in df.keys()):
                ds[v_].loc[crd] = df[v_]

    # Add data min and data max
    for v_ in [*ds.data_vars, *ds.coords]:
        ds[v_].attrs['data_max'] = ds[v_].max().values
        ds[v_].attrs['data_min'] = ds[v_].min().values
    ds['time'].attrs['data_min'] = str(ds['time'].min().values)[:19]
    ds['time'].attrs['data_max'] = str(ds['time'].max().values)[:19]

    # Coordinate attributes
    ds['time'].attrs['axis'] = 'T'
    ds['lon'].attrs['standard_name'] = 'longitude'
    ds['lon'].attrs['long_name'] = 'Longitude'
    ds['lon'].attrs['units'] = 'degrees_east'
    ds['lon'].attrs['axis'] = 'X'
    ds['lat'].attrs['standard_name'] = 'latitude'
    ds['lat'].attrs['long_name'] = 'Latitude'
    ds['lat'].attrs['units'] = 'degrees_north'
    ds['lat'].attrs['axis'] = 'Y'
    ds['z'].attrs['standard_name'] = 'sea_water_pressure'
    ds['z'].attrs['long_name'] = 'Pressure'
    ds['z'].attrs['positive'] = 'down'
    ds['z'].attrs['units'] = 'db'
    ds['z'].attrs['axis'] = 'Z'

    # Set creation date attribute
    ds.attrs['history'] = 'Created: %s' % datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # Remove depths with no data
    ds = ds.dropna(dim='z', how='all')

    # Save to netCDF
    ds.to_netcdf('%sviking_profiles_%s.nc' % (path, args.name))
