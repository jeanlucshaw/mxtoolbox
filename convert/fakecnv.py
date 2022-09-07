"""
A command line utility designed to recreate minimal seabird-like
cnv files from AZOMP netcdf files of CTD casts. As this will likely
be used mostly for Gulf of St. Lawrence and Newfoundland data,
longitudes are flipped towards the west by default.

By default, the expected coordinate names are time, level, longitude
and latitude. If they are named differently, this will cause an error
which can be fixed by first calling nc2fakecnv with the -s flag to
obtain coordinate names:

.. code::

   $ nc2fakecnv file_name.nc -s

then calling it again and specifying the coordinate names:

.. code::

   $ nc2fakecnv file_name.nc -l ... -L ... -t ... -z ...

By default, the following variable names are translated to better
match seabird formatting:

.. code::

   temperature -> tv290C: Temperature [ITS 90, deg C]
   salinity -> sal00: Salinity, Practical [PSU]
   conductivity -> c0S/m: Conductivity [S/m]
   sigma-t -> sigma-t00: Density, sigma-t [kg/m^3]
   pressure -> prdM: Pressure, Strain Gauge [db]

As this translation makes use of information not found in the netcdf
files it is hard-coded. The user must make sure the translated units
match the originals. To instead infer names and units, call with the -i
flag

.. code::

   $ nc2fakecnv file_name -i

"""
import argparse as ap
import xarray as xr
import pandas as pd
import numpy as np
import os

__all__ = ['xr2fakecnv']

# Seabird parameter name translation
SB_DESC = {'temperature': 'tv290C: Temperature',
           'salinity': 'sal00: Salinity, Practical',
           'pressure': 'prdM: Pressure, Strain Gauge',
           'conductivity': 'c0S/m: Conductivity',
           'sigma-t': 'sigma-t00: Density, sigma-t'}
SB_UNIT = {'temperature': 'ITS 90, deg C',
           'salinity': 'PSU',
           'pressure': 'db',
           'conductivity': 'S/m',
           'sigma-t': 'kg/m^3'}
MSGS = []
MSGS.append('Either no seabird name or unit available for variable: %s')
MSGS.append('Either no seabird name or unit available for depth coordinate: %s')
MSGS.append('Keeping inferred descriptor: %s [%s]')

# Conversion function
def xr2fakecnv(dataset,
               label=None,
               time_name=None,
               depth_name=None,
               lon_name=None,
               lat_name=None,
               dataset_info=False,
               flip_longitude=True,
               infer_descriptors=True,
               verbose=False,
               sort_by=None,
               require=None):
    """
    Write xarray dataset to fake cnv text file.

    The input xarray is expected to have time, level, longitude,
    and latitude as coordinates. Data variables are expected to
    have the strings `units` and `standard_name` as attributes. This
    routine is meant for 1D measurements along time, depth or both.

    Parameters
    ----------
    dataset: xarray.Dataset
        Datasets to write to file.
    label: str
        Prefix of date in output file name.
    time_name: str
        Name of time coordinate in all Datasets.
    depth_name: str
        Name of depth coordinate in all Datasets.
    lon_name: str
        Name of longitude coordinate in all Datasets.
    lat_name: str
        Name of latitude coordinate in all Datasets.
    flip_longitude: bool
        Switch longitude from east to west or vice versa.
    infer_descriptors: bool
        Infer variable descriptors from input file.
    verbose: bool
        Print to std when descriptors must be inferred.
    sort_by: str
        Sort data of the cnv file by this column.
    require: str
        Omit row if data are missing from this column.

    """
    # # Show variable and coordinate names
    # if dataset_info:
    #     try:
    #         DS = files[0]
    #     except TypeError():
    #         DS = xr.open_dataset(files)
    #     print(DS)

    # # Process the input file(s)
    # else:
    #     for FILE in files:
    # Default parameters
    TIME_NAME = time_name if time_name is not None else 'time'
    DEPTH_NAME = depth_name if depth_name is not None else 'level'
    LON_NAME = lon_name if lon_name is not None else 'longitude'
    LAT_NAME = lat_name if lat_name is not None else 'latitude'

    # Determine master coordinate
    if TIME_NAME in dataset[TIME_NAME].dims:
        MASTER_COORD = TIME_NAME
    elif DEPTH_NAME in dataset[DEPTH_NAME].dims:
        MASTER_COORD = DEPTH_NAME

    # ====================
    # Set output file name
    # ====================

    # If time is provided for each data
    if dataset[TIME_NAME].values.size > 1:
        mtime = dataset[TIME_NAME].mean().values
        DATE = str(mtime)[:10]
        TIME = str(mtime)[11:16]

    # If time is a single value
    else:
        DATE = str(dataset[TIME_NAME].values[0])[:10]
        TIME = str(dataset[TIME_NAME].values[0])[11:16]
    
    # Set output file name and increment if filename exists
    FILE_INC = 0
    if label:
        CNVNAME = "%s_%s_%03d.cnv" % (label, DATE, FILE_INC)
    else:
        CNVNAME = "%s_%03d.cnv" % (DATE, FILE_INC)
    while os.path.isfile(CNVNAME):
        FILE_INC += 1
        if label:
            CNVNAME = "%s_%s_%03d.cnv" % (label, DATE, FILE_INC)
        else:
            CNVNAME = "%s_%03d.cnv" % (DATE, FILE_INC)

    # Init output dataframe
    DF = pd.DataFrame()

    # Open output file
    with open(CNVNAME, 'w') as CNVFILE:

        # Manage header time info
        # TIME = str(dataset[TIME_NAME].values)[11:16]
        CNVFILE.write("* NL CTD file\n")
        CNVFILE.write("** Date:        %s %s UTC\n" % (DATE, TIME))

        # ===========================
        # Manage header position info
        # ===========================

        # Convert to dms
        longitude = np.unique(dataset[LON_NAME].values).squeeze()
        latitude = np.unique(dataset[LAT_NAME].values).squeeze()
        LOND = np.int64(longitude)
        LONM = longitude - LOND
        LONM *= 60                      # convert to decimal minutes
        LATD = np.int64(latitude)
        LATM = latitude - LATD
        LATM *= 60                      # convert to decimal minutes

        # Convert lat/lon info to usual format
        if dataset[LAT_NAME].units in ['degree_north', 'degrees_north']:
            LAT_DIR = 'N'
        else:
            LAT_DIR = 'S'
        if dataset[LON_NAME].units in ['degree_east', 'degrees_east']:
            if flip_longitude:
                LON_DIR = 'W'
                LOND *= -1
            else:
                LON_DIR = 'E'
        else:
            if flip_longitude:
                LON_DIR = 'E'
                LOND *= -1
            else:
                LON_DIR = 'W'

        # Write single value or mean of the array
        if isinstance(LATD, (int, np.int64)):
            CNVFILE.write("** Latitude:    %d %.3f %s\n" % (LATD, LATM, LAT_DIR))
            CNVFILE.write("** Longitude:   %03d %.3f %s\n" % (LOND, abs(LONM), LON_DIR))
        else:
            hlatd, hlatm = np.mean(LATD), np.mean(LATM)
            hlond, hlonm = np.mean(LOND), np.mean(LONM)
            CNVFILE.write("** Latitude:    %d %.3f %s\n" % (hlatd, hlatm, LAT_DIR))
            CNVFILE.write("** Longitude:   %03d %.3f %s\n" % (hlond, abs(hlonm), LON_DIR))

        # ===================
        # Manage z coordinate
        # ===================
        ZNAME = dataset[DEPTH_NAME].standard_name
        ZUNIT = dataset[DEPTH_NAME].units
        if infer_descriptors:
            try:
                ZNAME, ZUNIT = SB_DESC[ZNAME], SB_UNIT[ZNAME]
            except KeyError:
                if verbose:
                    print(MSGS[1] % ZNAME)
                    print(MSGS[2] % (ZNAME, ZUNIT))

        # Write depth as first column
        DF[ZNAME] = dataset[DEPTH_NAME].values
        CNVFILE.write("# name %d = %s [%s]\n" % (0, ZNAME, ZUNIT))

        # Remove for data variables if in data variables
        if DEPTH_NAME in dataset.data_vars:
            dataset = dataset.drop(DEPTH_NAME)

        # =====================
        # Write variable header
        # =====================
        i = 1                   # column number

        # for NAME in dataset.keys():
        for NAME in dataset.data_vars:
            # Check this variable is along the depth coordinate
            # if DEPTH_NAME in dataset[NAME].dims:
            if MASTER_COORD in dataset[NAME].dims:

                # Print non data variable
                if dataset[NAME].dtype == 'O':
                    FLD_NAME = NAME
                    if infer_descriptors:
                        try:
                            FLD_NAME = SB_DESC[NAME]
                        except KeyError:
                            if verbose:
                                print(MSGS[0] % FLD_NAME)

                    CNVFILE.write("# name %d = %s \n" % (i, FLD_NAME))
                    DF[NAME] = dataset[NAME].values
                    i += 1

                # Print data variable if not empty
                elif np.isfinite(dataset[NAME]).values.any():
                    FLD_NAME = NAME
                    if 'units' in dataset[NAME].attrs:
                        FLD_UNIT = dataset[NAME].units
                    else:
                        FLD_UNIT = 'N/A'
                    if infer_descriptors:
                        try:
                            FLD_NAME = SB_DESC[NAME]
                            FLD_UNIT = SB_UNIT[NAME]
                        except KeyError:
                            if verbose:
                                print(MSGS[0] % FLD_NAME)
                                print(MSGS[2] % (FLD_NAME, FLD_UNIT))

                    CNVFILE.write("# name %d = %s [%s]\n" % (i, FLD_NAME, FLD_UNIT))
                    DF[NAME] = dataset[NAME].values
                    i += 1
        CNVFILE.write("*END*\n")

        # Filter if requested
        if require is not None:
            DF = DF.query('~%s.isnull()' % require, engine='python')

        # Sort if requested
        if sort_by is not None:
            DF = DF.sort_values(by=sort_by)

        # Append data
        DF.to_csv(CNVFILE,
                  mode='a',
                  sep=' ',
                  header=False,
                  index=False,
                  float_format='%.5f',
                  na_rep='nan')

if __name__ == '__main__':
    # Manage input flags
    PARSER = ap.ArgumentParser(prog='nc2fakecnv',
                               formatter_class=ap.RawDescriptionHelpFormatter,
                               usage='nc2fakecnv.py ctdfile(s)',
                               description=__doc__)
    PARSER.add_argument('files',
                        metavar='',
                        help='expression identifying ctd netcdf files (e.g. *.nc)',
                        nargs='+')
    PARSER.add_argument('-t', '--time-name',
                        metavar='',
                        help='name of time coordinate')
    PARSER.add_argument('-z', '--depth-name',
                        metavar='',
                        help='name of time coordinate')
    PARSER.add_argument('-l', '--lon-name',
                        metavar='',
                        help='name of longitude coordinate')
    PARSER.add_argument('-L', '--lat-name',
                        metavar='',
                        help='name of latitude coordinate')
    PARSER.add_argument('-s', '--dataset-info',
                        help='show netcdf variable and coordinate names',
                        action='store_true')
    PARSER.add_argument('-f', '--flip-longitude',
                        help='switches longitude from east to west or vice versa',
                        action='store_false')
    PARSER.add_argument('-i', '--infer-descriptors',
                        help='infer variable descriptors from input file',
                        action='store_false')
    PARSER.add_argument('-v', '--verbose',
                        help='print to std when descriptors must be inferred',
                        action='store_true')
    ARGS = PARSER.parse_args()

    # Function call
    for file_ in ARGS.files:
        dataset = xr.open_dataset(file_)
        xr2fakecnv(dataset,
                   ARGS.time_name,
                   ARGS.depth_name,
                   ARGS.lon_name,
                   ARGS.lat_nama,
                   ARGS.dataset_info,
                   ARGS.flip_longitude,
                   ARGS.infer_descriptors,
                   ARGS.verbose)
