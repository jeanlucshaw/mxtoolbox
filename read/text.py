"""
Read text files of known formats into python variables
"""
import dateparser
import os
import re
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ..process.convert import dmd2dd

__all__ = ['list2cm',
           'mangle_list_duplicates',
           'pd_cat_column_files',
           'pd_read_cnv',
           'pd_read_odf',
           'read_cnv_metadata',
           'read_odf_metadata',
           'xr_print_columns']


def list2cm(array_file, N: 'integer' = 256):
    '''
    Returns a N step colormap object generated from the RGB
    triplets in array_file.
    '''
    colors = np.genfromtxt(array_file)
    return LinearSegmentedColormap.from_list("Tsat", colors, N=N)


def mangle_list_duplicates(list_):
    """
    Add suffix to duplicate strings in list strings

    Parameters
    ----------
    list_, : list of str
        List of names to check for duplicates

    Returns
    -------
    list:
        List of names with duplicates suffixed.

    """
    # Initialize output and duplicate counter
    list_new = list_.copy()
    counts = dict()

    # Loop over list elements
    for index, name in enumerate(list_):
        occurences = list_.count(name)

        # Modify if one of duplicate set
        if occurences > 1:
            if name in counts.keys():
                counts[name] += 1
                list_new[index] += '-%d' % counts[name] 
            else:    
                counts[name] = 1
                list_new[index] = '%s-1' % list_[index]

    return list_new


def pd_cat_column_files(path_list,
                        cols,
                        index_cols,
                        sep=r'\s+',
                        axis=1,
                        parse_dates=True,
                        **rc_kw):
    """
    Merge ascii column files to dataframe.

    Takes as input a list of file paths pointing to ascii
    column data with the same columns, but not necessarily the
    the same rows and merges them into one pandas DataFrame.

    If merging on axis 0, make sure the text files have no header!

    Parameters
    ----------
    path_list : list of str
        Paths to files to merge.
    cols : list of str or int
        If a column is defined by an integer list, the column name
        will be extracted from the file name.
    index_cols : list of str
        Column(s) to use as coordinates.

    """
    # Separate path and file name
    fname = os.path.basename(path_list[0])
    fpath = os.path.dirname(path_list[0])

    # Exception for file in current directory
    if fpath != '':
        fpath += '/'

    # Build column name vector
    col_names = []
    for col_name in cols:
        if isinstance(col_name, list):
            col_names.append(fname[col_name[0]:col_name[1]])
        else:
            col_names.append(col_name)

    # Read first file to init dataframe
    dataframe = pd.read_csv('%s%s' % (fpath, fname),
                            sep=sep,
                            names=col_names,
                            index_col=index_cols,
                            skiprows=1,
                            parse_dates=parse_dates,
                            infer_datetime_format=True,
                            usecols=col_names,
                            **rc_kw)

    # Make a new column for every other file
    for path in path_list[1:]:
        # Separate path and file name
        fname = os.path.basename(path)
        fpath = os.path.dirname(path)

        # Exception for file in current directory
        if fpath != '':
            fpath += '/'

        # Build column name vector
        col_names = []
        for col_name in cols:
            if isinstance(col_name, list):
                col_names.append(fname[col_name[0]:col_name[1]])
            else:
                col_names.append(col_name)

        # Read this file
        new_dataframe = pd.read_csv('%s%s' % (fpath, fname),
                                    sep=sep,
                                    names=col_names,
                                    index_col=index_cols,
                                    skiprows=1,
                                    parse_dates=parse_dates,
                                    infer_datetime_format=True,
                                    usecols=col_names,
                                    **rc_kw)

        # Concatenate
        dataframe = pd.concat((dataframe, new_dataframe), axis=axis)

    # Return dataframe
    return dataframe


def pd_read_cnv(FNAME,
                sep=r'\s+',
                usecols=None,
                metadata_cols=True,
                short_names=True,
                **kw_read_csv):
    """
    Read seabird(like) files into a pandas dataframe.

    Parameters
    ----------
    FNAME: str
        Path and name of odf file.
    sep: str
        Data delimiter of odf file. Default is whitespace.
    usecols: list of int
        Index of columns to extract from odf file. Passed to pd.read_csv.
        By default all columns are returned.
    metadata_cols: bool
        Add columns with date, lon, lat repeated from header.
    short_names: bool
        Give output columns shorter names.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the requested data columns of the cnv file.

    """

    # Read the file as a list of lines
    LINES = open(FNAME, 'r', errors='replace').readlines()

    # Get metadata from header
    md = read_cnv_metadata(FNAME, short_names=short_names)

    # Manually mangle duplicate names
    md['names'] = mangle_list_duplicates(md['names'])
    
    # Select columns to read
    if usecols is None:
        usecols = [i_ for i_, _ in enumerate(md['names'])]

    # Read the data
    DF = pd.read_csv(FNAME,
                     skiprows=md['header_lines'],
                     sep=sep,
                     usecols=usecols,
                     names=md['names'],
                     na_values=md['missing_values'],
                     **kw_read_csv)

    # Add metadata columns
    if metadata_cols:
        if md['date']:
            DF.loc[:, 'date'] = md['date']
        if md['lon']:
            DF.loc[:, 'Longitude'] = md['lon']
        if md['lat']:
            DF.loc[:, 'Latitude'] = md['lat']

    return DF


def pd_read_odf(FNAME,
                sep=r'\s+',
                col_name='WMO_CODE',
                usecols=None,
                missing_values=None,
                header_depth=False):
    """
    Read ocean data format (odf) files into pandas dataframe.

    Parameters
    ----------
    FNAME: str
        Path and name of odf file.
    sep: str
        Data delimiter of odf file. Default is whitespace.
    col_name: str
        Header line to use as column name. Typical choices are 'NAME',
        'CODE' or 'WMO_CODE'.
    usecols: list of int
        Index of columns to extract from odf file. Passed to pd.read_csv.
        By default all columns are returned.
    missing_values: str or float
        Expression to evaluate as missing_values. Passed to pd.read_csv.
    header_depth: bool
        Add column with depth read in the odf header. Disabled by default.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the requested data columns of the odf file.
        Column names are `col_name [unit]` read directly from the odf file
        header.

    """
    LINES = open(FNAME, 'r', errors='replace').readlines()

    # Parameters
    DEPTH = None
    SEEK_NAME = False
    SEEK_UNIT = False
    SEEK_DEPTH = False
    NAMES = []
    UNITS = []
    PARSE_DATES = False
    PAR_STRING = re.compile('[a-zA-Z0-9_/]+')

    # Parse header
    for (LN, L) in zip(range(len(LINES)), LINES):
        if re.search('EVENT_HEADER', L):
            SEEK_DEPTH = True
        if SEEK_DEPTH and ('DEPTH' in L):
            DEPTH = float(re.findall('[0-9]+\.[0-9]+', L)[0])
            SEEK_DEPTH = False
        if re.search('PARAMETER_HEADER', L):
            SEEK_NAME, SEEK_UNIT = True, True
        if SEEK_NAME and re.search(col_name, L):
            # NAME = PAR_STRING.findall(L.split('=')[-1])[0]
            NAME = ' '.join(PAR_STRING.findall(L.split('=')[-1]))
            NAMES.append(NAME)
            if 'SYTM' in NAME:
                NAMES[-1] = 'SYTM'
                NAMES.append('HOUR')
                UNITS.append('None')
                PARSE_DATES = [['SYTM', 'HOUR']]
            SEEK_NAME = False
        if SEEK_UNIT and re.search('UNITS', L):
            try:
                UNIT = PAR_STRING.findall(L.split('=')[-1])[0]
            except:
                UNIT = 'None'
            UNITS.append(UNIT)
            SEEK_UNIT = False
        if re.search('-- DATA --', L):
            HEADER_LINES = LN + 1
            break

    # Merge variable names and units
    COLUMNS = ['%s [%s]' % (NAME, UNIT) for NAME, UNIT in zip(NAMES, UNITS)]

    # Manually mangle duplicate names
    for index, name in enumerate(COLUMNS):
        if name in COLUMNS[:index]:
            COLUMNS[index] = name + 'duplicate'
    
    # Select columns to read
    if usecols is None:
        usecols = list(range(len(COLUMNS)))

    # Read the data
    DF = pd.read_csv(FNAME,
                     skiprows=HEADER_LINES,
                     sep=sep,
                     usecols=usecols,
                     names=COLUMNS,
                     parse_dates=PARSE_DATES,
                     na_values=missing_values)

    # Add depth column if specified
    if header_depth:
        DF['header depth'] = DEPTH * np.ones(DF.shape[0])

    return DF


def read_cnv_metadata(FNAME, short_names=True):
    """
    Get information from cnv file header.

    Parameters
    ----------
    FNAME: str
        Name and path of cnv file.
    short_names: bool
        Return short variable names.

    Returns
    -------
    NAMES: list of str
        
    """
    # Read the file as a list of lines
    LINES = open(FNAME, 'r', errors='replace').readlines()

    # Parameters
    metadata = dict(names=[],
                    seabird_names=[],
                    units=[],
                    date=None,
                    lon=None,
                    lat=None,
                    missing_values=None,
                    header_lines=0)

    # Parse header
    for LN, L in enumerate(LINES):

        # Scan for metadata
        if re.search('\* Date:', L):
            day_ = re.findall('\d{4}-\d{2}-\d{2}', L)[0]
            hour = re.findall('\d{2}:\d{2}[:0-9]*', L)[0]
            metadata['date'] = np.datetime64('%sT%s' % (day_, hour))
        if re.search('\* Longitude:', L):
            degree = float(L.split()[-3])
            minute = float(L.split()[-2])
            direction = L.split()[-1]
            metadata['lon'] = dmd2dd(degree, minute, direction)
        if re.search('\* Latitude:', L):
            degree = float(L.split()[-3])
            minute = float(L.split()[-2])
            direction = L.split()[-1]
            metadata['lat'] = dmd2dd(degree, minute, direction)
        if re.search('# bad_flag', L):
            metadata['missing_values'] = L.split()[-1]

        # Scan for variables
        if re.search('# name', L):
            full_name = L.split('=')[-1].strip()
            short_name = full_name.split()[1]
            short_name = re.findall('[A-Za-z]+', short_name)[0]
            if short_names:
                name = short_name
            else:
                name = full_name
            metadata['names'].append(name)
            metadata['seabird_names'].append(full_name.split(':')[0])

            if re.findall('\[.*\]', L):
                units = re.findall('\[.*\]', L)[0].strip('[]')
                metadata['units'].append(units)
            else:
                metadata['units'].append('')

        # Scan for end of header
        if re.search('\*END\*', L):
            metadata['header_lines'] = LN + 1
            break

    return metadata


def read_odf_metadata(fname, vnames, dtype=None):
    """
    Get info from odf file header.

    Common examples of `vnames` for the trawl CTD dataset are the
    following:

    'INITIAL_LATITUDE'
    'INITIAL_LONGITUDE'
    'START_DATE_TIME'
    'END_DATE_TIME'
    'SAMPLING_INTERVAL'

    Parameters
    ----------
    fname: str
        Name of odf file to read.
    vnames: list of str
        Names of variables to extract. Text left of the `=` sign
        with all space characters removed.
    dtype: list of Class or None
        If specified must be an iterable of type class constructors
        the same length as `vnames`. If unspecified, all variables
        are returned as strings.

    Returns
    -------
    list
        Ordered list of queried metadata information.

    """
    
    # Open file
    LINES = open(fname, 'r', errors='replace').read().splitlines()

    # Init data structure
    results =  ['' for _ in range(len(vnames))]

    # Loop over header lines
    for L in LINES:
        # Parse line
        PARAMETER = L.replace(" ", "").split('=')[0]
        VALUE = L.replace(",", "").split('=')[-1]

        # Keep if requested
        if PARAMETER in vnames:
            position = vnames.index(PARAMETER)
            results[position] = VALUE

        # Break if data section reached
        if re.search('-- DATA --', L):
            break

    # Convert if requested
    for i, (r, t) in enumerate(zip(results, dtype)):
        # Parse datestrings before datetime conversions
        if t == np.datetime64:
            r = dateparser.parse(r)

        # Type conversion
        results[i] = t(r)
    
    return results


def xr_print_columns(xr_obj, cols, filename, csv_kw={}):
    """
    Print variables and coordinates of an xarray dataset or
    dataarray to a simple ascii file.

    Parameters
    ----------
    xr_obj: xarray.Dataset or xarray.DataArray
        From which to write.
    filename: str
        Path and name of output file
    csv_kw: dict
        Dictionary of arguments passed to pd.to_csv
    cols: iterable of strings
        Names of variables or coordinates to print. Order defines the 
        output file column order.

    Note
    ----
    Possible values of cols are all variables and coordinates
    of `xr_obj`, as well as year, month, day and date. If `xr_obj` is
    a dataarray enter `values` to print the data values.
    """
    # Build a dictionary of the proper columns
    col_dict = {}
    for col in cols:
        if col == 'year':
            col_dict[col] = xr_obj.time.dt.year.values
        elif col == 'month':
            col_dict[col] = xr_obj.time.dt.month.values
        elif col == 'day':
            col_dict[col] = xr_obj.time.dt.day.values
        elif col == 'date':
            col_dict[col] = xr_obj.time
        elif col == 'values':
            col_dict[col] = xr_obj.values
        else:
            col_dict[col] = xr_obj[col]
        
    # Init dataframe
    dataframe = pd.DataFrame.from_dict(col_dict)

    # Handle pd.to_csv defaults and keyword arguments
    csv_defaults = {'sep': ' ',
                    'float_format': '%f',
                    'header': False,
                    'index': False,
                    'na_rep': np.nan}
    for key in csv_defaults:
        if not key in csv_kw.keys():
            csv_kw[key] = csv_defaults[key]

    # Print to csv
    dataframe.to_csv(filename, **csv_kw)
