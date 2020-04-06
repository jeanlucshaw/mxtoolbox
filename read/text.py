"""
Read text files of known formats into python variables
"""
import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


__all__ = ['list2cm',
           'pd_cat_column_files',
           'pd_read_odf']


def list2cm(array_file, N: 'integer' = 256):
    '''
    Returns a N step colormap object generated from the RGB
    triplets in array_file.
    '''
    colors = np.genfromtxt(array_file)
    return LinearSegmentedColormap.from_list("Tsat", colors, N=N)


def pd_cat_column_files(path_list,
                     cols,
                     index_cols,
                     sep=r'\s+',
                     axis=1,
                     parse_dates=True):
    """
    Takes as input a list of file paths pointing to ascii
    column data with the same columns, but not necessarily the
    the same rows and merges them into one pandas DataFrame.

    If merging on axis 0, make sure the text files have no header!

    path_list:    list of strings, paths to files to merge
    cols:         list of strings or integer lists. If a column is
                  defined by an integer list, the column name will
                  be extracted from the file name.
    index_cols:   list of strings, column(s) to use as coordinates
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
                            parse_dates=parse_dates,
                            infer_datetime_format=True,
                            usecols=col_names)

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
                                    parse_dates=parse_dates,
                                    infer_datetime_format=True,
                                    usecols=col_names)

        # Concatenate
        dataframe = pd.concat((dataframe, new_dataframe), axis=axis)

    # Return dataframe
    return dataframe


def pd_read_odf(FNAME,
           sep=r'\s+',
           col_name='WMO_CODE',
           missing_values=None):
    """
    Read ocean data format (odf) files and return a pandas
    dataframe. Column names are determined by reading the header. By
    default, the WMO_CODE field of the parameter headings is used to
    choose column names but this can be changed by setting the col_name
    option.
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
            NAME = PAR_STRING.findall(L.split('=')[-1])[0]
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

    # Read the data
    DF = pd.read_csv(FNAME,
                     skiprows=HEADER_LINES,
                     sep=sep,
                     names=NAMES,
                     parse_dates=PARSE_DATES,
                     na_values=missing_values)

    # Add depth column if specified
    if not DEPTH is None:
        DF['z'] = DEPTH * np.ones(DF.shape[0])

    return DF
