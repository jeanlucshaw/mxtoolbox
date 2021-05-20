"""
Convert SBE files from moorings (.asc, .RAW) to netcdf.
"""
import pandas as pd
import gsw
import argparse
import re
import os


__all__ = ['_parser_sberaw_header', '_read_sberaw']


def _parser_sberaw_header(fname, manual_column_order=''):
    """
    Get included variables and header length.

    Parameters
    ----------
    fname: str
       Path and name of SBE file.
    manual_column_order: str
       User specified column order {'t','s','c','p','h'}. The character
       `h` specifies placed of the date and hour columns.

    Returns
    -------
    list:
        Characters representing the order of variable columns.
    int:
        Number of lines to skip in header when reading data.

    """
    # Parse header
    LINES = open(fname, 'r', errors='replace').readlines()

    # Header loop switches
    in_var_block = False
    salinity = False
    column_order = []
    header_lines = 0

    # Loop over header lines
    for L in LINES:

        # Is salinity output
        if re.search(r'\* output salinity with each sample', L):
            salinity = True

        # Start looking column names
        if re.search(r'\* S>', L):
            in_var_block = True

        # Note column order
        if in_var_block:
            if re.search('temperature', L):
                column_order.append('t')
            if re.search('salinity', L):
                column_order.append('s')
            if re.search('conductivity', L):
                column_order.append('c')
            if re.search('pressure', L):
                column_order.append('p')
            if re.search('rtc', L):
                column_order.append('h')

        # Exit if the header is read
        if re.search(r'\*END\*', L):
            break
        else:
            header_lines += 1

    # Manage user specified column order
    if manual_column_order:
        column_order = list(manual_column_order)
    else:
        # Add salinity to column order in output
        if salinity:
            column_order.insert(-1, 's')

    # Expected column format
    col_fmt = ('.*,' * len(column_order))[:-1]

    # Exit when the column format is found
    for L in LINES[header_lines:]:
        if re.search(col_fmt, L):
            break
        else:
            header_lines += 1

    return column_order, header_lines


def _read_sberaw(fname,
                 column_order,
                 header_lines,
                 min_depth=0,
                 min_sal=15,
                 max_temp=20,
                 max_date=None,
                 pressure=None,
                 latitude=49):
    """
    Read SBE file to dataframe.

    Parameters
    ----------
    fname: str
       Path and name to SBE file.
    column_order: list
       Characters showing column order {'t', 'c', 'p', 's', 'h'}.
    header_lines: int
       Number of lines in the SBE file header.
    min_depth: float
       Exclude data shallower than this value (m).
    min_sal: float
       Exclude data less saline than this value (PSU).
    max_temp: float
       Exclude data warmer than this value (dC).
    pressure: float
       Manually specify this as the constant pressure of the data.
    latitude: float
       For calculating depth from pressure.

    """
    # Names of output columns
    ext_translator = {'t': 'temp',
                      's': 'sal',
                      'c': 'cond',
                      'p': 'pres',
                      'h': ['day_month_year', 'hour']}

    # Build column name list from file extension
    cols, date_cols = [], []
    for i, c in enumerate(column_order):
        if c == 'h':
            for k, cc in enumerate(ext_translator[c]):
                cols.append(cc)
                date_cols.append(i + k)
        else:
            cols.append(ext_translator[c])

    # Read to dataframe
    df = pd.read_csv(fname,
                     names=cols,
                     skiprows=header_lines,
                     infer_datetime_format=True,
                     parse_dates={'time': date_cols})

    # Add pressure if not present and manually specified
    if pressure is not None and 'p' not in column_order:
        df.loc[:, 'pres'] = pressure

    # Add salinity if not present
    if 'sal' not in df.keys():
        # Check required variables are present
        if [v in df.keys() for v in ['cond', 'temp', 'pres']] == [True, True, True]:
            # Conversion of conductivity units from S/m to mS/cm
            conductivity = 10 * df.cond

            # Add salinity column
            df.loc[:, 'sal'] = gsw.SP_from_C(conductivity, df.temp, df.pres)

        # Missing variables to output salinity
        else:
            print('No salinity information in file')

    # Add depth
    if 'pres' in df.keys():
        df.loc[:, 'z'] = -1 * gsw.z_from_p(df.pres, latitude)

    # Quality control
    if 'sal' in df.keys():
        df.sal.where(df.sal > min_sal, inplace=True)
    if 'temp' in df.keys():
        df.temp.where(df.temp < max_temp, inplace=True)
    if 'z' in df.keys():
        df = df.query('z > %f' % min_depth)

    # Eliminate timestamp duplicates
    df = df.drop_duplicates(subset='time')

    # Sort by time
    df = df.sort_values('time')

    # Remove after max date if required
    if args.max_date:
        df = df.query('time < "%s"' % args.max_date)

    return df.set_index('time')

# Command line interface
if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("fname")
    parser.add_argument("sname")
    parser.add_argument('-d', '--min-depth',
                        metavar='', type=float, default=0,
                        help='discard data shallower than this depth in meters')
    parser.add_argument('-e', '--col-order',
                        metavar='', type=str, default='',
                        help='discard data shallower than this depth in meters')
    parser.add_argument('-t', '--max-temp',
                        metavar='', type=float, default=20,
                        help='discard data hotter than this (dC)')
    parser.add_argument('-s', '--min-sal',
                        metavar='', type=float, default=0,
                        help='discard data less salty than this (PSU)')
    parser.add_argument('-p', '--pressure',
                        metavar='', type=float, default=None,
                        help='pressure to use for salinity calculation (dbar)')
    parser.add_argument('-D', '--max-date',
                        metavar='', type=str, default='',
                        help='discard data after this date')
    args = parser.parse_args()

    # Get variable names and header length
    column_order, header_lines = _parser_sberaw_header(args.fname,
                                                       manual_column_order=args.col_order)

    # Read data
    df = _read_sberaw(args.fname,
                      column_order,
                      header_lines,
                      min_depth=args.min_depth,
                      min_sal=args.min_sal,
                      max_temp=args.max_temp,
                      pressure=args.pressure,
                      max_date=args.max_date)

    # Save location
    abspath = os.path.abspath(args.fname)
    abspath = os.path.dirname(abspath)

    # Save to netcdf
    strt = str(df.index.values[0])[:10]
    stop = str(df.index.values[-1])[:10]
    savename = '%s_%s_%s_SBE37.nc' % (args.sname, strt, stop)
    df.to_xarray().to_netcdf("%s/%s" % (abspath, savename))
