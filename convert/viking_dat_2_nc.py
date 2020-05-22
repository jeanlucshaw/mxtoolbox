"""
Make Viking GPS netcdf files used by adcp2nc.

When Viking buoys have no bottom track data adcp2nc can
use files produced by this routine to correct for profiler
movement.

The options below are generated automatically by the python
part of this routine but the first postional argument is
passed to the shell part of this routine. It is the path
to the raw files containing GPS coordinates. Typically,

.. code::

   /data/Viking/DropBox/Dropbox/BUOY_DATA/BUOYNAME/YYYY

"""
import argparse as ap
import pandas as pd
import matplotlib.pyplot as plt
import mxtoolbox.process as ps

# Parse input arguments
parser  =   ap.ArgumentParser(usage=__doc__)
parser.add_argument("filename")
parser.add_argument('-e', '--west-bound',
                    metavar='',
                    type=float,
                    help='Exclude data west of this longitude.')
parser.add_argument('-E', '--east-bound',
                    metavar='',
                    type=float,
                    help='Exclude data east of this longitude.')
parser.add_argument('-n', '--south-bound',
                    metavar='',
                    type=float,
                    help='Exclude data south of this latitude.')
parser.add_argument('-N', '--north-bound',
                    metavar='',
                    type=float,
                    help='Exclude data north of this latitude.')
parser.add_argument('-t', '--track',
                    action='store_true',
                    help='Show plot of buoy tracks.')
parser.add_argument('-p', '--printout',
                    action='store_true',
                    help='Show plot of buoy tracks.')
parser.add_argument("station")
args    =   parser.parse_args()

# Functions
def pprint(variable, units):
    maximum = dataframe[variable].max()
    minimum = dataframe[variable].min()
    print('%15s : range (%.2f, %.2f) %s' % (variable,
                                            minimum,
                                            maximum,
                                            units))
# Read data file
dataframe = pd.read_csv(args.filename,
                        sep=r"\s+",
                        parse_dates=['time'],
                        names=['time',
                               'lat',
                               'lon',
                               'heading'])
# Filter by lat/lon
if args.west_bound:
    dataframe = dataframe.loc[dataframe.lon > args.west_bound]
if args.east_bound:
    dataframe = dataframe.loc[dataframe.lon < args.east_bound]
if args.south_bound:
    dataframe = dataframe.loc[dataframe.lat > args.south_bound]
if args.north_bound:
    dataframe = dataframe.loc[dataframe.lat < args.north_bound]

# Calculate velocity
u, v, speed = ps.lonlat2speed(dataframe.lon.values,
                              dataframe.lat.values,
                              dataframe.time.values,
                              heading=dataframe.heading.values,
                              top_speed=5)

# Calculate distance from center
lon_0 = (dataframe.lon.max() + dataframe.lon.min()) / 2
lat_0 = (dataframe.lat.max() + dataframe.lat.min()) / 2
radial_distance = ps.lonlat2distancefrom(dataframe.lon.values,
                                         dataframe.lat.values,
                                         lon_0,
                                         lat_0)

# Add new data to dataframe
dataframe['u'] = u
dataframe['v'] = v
dataframe['speed'] = speed
dataframe['radial_distance'] = radial_distance

# Show buoy track and center point
if args.track:
    plt.plot(dataframe.lon, dataframe.lat)
    plt.plot(lon_0, lat_0, 'o', mfc='r', mec='r', ms=10)
    plt.show()

# Print out
if args.printout:
    print('=' * 80)
    pprint('u', 'm/s')
    pprint('v', 'm/s')
    pprint('speed', 'm/s')
    pprint('radial_distance', 'm')
    pprint('lon', 'degrees east')
    pprint('lat', 'degrees north')

# Sort by time and eliminate possible duplicates
dataframe.sort_values('time', inplace=True)
dataframe.drop_duplicates(subset='time', keep='first', inplace=True)

# Export to netcdf
strt = str(dataframe.time.values[0])[:10]
stop = str(dataframe.time.values[-1])[:10]
dataframe.set_index('time').to_xarray().to_netcdf(args.station+'_'+strt+'_'+stop+'_GPS.nc')
