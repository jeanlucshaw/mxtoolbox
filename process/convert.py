"""
Simple data conversions. Includes unit conversions and thermodynamic property
conversions.
"""
import gsw
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from scipy.interpolate import interp1d
from datetime import datetime, timezone, timedelta
from .math_ import broadcastable, rotate_frame

__all__ = ['anomaly2rgb',
           'binc2edge',
           'bine2center',
           'dd2dms',
           'dms2dd',
           'hd2uv',
           'lonlat2distances',
           'lonlat2distancefrom',
           'lonlat2heading',
           'lonlat2speed',
           'pd_add_seasons',
           'tetha2hd',
           'uv2hd',
           'xr_SA_CT_pden',
           'dayofyear2dt']


def anomaly2rgb(value):
    """
    Map standardized anomaly to discretized red to blue color scheme.

    Returns an RGB triplet according to standardized anomaly value
    passed as argument. Colors are defined in intervals of 0.5 std and
    saturate at -2 and 2.

    Parameters
    ----------
    value : float
        Standardized anomaly value.

    Returns
    -------
    list
        RGB triplet.

    """
    # Select color
    if -0.5 < value < 0.5:
        col = [1.0, 1.0, 1.0]
    elif -0.5 > value > -1:
        col = [0.875, 0.875, 1.000]
    elif -1.0 > value > -1.5:
        col = [0.700, 0.700, 1.000]
    elif -1.5 > value > -2.0:
        col = [0.360, 0.360, 0.900]
    elif -2.0 > value:
        col = [0.000, 0.000, 0.900]
    elif 0.5 < value < 1:
        col = [1.000, 0.875, 0.875]
    elif 1.0 < value < 1.5:
        col = [1.000, 0.700, 0.700]
    elif 1.5 < value < 2.0:
        col = [1.000, 0.360, 0.360]
    elif 2.0 < value:
        col = [0.900, 0.000, 0.000]
    else:
        print("No mapping for anomaly value %.2f" % value)
        col = 'k'
    return col


def binc2edge(z):
    """
    Get bin edges from bin centers.

    Bin centers can be irregularly spaced. Edges are halfway between
    one point and the next.

    Parameters
    ----------
    z : numpy.array, pandas.DatetimeIndex, pandas.Series
        Bin centers.

    Returns
    -------
    numpy.array, pandas.DatetimeIndex, pandas.Series
        Bin edges.

    See Also
    --------

       * convert.bine2center

    """
    if type(z) is pd.core.indexes.datetimes.DatetimeIndex:
        TIME = pd.Series(z)
        DT = TIME.diff()[1:].reset_index(drop=True)

        # Extend time vector
        TIME = TIME.append(TIME.take([-1])).reset_index(drop=True)

        # Make offset pandas series
        OS = pd.concat((-0.5 * pd.Series(DT.take([0])),
                        -0.5 * DT,
                        0.5 * pd.Series(DT.take([-1])))).reset_index(drop=True)

        # Make bin edge vector
        EDGES = TIME + OS
    elif type(z) is pd.core.series.Series:
        DT = z.diff()[1:].reset_index(drop=True)

        # Extend time vector
        z = z.append(z.take([-1])).reset_index(drop=True)

        # Make offset pandas series
        OS = pd.concat((-0.5 * pd.Series(DT.take([0])),
                        -0.5 * DT,
                        0.5 * pd.Series(DT.take([-1])))).reset_index(drop=True)

        # Make bin edge vector
        EDGES = z + OS
    else:
        dz = np.diff(z)
        EDGES = np.r_[z[0]-dz[0]/2, z[1:]-dz/2, z[-1] + dz[-1]/2]

    return EDGES


def bine2center(bine):
    """
    Get bin centers from bin edges.

    Bin centers can be irregularly spaced. Edges are halfway between
    one point and the next.

    Parameters
    ----------
    bine : 1D array
        Bin edges.

    Returns
    -------
    1D array
        Bin centers.

    See Also
    --------

       * convert.binc2edge
    """
    return bine[:-1] + np.diff(bine) / 2


def dms2dd(degrees, minutes, seconds):
    """
    Convert coordinates from deg., min., sec., to decimal degress.

    Parameters
    ----------
    degrees : float or 1D array
        Degree of coordinate, positive east.
    minutes : float or 1D array
        Minutes of coordinate, positive definite.
    seconds : float or 1D array
        Seconds of coordinate, positive definite.

    Returns
    -------
    float or 1D array
        Coordinates in decimal degrees.

    See Also
    --------

       * convert.dd2dms

    """
    sign = np.sign(degrees)
    return degrees + sign * minutes / 60 + sign * seconds / 3600


def dd2dms(degrees_decimal):
    """
    Convert coordinates from decimal degrees to deg., min., sec.

    Parameters
    ----------
    degrees_decimal : float or 1D array
        Coordinates in decimal degrees, positive east.

    Returns
    -------
    degrees : float or 1D array
        Degree of coordinate, positive east.
    minutes : float or 1D array
        Minutes of coordinate, positive definite.
    seconds : float or 1D array
        Seconds of coordinate, positive definite.

    See Also
    --------

       * convert.dms2dd

    """
    degrees = np.int16(degrees_decimal)
    minutes = np.int16((degrees_decimal - degrees) * 60.)
    seconds = (degrees_decimal - degrees - minutes / 60.) * 3600.
    return (degrees, abs(minutes), abs(seconds))


def hd2uv(heading, magnitude, rotate_by=0):
    """
    Return u, v vector components from heading and magnitude.

    `heading` is assumed to start from North at 0 and rotate clockwise (90 at
    East). The components can be returned in a frame of reference rotated
    clockwise by setting `rotate_by` in degrees.

    Parameters
    ----------
    heading : array_like
        Compass direction (degrees).
    magnitude : array_like
        Vector norm.
    rotate_by : float
        Theta direction of returned `u` component.

    Returns
    -------
    u, v : array_like
        Vector components in chosen frame of reference.

    """
    u = magnitude * np.sin(np.pi * heading /180)
    v = magnitude * np.cos(np.pi * heading / 180)
    u, v = rotate_frame(u, v, -rotate_by, units='deg')

    return u, v


def lonlat2distances(lon, lat, meters_per_unit=1, **kwargs):
    """
    Get distance between points of a GPS track.

    Distances are returned in meters by default. This
    can be adjusted by setting the `meters_per_unit`
    parameter (e.g. 1000 for km).

    Parameters
    ----------
    lon, lat : 1D array
        Plate carree coordinates to process.
    meters_per_unit : float
        Number of meters in the desired output unit of distance.
    kwargs : keyword arguments
        Passed to `pyroj.Geod` .

    Returns
    -------
    distances : 1D array
        Separating the input coordinates.
    """
    kwargs = {'ellps': 'WGS84', **kwargs}
    _geod = Geod(**kwargs)

    distances = np.array([_geod.inv(lon1, lat1, lon2, lat2)
                          for (lon1, lat1, lon2, lat2)
                          in zip(lon[:-1], lat[:-1], lon[1:], lat[1:])])[:, 2]

    distances /= meters_per_unit

    return distances


def lonlat2distancefrom(lon, lat, lon_0, lat_0, meters_per_unit=1, **kwargs):
    """
    Get distance between GPS track and a fixed coordinate.

    Distances are returned in meters by default. This
    can be adjusted by setting the `meters_per_unit`
    parameter (e.g. 1000 for km).

    Parameters
    ----------
    lon, lat : 1D array
        Plate carree coordinates of GPS track.
    lon_0, lat_0 : 1D array
        Plate carree coordinates of fixed coordinate.
    meters_per_unit : float
        Number of meters in the desired output unit of distance.
    kwargs : keyword arguments
        Passed to `pyroj.Geod` .

    Returns
    -------
    distances : 1D array
        Separating the input coordinates.
    """
    kwargs = {'ellps': 'WGS84', **kwargs}
    _geod = Geod(**kwargs)

    distances = np.array([_geod.inv(lon_, lat_, lon_0, lat_0)
                          for (lon_, lat_)
                          in zip(lon, lat)])[:, 2]

    distances /= meters_per_unit

    return distances


def lonlat2heading(lon, lat, **kwargs):
    """
    Get forward azimuth from GPS track.

    Parameters
    ----------
    lon, lat : 1D array
        Plate carree coordinates to process.
    kwargs : keyword arguments
        Passed to `pyroj.Geod` .

    Returns
    -------
    heading : 1D array
        Forward azimuth.
    """
    kwargs = {'ellps': 'WGS84', **kwargs}
    _geod = Geod(**kwargs)

    heading = np.array([_geod.inv(lon1, lat1, lon2, lat2)
                          for (lon1, lat1, lon2, lat2)
                          in zip(lon[:-1], lat[:-1], lon[1:], lat[1:])])[:, 0]

    return heading



def lonlat2speed(lon,
                 lat,
                 time,
                 heading=None,
                 top_speed=None,
                 meters_per_unit=1,
                 seconds_per_unit=1):
    """
    Convert GPS track u, v and speed.

    Speeds are returned in meters per second by default, but
    this can be adjusted using the `meters_per_unit` and
    `seconds_per_unit` optional paramters.

    Parameters
    ----------
    lon, lat : 1D array
        Plate carree coordinates to process.
    time : 1D array of np.datetime64
        Must convertible to `datetime64[s]`.
    heading : 1D array
        Forward azimuth vector. Calculated if not specified.
    top_speed : float
        Filter out speeds greater than this value.
    meters_per_unit : float
        Number of meters in the desired output unit of distance.
    seconds_per_unit : float
        Number of seconds in the desired output unit of time.

    Returns
    -------
    u, v : 1D array
        Eastward and northward velocity components.
    speed : 1D array
        Magnitude of velocity.
    """

    # Convert to datetime
    time = time.astype('datetime64[s]')

    # Compute dx
    distances = lonlat2distances(lon, lat, meters_per_unit=meters_per_unit)

    # Compute dt
    time_delta = np.diff(time)
    dt_seconds = np.float64(time_delta)

    # Compute speed
    speed = distances / dt_seconds * seconds_per_unit

    # Compute centered time vector
    time_centered = time[:-1] + time_delta / 2

    # Interpolate to input time grid
    dataarray = xr.DataArray(speed, dims=['time'], coords={'time': time_centered})
    if top_speed:
        dataarray = dataarray.where(dataarray < top_speed)
    speed = dataarray.interp(time=time, kwargs=dict(fill_value='extrapolate')).values

    # Compute heading
    if heading is None:
        heading = lonlat2heading(lon, lat)

    # Compute velocity components
    u, v = hd2uv(heading, speed)

    return u, v, speed


def pd_add_seasons(dataframe, time='time', stype='astro'):
    """
    Add season column to a dataframe containing a datetime column. The name
    of the datetime column can be specified in parameter time. Astronimical
    or meteorological seasons can be returned by specifiying 'astro' or
    or 'met' at the stype parameter. Astronomical seasons are taken to start
    on the 20th (near the solstice and equinoxes) and meteorological seasons
    are given starting one the 1st of the months containing the solstices and
    equinoxes.
    """
    # Init season column
    dataframe['season'] = np.empty(dataframe[time].values.size, dtype=str)

    # Loop over years
    for year in dataframe[time].dt.year.unique():
        if stype == 'astro':
            strt_w_pv = datetime(year - 1, 12, 20)
            strt_w_nx = datetime(year, 12, 20)
            strt_sprg = datetime(year, 3, 20)
            strt_summ = datetime(year, 6, 20)
            strt_fall = datetime(year, 9, 20)
        elif stype == 'met':
            strt_w_pv = datetime(year - 1, 12, 1)
            strt_w_nx = datetime(year, 12, 1)
            strt_sprg = datetime(year, 3, 1)
            strt_summ = datetime(year, 6, 1)
            strt_fall = datetime(year, 9, 1)
        else:
            raise ValueError('%s is not a valid season type value.' % stype)

        dataframe.loc[(dataframe[time] >= strt_w_pv) &
                      (dataframe[time] < strt_sprg), 'season'] = 'winter'
        dataframe.loc[(dataframe[time] >= strt_sprg) &
                      (dataframe[time] < strt_summ), 'season'] = 'spring'
        dataframe.loc[(dataframe[time] >= strt_summ) &
                      (dataframe[time] < strt_fall), 'season'] = 'summer'
        dataframe.loc[(dataframe[time] >= strt_fall) &
                      (dataframe[time] < strt_w_nx), 'season'] = 'fall'

    return dataframe


def tetha2hd(angle):
    """ Function tta2hd usage :           output    = tta2hd(angle)

    Takes as angle angle vector in cartesian coordinates running counter-
    clockwise from 0 (east) to 360 degrees and transforms it into a heading
    vector running clockwise from 0 to the north to 360 degrees. 

    """


    angle            = angle - 90;
    angle            = 360 - angle ;
    angle[angle < 0] = angle[angle < 0] + 360 ;
    output           = angle % 360 ;
    return output


def uv2hd(u,v):
    """ Function uv2hd usage :          nm, hd      = uv2hd(u,v)

    Converts vector components (u in the abscissa and v in the ordinates) of a 2D space
    to a heading angle 'hd' in degrees with 0 towards positive y and 90 towards positive x.
    Also returns the norm of vector (u,v) to variable 'nm'.

    """
    I       = np.logical_and(np.isfinite(u), np.isfinite(v))
    out     = np.nan*np.ones(u.shape)

    hd      = np.arctan2(v[I],u[I])
    hd      = hd*180/np.pi
    hd      = r1802360(hd)
    hd      = tta2hd(hd)
    out[I]  = hd
    nm      = np.sqrt(u**2+v**2)

    return nm, out


def xr_SA_CT_pden(dataset,
                  t='temperature',
                  s='salinity',
                  p='z',
                  lon='longitude',
                  lat='latitude'):
    """
    Returns an xarray dataset width added CT, SA and sigma0
    columns as calculated by the Gibbs Seawater package gsw.
    """
    # Make sure spatial coordinate fields can broadcast with variables
    new_dims = broadcastable(dataset[t].values, dataset[p].values)
    pressure = np.reshape(dataset[p].values, new_dims)
    new_dims = broadcastable(dataset[t].values, dataset[lon].values)
    longitude = np.reshape(dataset[lon].values, new_dims)
    new_dims = broadcastable(dataset[t].values, dataset[lat].values)
    latitude = np.reshape(dataset[lat].values, new_dims)

    # Get absolute salinity
    SA = gsw.SA_from_SP(dataset[s].values,
                        pressure,
                        longitude,
                        latitude)

    # Get conservative temperature
    CT = gsw.CT_from_t(SA,
                       dataset[t].values,
                       pressure)

    # Get in situ density
    RHO = gsw.rho(SA, CT, pressure)

    # Get density anomaly
    sigma0 = gsw.density.sigma0(SA, CT)

    # Add required fields to dataset
    dataset['SA'] = (dataset[t].dims, SA)
    dataset['CT'] = (dataset[t].dims, CT)
    dataset['sigma0'] = (dataset[t].dims, sigma0)

    return dataset


def dayofyear2dt(days, yearbase):
    """ Function yb2dt() syntax:        out     = yb2dt(input,yearbase)

    Takes as input days since january first of the year specified by yearbase and
    returns a list of datetime objects.

    Convert UTC days of year since January 1, 00:00:00 of `yearbase` to datetime array.

    Parameters
    ----------
    days : 1D array
        Dates in day of year format.
    yearbase : int
        Year when the data starts.

    Returns
    -------
    1D array
        Datetime equivalent of day of year dates.

    """
    start = np.array(['%d-01-01' % yearbase], dtype='M8[us]')
    deltas = np.array([np.int32(np.floor(days * 24 * 3600))], dtype='m8[s]')
    return (start + deltas).flatten()
