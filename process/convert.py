"""
Simple data conversions. Includes unit conversions and thermodynamic property
conversions.
"""
import gsw
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import warnings
from pyproj import Geod
from shapely.geometry import LineString, Point, Polygon
from scipy.interpolate import interp1d
from datetime import datetime, timezone, timedelta
from .math_ import broadcastable, rotate_frame

__all__ = ['anomaly2rgb',
           'binc2edge',
           'bine2center',
           'crs2crs',
           'dd2dms',
           'dms2dd',
           'degrees_180_to_360',
           'degrees_360_to_180',
           'dt2epoch',
           'gsw_SA_CT_rho_sigma0',
           'hd2uv',
           'lonlat2area',
           'lonlat2distances',
           'lonlat2distancefrom',
           'lonlat2heading',
           'lonlat2speed',
           'lonlat2perimeter',
           'pd_add_seasons',
           'theta2hd',
           'uv2hd',
           'xr_SA_CT_pden',
           'dayofyear2dt']


def anomaly2rgb(values, normal=0.5):
    """
    Map standardized anomaly to discretized red to blue color scheme.

    Parameters
    ----------
    values : float or 1D array
        Standardized anomaly value.
    normal : float
        Defines the absolute range of anomalies for which to return
        a white RGB triplet.

    Returns
    -------
    list
        RGB triplet.

    See Also
    --------

       Color definitions from :code:`/usr/local/bin/anomaly5steps_2RGB.pl`

    """

    # Set white anomaly interval
    values /= normal

    # Make function compatible with scalars or arrays
    if np.array(values).size == 1:
        values = [values]
        scalar = True
    else:
        cols = []
        scalar = False

    # Loop over values to process
    for value in values:

        # Blue 5
        if value <= -5.0:
            col = [0.500, 0.000, 0.900]
        # Blue 4
        elif -5.0 < value <= -4.0:
            col = [0.000, 0.000, 0.900]
        # Blue 3
        elif -4.0 < value <= -3.0:
            col = [0.450, 0.450, 0.900]
        # Blue 2
        elif -3.0 < value <= -2.0:
            col = [0.700, 0.700, 1.000]
        # Blue 1
        elif -2.0 < value <= -1.0:
            col = [0.875, 0.875, 1.000]

        # White
        elif -1.0 < value < 1.0:
            col = [1.0, 1.0, 1.0]

        # Red 1
        elif 1.0 <= value < 2.0:
            col = [1.000, 0.875, 0.875]
        # Red 2
        elif 2.0 <= value < 3.0:
            col = [1.000, 0.700, 0.700]
        # Red 3
        elif 3.0 <= value < 4.0:
            col = [0.900, 0.450, 0.450]
        # Red 4
        elif 4.0 <= value < 5.0:
            col = [0.900, 0.000, 0.000]
        # Red 5
        elif 5.0 <= value:
            col = [0.500, 0.000, 0.000]

        # Missing values set to gray
        else:
            col = [0.6, 0.6, 0.6]

        # Add to RGB triplet list
        if scalar:
            cols = col
        else:
            cols.append(col)

    return cols


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


def crs2crs(x, y, target, origin=ccrs.PlateCarree()):
    """
    Tranform coordinates between coordinate reference systems.

    Note that CRS instances provided to the target and orgin
    parameters must both be called `()`.

    Parameters
    ----------
    x, y: 1D array
        Coordinates in origin CRS.
    target: cartopy.crs
        Destination CRS.
    origin: cartopy.crs
        Original CRS.

    Returns
    -------
    1D array
        Horizontal coordinate in desitination CRS.
    1D array
        Vertical coordinate in desitination CRS.

    """
    if np.array(x).size == 1:
        x, y = np.array([x, x]), np.array([y, y])
        transformed = target.transform_points(origin, x, y)
        x_dest, y_dest = transformed[0, 0], transformed[0, 1]
    elif len(x.shape) <= 1:
        transformed = target.transform_points(origin, x, y)
        x_dest, y_dest = transformed[:, 0], transformed[:, 1]
    else:
        shape = np.array(x).shape
        transformed = target.transform_points(origin, x.flatten(), y.flatten())
        x_dest, y_dest = transformed[:, 0].reshape(shape), transformed[:, 1].reshape(shape)

    return x_dest, y_dest


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


def degrees_180_to_360(angles):
    """
    Change degree range: (-180, 180) -> (0, 360)

    Parameters
    ----------
    angles : 1D array
        The angles to convert.

    Returns
    -------
        Converted angles.

    """
    I = angles < 0
    angles[I] = angles[I] + 360
    return angles


def degrees_360_to_180(angles):
    """
    Change degree range: (0, 360) -> (-180, 180)

    Parameters
    ----------
    angles : 1D array
        The angles to convert.

    Returns
    -------
        Converted angles.

    """
    I = angles > 180
    angles[I] = angles[I] - 360
    return angles


def dt2epoch(datetimes, div=1):
    """
    Convert datetime array to numeric (epoch).

    The output units are defined by the div parameter. For
    reference,

       * :code:`div=1` means seconds.
       * :code:`div=60` means minutes.
       * :code:`div=3600` means hours.
       * :code:`div=86400` means days.

    Parameters
    ----------
    datetimes : 1D array of datetime64
        Times to convert.
    div : float
        Divide times by this value to set unit.

    Returns
    -------
    1D array
        Time in numeric values.
    """
    return datetimes.astype('datetime64[s]').astype('int') / div


def gsw_SA_CT_rho_sigma0(temperature,
                         salinity,
                         pressure,
                         lon=-60,
                         lat=47):
    """
    Get abs. sal., cons. temp, in situ and potential densities.

    Parameters
    ----------
    temperature: float or array
        In situ temperature [degreeC].
    salinity: float or array
        Practical salinity [PSU].
    pressure: float or array
        Sea pressure or depth [m or decibars].
    lon: float or array
        Longitude of measurement [degrees+east].
    lat: float or array
        Latitude of measurement [degrees+north].

    Returns
    -------
    float or array
        Absolute salinity.
    float or array
        Conservative temperature.
    float or array
        In situ density.
    float or array
        Potential density.

    """
    # Get absolute salinity
    SA = gsw.SA_from_SP(salinity, pressure, lon, lat)

    # Get conservative temperature
    CT = gsw.CT_from_t(SA, temperature, pressure)

    # Get in situ density
    rho = gsw.rho(SA, CT, pressure)

    # Get density anomaly
    sigma0 = gsw.density.sigma0(SA, CT)

    return SA, CT, rho, sigma0


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
    u = magnitude * np.sin(np.pi * heading / 180)
    v = magnitude * np.cos(np.pi * heading / 180)
    u, v = rotate_frame(u, v, -rotate_by, units='deg')

    return u, v


def lonlat2area(lon, lat, sq_meters_per_unit=10**6):
    """
    Calculate area of lon, lat polygon using PROJ4.

    Parameters
    ----------
    lon, lat: 1D array
        Coordinates of the polygon.
    sq_meters_per_unit: float
        Number of square meters per output unit (e.g., 10 ** 6 for km^2).

    Returns
    -------
    float
        Area of the polygon.
    """
    # Format as shapely geometry
    plist = [Point(lon_, lat_) for lon_, lat_ in zip(lon, lat)]
    polygon = Polygon(LineString(plist))

    # Calculate area
    geod = Geod(ellps='WGS84')
    area, _ = geod.geometry_area_perimeter(polygon)

    return np.abs(area / sq_meters_per_unit)


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
        Plate carree coordinates to process. Length m.
    kwargs : keyword arguments
        Passed to `pyroj.Geod` .

    Returns
    -------
    heading : 1D array
        Forward azimuth. Length m-1.
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
    time = time.astype('datetime64[ns]')

    # Compute dx
    distances = lonlat2distances(lon, lat, meters_per_unit=meters_per_unit)

    # Compute dt
    time_delta = np.diff(time) / 10 ** 9
    dt_seconds = np.float64(time_delta)

    # Compute speed
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

        speed = distances / dt_seconds * seconds_per_unit

    # Compute centered time vector
    time_centered = time[:-1] + time_delta / 2

    # Interpolate to input time grid
    dataarray = xr.DataArray(speed, dims=['time'], coords={
                             'time': time_centered})

    # Compute heading
    if heading is None:
        heading = lonlat2heading(lon, lat)

    # Compute velocity components
    u, v = hd2uv(heading, speed)

    # Interpolate to input time vector
    dataset = xr.Dataset({'u': ('time', u), 'v': ('time', v), 'spd': ('time', speed)}, coords={'time': time_centered})
    dataset = dataset.interp(time=time, kwargs=dict(fill_value='extrapolate'))
    if top_speed:
        dataset = dataset.where(dataset.spd < top_speed)
    
    return dataset.u.values, dataset.v.values, dataset.spd.values


def lonlat2perimeter(lon, lat, meters_per_unit=10**3):
    """
    Calculate perimeter of lon, lat polygon using PROJ4.

    Parameters
    ----------
    lon, lat: 1D array
        Coordinates of the polygon.
    meters_per_unit: float
        Number of meters per output unit (e.g., 10 ** 3 for km).

    Returns
    -------
    float
        Perimeter of the polygon.
    """
    # Format as shapely geometry
    plist = [Point(lon_, lat_) for lon_, lat_ in zip(lon, lat)]
    polygon = Polygon(LineString(plist))

    # Calculate area
    geod = Geod(ellps='WGS84')
    _, perimeter = geod.geometry_area_perimeter(polygon)

    return np.abs(perimeter / meters_per_unit)


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


def theta2hd(theta):
    """
    Convert angles: (E=0, N=90) to (E=90, N=0).

    Parameters
    ----------
    theta : 1D array
        Angle in x,y reference frame.

    Returns
    -------
    1D array
        Angle as read on compass.

    """
    if np.array(theta).size > 1:
        theta = theta - 90
        theta = 360 - theta
        theta[theta < 0] = theta[theta < 0] + 360
        bearing = theta % 360
    else:
        theta = theta - 90
        theta = 360 - theta
        if theta < 0:
            theta = theta + 360
        bearing = theta % 360
    return bearing


def uv2hd(u, v):
    """
    Convert 2D cartesian (u, v) vectors to polar (r, bearing).

    The returned polar coordinates are distance from 0 and
    angle in degrees with 0 at north and 90 at east.

    Parameters
    ----------
    u, v : 1D arrays
        Data in cartesian form.

    Returns
    -------
    r, bearing : 1D arrays
        Data in polar form.
    """
    # Handle missing values
    I = np.isfinite(u) & np.isfinite(v)
    bearing = np.nan * np.ones(u.shape)

    hd = 180 * np.arctan2(v[I], u[I]) / np.pi   # Values in degrees
    hd = degrees_180_to_360(hd)                 # Change degree range
    hd = theta2hd(hd)                           # Theta to compass
    bearing[I] = hd

    # Calculate vector norms
    r = np.sqrt(u ** 2 + v ** 2)

    return r, bearing


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
