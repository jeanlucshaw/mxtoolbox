"""
Simple data conversions. Includes unit conversions and thermodynamic property
conversions.
"""
import gsw
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from .math_ import broadcastable, rotate_frame

__all__ = ['anomaly2rgb',
           'binc2edge',
           'bine2center',
           'dd2dms',
           'dms2dd',
           'hd2uv',
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


def lonlat2heading(lon, lat):
    '''
    Function ll2heading usage:  h   =   ll2heading(lon,lat)

    Returns the initial heading for a great cirle line between successive
    geographical coordinates given by 'lon', 'lat'. To maintain vector 
    size, the last element is repeated.
    '''

    # 0-360
    lon[lon<0]  =   lon[lon<0]  +   360
    lat[lat<0]  =   lat[lat<0]  +   360

    # Heading to radians
    lon =   lon*np.pi/180
    lat =   lat*np.pi/180

    X   =   np.array([np.cos(lat[i])*np.sin(lat[i+1])-np.sin(lat[i])*np.cos(lat[i+1])*np.cos(lon[i+1]-lon[i]) for i in range(lon.size-1)])
    Y   =   np.array([np.cos(lat[i+1])*np.sin(lon[i+1]-lon[i]) for i in range(lon.size-1)])
    h   =   np.arctan2(Y,X)

    # Radians to heading
    h   =   180*np.array(h)/np.pi

    # 0-360
    h[h<0]      =   h[h<0]  +   360

    # Keep vector size
    h           =   np.append(h,h[-1])

    return h

def lonlat2speed(lon, lat, time):
    '''
    Function llt2spd usage:     u,v,nm,h    =   llt2spd(lon,lat,time)

    Takes as input a series of coordinates and time in either UNIX time, or
    numpy datetime64 format and returns horizontal velocity components 'u'/'v', as
    well as the norm of horizontal velocity 'nm' and heading 'h'.
    '''
    # Manage time
    if np.issubdtype(time[0],np.datetime64):
        time    =   dt64tots(time)

    # Compute norm and heading
    dt  =   np.diff(time)
    tc  =   time[0:-1]  +   0.5*dt
    d   =   np.array([haversine(lon[i],lat[i],lon[i+1],lat[i+1]) for i in range(tc.size)])
    n   =   d/dt
    h   =   ll2heading(lon,lat)

    # Reinterpolate to time vector
    n   =   interp1d(tc,n,fill_value='extrapolate').__call__(time)

    # Project to u,v components
    u   =   n*np.sin(h*np.pi/180)
    v   =   n*np.cos(h*np.pi/180)

    return u,v,n,h


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
