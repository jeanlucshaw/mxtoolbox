"""
Mathematical, geometrical and simple matrix operations.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from math import radians, cos, sin, asin, sqrt
import shapely.speedups
# from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon
from scipy.interpolate import interp1d
# import mxtoolbox.process.convert.crs2crs as crs2crs
import scipy.stats as ss
import matplotlib.path as mpltPath
import cartopy.crs as ccrs
shapely.speedups.enable()

__all__ = ['array_corners',
           'array_outline',
           'broadcastable',
           'circular_distance',
           'consecutive_duplicates',
           'date_abb',
           'date_full',
           'doy_mean',
           'doy_std',
           'destination_point',
           'distance_along_bearing',
           'f_gaussian',
           'f_sine',
           'haversine',
           'increase_resolution',
           'in_polygon',
           'get_contour_xy',
           'lonlat_rectangle',
           'perpendicular_line',
           'project_to_line',
           'proj_grid',
           'proximity_group',
           'polygon_area',
           'rotate_frame',
           'square_dimensions',
           'xr_abs',
           'xr_contour',
           'xr_time_step',
           'xr_unique',
           'mxy2abc']


# Brief functions
date_abb = lambda x: '%s, %.0f' % (x.month_name()[:3], x.day)
date_full = lambda x: '%s %.0f' % (x.month_name(), x.day)


def array_outline(array):
    """
    Select outer limits of numpy array.

    Parameters
    ----------
    array: 2D array
        The data from which to extract an outline.

    Returns
    -------
    1D array
        Outer limits of the input array.

    """
    return np.hstack((array[:-1, 0], array[-1, :-1], array[::-1, -1], array[0, ::-1]))


def array_corners(a):
    """
    Select corners of numpy array.

    Parameters
    ----------
    array: 2D array
        The data from which to extract corners.

    Returns
    -------
    1D array of size 4
        Corners of the input array.

    """
    return np.array([a[0, 0], a[-1, 0], a[0, -1], a[-1, -1]])


def broadcastable(a, b):
    """
    Return shape in which b could be broadcasted with a.

    If arrays can not be broadcast None is returned, therefore this
    function can also serve as a boolean test answering if b be broadcast
    to a through appending through prepending and appending empty dimensions.

    Parameters
    ----------
    a : ndarray
        Array whose shape will not change.
    b : ndarray
        Array to broadcast to `a`.

    Returns
    -------
    tuple or None
        If `tuple`, numpy.reshape(b, `tuple`) broadcasts to `a`, otherwise
        arrays can not be broadcast.

    """
    # Fail if input conditions are not respected
    if len(a.shape) < len(b.shape):
        raise TypeError('a must be matrix of higher or same order as b')

    # Initialize lists
    b_remaining_dims = list(b.shape)
    b_broadcast_dims = []

    # Loop backwards through a dimensions
    for dim_s in a.shape[::-1]:
        if dim_s in b_remaining_dims:
            b_broadcast_dims.insert(0, dim_s)
            b_remaining_dims.remove(dim_s)
        else:
            b_broadcast_dims.insert(0, 1)

    if b_remaining_dims:
        # Arrays can not be broadcast
        return None
    else:
        # Return broadcast dimensions
        return tuple(b_broadcast_dims)


def circular_distance(a1, a2, units='rad'):
    '''
    Function circdist usage:

        d   =   circdist(a1,a2,units='rad')

    Returns to 'd' the distance between angles a1 and a2
    expected to be radians by default, or degrees if units
    is specified to 'deg'.

    Parameters
    ----------
    a1, a2 : float
        Input angle.
    units: str
        Units of input angles ('deg', 'rad')

    Returns
    -------
    float
        Angular distance between `a1` and `a2`.

    '''
    if units == 'deg':
        a1 = np.pi*a1/180
        a2 = np.pi*a2/180

    if np.isscalar(a1) and np.isscalar(a2):
        v1 = np.array([np.cos(a1), np.sin(a1)])
        v2 = np.array([np.cos(a2), np.sin(a2)])
        dot = np.dot(v1, v2)
    elif not np.isscalar(a1) and np.isscalar(a2):
        a2 = np.tile(a2, a1.size)
        v1 = np.array([np.cos(a1), np.sin(a1)]).T
        v2 = np.array([np.cos(a2), np.sin(a2)]).T
#        dot =   np.diag( v1 @ v2.T )
        dot = (v1 * v2).sum(-1)
    else:
        v1 = np.array([np.cos(a1), np.sin(a1)]).T
        v2 = np.array([np.cos(a2), np.sin(a2)]).T
#        dot =   np.diag( v1 @ v2.T )
        dot = (v1 * v2).sum(-1)

    res = np.arccos(np.clip(dot, -1., 1.))

    if units == 'deg':
        res = 180*res/np.pi

    return res

def consecutive_duplicates(x_pts, y_pts):
    """
    Find consecutive duplicate coordinates in 2D space.

    Given coordinates in 2D space `x_pts` and `y_pts`, a
    consecutive duplicate point has the same x and y values
    as the point before it.

    Parameters
    ----------
    x_pts, y_pts : 1D array
        Input coordinates in 2D space.

    Returns
    -------
    x_nd, y_nd : 1D array
        Input coordinates with consecutive duplicates removed.
    duplicates : 1D array of bool
        True at index value of consecutive duplicates.

    """
    index = np.arange(x_pts.size)
    duplicate = np.array(
        [False, *((np.diff(index) == 1) & (np.diff(x_pts) == 0) & (np.diff(y_pts) == 0))])
    x_nd, y_nd = x_pts[~duplicate], y_pts[~duplicate]

    return x_nd, y_nd, duplicate


def doy_mean(series, abb=True):
    """
    Circular mean pandas.Timestamp series as day of year.

    Parameters
    ----------
    series: pandas.Series of pandas.Timestamp
        Series to analyse.

    Returns
    -------
    float:
        Average day of year of series in units of days.
    """
    # Exclude NaN before computing mean
    series = series.loc[(~series.isnull())]

    # Compute circular mean
    doy = ss.circmean(series.apply(lambda x: x.dayofyear), low=1, high=365)

    # Generate text label
    timestamp = pd.Timestamp('2000-01-01') + pd.Timedelta(int(round(doy)), 'D')
    if abb:
        label = date_abb(timestamp)
    else:
        label = date_full(timestamp)
    return doy, label


def doy_std(series):
    """
    Circular std pandas.Timestamp series as day of year.

    Parameters
    ----------
    series: pandas.Series of pandas.Timestamp
        Series to analyse.

    Returns
    -------
    float:
        Standard deviation in units of days.
    """
    # Exclude NaN before computing mean
    series = series.loc[(~series.isnull())]

    return ss.circstd(series.apply(lambda x: x.dayofyear), low=1, high=365)


def _distance(xpts, ypts, x0, y0):
    """
    Return cartesian distance from (xpts, ypts) to (x0, y0).
    """
    return np.sqrt((xpts - x0) ** 2 + (ypts - y0) ** 2)


def destination_point(x0, y0, distance, bearing, meters_per_unit=1000):
    """
    Find coordinates of a point `distance` km in initial direction `bearing`.

    Let :math:`\lambda_0, \phi_0` be the longitude and latitude of some
    initial coordinate. On a spherical earth, coordinates of a point
    :math:`\lambda_f, \phi_f` a distance `d` along a great-circle line
    with initial bearing :math:`\\theta` can be computed as,

    .. math::

       \\phi_f = \\sin^{-1}\\left(\\sin\\phi_0\\cos\\delta + \\cos\\phi_0\\sin\\delta\\cos\\theta\\right)

    and,

    .. math::

       \\lambda_f = \\lambda_0 + \\tan^{-1}\\left(\\frac{\\sin\\theta\\sin\\delta\\cos\\phi_0}
       {\\cos\\delta - \\sin\\phi_0\\sin\\phi_f}\\right)

    where :math:`\delta` is the `d` divided by the earth's radius set
    at 6371 km.

    Parameters
    ----------
    x0, y0 : float
        Starting longitude and latitude.
    distance : float
        Distance to travel.
    bearing : float
        Initial azimuth in degrees, 0 north and 90 east.
    meters_per_unit : float
        Defines the unit of `distance`. Defaults to 1000 for km.

    Returns
    -------
    float
        Longitude and latitude of destination point.

    """
    # Unit conversions
    delta = meters_per_unit * distance / 6371008.8
    phi_0 = y0 * np.pi / 180
    lbd_0 = x0 * np.pi / 180
    theta = bearing * np.pi / 180

    # Destination latitude in radians
    phi_f = np.arcsin(np.sin(phi_0) * np.cos(delta) +
                      np.cos(phi_0) * np.sin(delta) * np.cos(theta))

    # Destination longitude in radians
    arg_y = np.sin(theta) * np.sin(delta) * np.cos(phi_0)
    arg_x = np.cos(delta) - np.sin(phi_0) * np.sin(phi_f)
    lbd_f = lbd_0 + np.arctan2(arg_y, arg_x)

    return (180 * lbd_f / np.pi, 180 * phi_f / np.pi)


def distance_along_bearing(xpts,
                           ypts,
                           bearing,
                           x0=0,
                           y0=0,
                           dfunc=None):
    """
    Return distance from (`x0`, `y0`) in an arbitrary direction.

    Project coordinates defined by (`xpts`, `ypts`) to the nearest point
    on a line passing through (`x0`, `y0`) and who's angle to the x axis
    is given by `bearing`. Then use `dfunc` to calculate distance of them
    projected points to the origin. The distance function is assumed to
    be of the form,

        distance = dfunc(xpts, x0, ypts, y0)

    and defautls to cartesian distance,

        distance = np.sqrt((xpts - x0) ** 2 + (ypts - y0) ** 2)

    Points where ypts < y0 are returned as negative along the new dimension.

    Parameters
    ----------
    xpts, ypts : array_like
        Input coordinates.
    bearing : float
        Direction along which to measure distance (degrees).
    x0, y0 : float
        Origin coordinates.
    dfunc : Object
        Distance calculating function.

    Returns
    -------
    array_like
        Distance along new dimension.

    """
    # Project coordinates to bearing line
    # a = -np.tan(bearing * np.pi / 180)
    # b = 1
    # c = -y0
    a, b, c = mxy2abc(np.tan(bearing * np.pi / 180), x0, y0)
    xprj, yprj = project_to_line(xpts, ypts, a, b, c)

    # Calculate distance
    if dfunc:
        distance = dfunc(xprj, yprj, x0, y0)
    else:
        distance = _distance(xprj, yprj, x0, y0)

    # Make negative distances where y < y0
    if np.array(yprj).size > 1:
        distance[yprj < y0] = -distance[yprj < y0]
    else:
        if yprj < y0:
            distance = -distance

    return distance


def f_gaussian(x, a_1, a_2, a_3, a_4):
    """
    Calculates,

    .. math::

       y = a_1 \\exp\\left(-\\frac{(x - a_2)^2}{a_3}\\right) + a_4

    """
    return a_1 * np.exp(-(x - a_2) ** 2 / a_3) + a_4


def f_sine(x, a_1, a_2, a_3, a_4):
    """
    Calculates,

    .. math::

       y = a_1 \\sin\\left((x - a_2)a_3\\right) + a_4

    """
    return a_1 * np.sin((x - a_2) * a_3) + a_4


def get_contour_xy(contour):
    """
    Return (x, y) coordinates of contour line.

    Input is assumed to be a QuadContourSet object
    containing only one contour line.

    Parameters
    ----------
    contour : QuadContourSet
        Object returned by matplotlib.pyplot.contour

    Returns
    -------
    ndarray
        Coordinates (x, y) as array of shape (m, 2)

    """
    nctr = len(contour.collections[0].get_paths())
    v = [contour.collections[0].get_paths()[i].vertices for i in range(nctr)]
    return v


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points.

    Used to calculate the distance between points defined in decimal
    degrees. Uses the equation,

    .. math::

        h = 2R\\sin^{-1}\\left(\\left[ 
        \\sin^{-1}\\left(\\frac{lat_2-lat_1}{2}\\right)^2 +
        \\cos(lat_1)\\cos(lat_2)\\left(\\frac{lon_2-lon_1}{2}\\right)^2
        \\right]^{1/2}\\right)

    where the earth radius `R` is set to 6371 km.

    Parameters
    ----------
    lon1, lat1 : float
        Compare to this coordinate (decimal degrees).
    lon2, lat2 : array_like
        Get distance from (`lon1`, `lat1`) for these coordinates (decimal degrees).

    Returns
    -------
    array_like
        Distance between coordinates2 and coordinate1 (meters).

    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.arcsin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def in_polygon(xpts, ypts, x_poly, y_poly, lib='mpl'):
    """
    Find points inside an arbitraty polygon.

    Given coordinates of 2D space (`xpts`, `ypts`), returns a
    boolean array of the same size as `xpts` and `ypts` where
    True values indicate coordinates inside the polygon
    defined by (`x_poly`, `y_poly`). Setting the `lib` parameter
    chooses to use tools from matplotlib.path or shapely.

    Parameters
    ----------
    xpts, ypts : array_like
        Input coordinates.
    x_poly, y_poly: array_like
        Polygon coordinates.
    lib : str
        Library to use ('mpl' or 'shp')

    Returns
    -------
    boolean array
        True for points inside polygon.

    """
    if lib == 'shp':
        # Polygon border
        poly = Polygon([(xp, yp) for (xp, yp) in zip(x_poly, y_poly)])

        # Bool vector
        boolean = [poly.contains(Point(x_pt, y_pt))
                   for (x_pt, y_pt)
                   in zip(xpts, ypts)]
    else:
        # Set up input
        pts = np.array([xpts, ypts]).T
        poly = mpltPath.Path([[xp, yp] for (xp, yp) in zip(x_poly, y_poly)])

        # Bool vector
        boolean = poly.contains_points(pts)

    return boolean


def increase_resolution(xpts, ypts, N, offset_idx=0):
    """
    Get high resolution track from low resolution track.

    Create a higher resolution track described in 2D by xpts and
    ypts, 2 numpy 1D arrays, with N linearly equidistant points
    added in between.

    xpts and ypts are assumed to be longitudes and latitudes and
    along track distance from the first point is returned in
    km.

    Parameters
    ----------
    xpts, ypts : array_like
        Coordinates of low resolution track.
    N : int
        Number of points to have between input coordinates.
    offset_idx : int
        Index of the LR track used as origin in distance calculation.

    Returns
    -------
    array_like
        HR track horizontal coordinate.
    array_like
        HR track vertical coordinate.
    array_like
        HR track distance from origin.

    """
    xout = xpts[0]
    yout = ypts[0]
    dout = np.array(0)
    for (x1, y1, x2, y2) in zip(xpts[:-1], ypts[:-1], xpts[1:], ypts[1:]):
        xout = np.hstack((xout, np.linspace(x1, x2, N)[1:]))
        yout = np.hstack((yout, np.linspace(y1, y2, N)[1:]))
    for (x1, y1, x2, y2) in zip(xout[:-1], yout[:-1], xout[1:], yout[1:]):
        dout = np.hstack((dout, haversine(x1, y1, x2, y2)))
    dout = np.cumsum(dout) / 1000

    # Make the distance with respect to poin offset_idx
    origin = np.flatnonzero(np.logical_and(xout == xpts[offset_idx],
                                           yout == ypts[offset_idx]))
    dout -= dout[origin]

    return xout, yout, dout


def lonlat_rectangle(lon_0, lat_0, bearing, box_len_a, box_wid_a, box_len_b=None, box_wid_b=None, axes=None):
    """
    Draw an almost rectangular polygon in plate carree coordinates.

    Parameters
    ----------
    lon_0, lat_0: float
        The origin of the rectangle.
    bearing: float
        The rectangle's tilt eastward from north (degrees).
    box_len_a, box_len_b: float
        Long dimensions of the rectangle (km).
    box_wid_a, box_wid_b: float
        Short dimensions of the rectangle (km).
    axes: cartopy.GeoAxes
        Plot to these axes if supplied.

    Returns
    -------
    lon_box, lat_box: 1D array
        Rectangle vertices.
    lon_center, lat_center: 1D array
        Vectices `a` and `b`.

    Note
    ----
    For definitions of the input dimensions refer to the following diagram.

    ..code::

       e -bwa- b --bwb-- f
       |       |         |
       |       |         |
       |      blb        |
       |       |         |
       | ----- 0 ------- |
       |       |         |
       |      bla        |
       |       |         |
       c -bwa- a --bwb-- d
               |
               V
            bearing

    """
    def _get_vertex(lon_0, lat_0, bearing, distance, name):
        lon_v, lat_v = destination_point(lon_0, lat_0, distance, bearing)
        if axes is not None:
            axes.plot(lon_v, lat_v, '.', transform=ccrs.PlateCarree())
            axes.text(lon_v, lat_v, name, transform=ccrs.PlateCarree())
        return lon_v, lat_v

    # Default has same dimensions both sides of the origin
    box_len_b = box_len_b or box_len_a
    box_wid_b = box_wid_b or box_wid_a

    # Get the individual vertices
    lon_a, lat_a = _get_vertex(lon_0, lat_0, bearing, box_len_a, 'a')
    lon_b, lat_b = _get_vertex(lon_0, lat_0, bearing - 180, box_len_b, 'b')
    lon_c, lat_c = _get_vertex(lon_a, lat_a, bearing + 90, box_wid_a, 'c')
    lon_d, lat_d = _get_vertex(lon_a, lat_a, bearing - 90, box_wid_b, 'd')
    lon_e, lat_e = _get_vertex(lon_b, lat_b, bearing - 90, box_wid_b, 'e')
    lon_f, lat_f = _get_vertex(lon_b, lat_b, bearing + 90, box_wid_a, 'f')

    # Format vertices for output
    lon_box = np.array([lon_c, lon_d, lon_e, lon_f, lon_c])
    lat_box = np.array([lat_c, lat_d, lat_e, lat_f, lat_c])
    lon_center = np.array([lon_a, lon_b])
    lat_center = np.array([lat_a, lat_b])

    # Plot polygon if requested
    if axes is not None:
        axes.plot(lon_box, lat_box, 'k', transform=ccrs.PlateCarree())
        axes.plot([lon_a, lon_b], [lat_a, lat_b], 'k--', transform=ccrs.PlateCarree())

    return lon_box, lat_box, lon_center, lat_center


def polygon_area(xpts, ypts, lonlat=False, pos=True):
    return None
#     """
#     Compute polygon area.

#     Use Greene's theorem to compute polygon area. Input vectors
#     can be longitudes and latitudes if lonlat is specified. Otherwise
#     they are assumed coordinates of linear space.

#     Parameters
#     ----------
#     xpts, ypts : array like
#         Polygon coordinates.
#     lonlat : bool
#         If input is lonlat, convert to CEA projection.
#     pos : bool
#         Return absolute value of area, defaults to True.

#     Returns
#     -------
#     float
#         Polygon area. Units are coordinate dependent.

#     """
#     # Ensure polygon is closed
#     if xpts[0] != xpts[-1] or ypts[0] != ypts[-1]:
#         xpts, ypts = np.hstack((xpts, xpts[0])), np.hstack((ypts, ypts[0]))

#     # If in longitudes and latitudes, convert to linear 2D space
#     if lonlat:
#         cea = Basemap(projection='cea',
#                       llcrnrlat=ypts.min(),
#                       urcrnrlat=ypts.max(),
#                       llcrnrlon=xpts.min(),
#                       urcrnrlon=xpts.max())
#         xpts, ypts = cea(xpts, ypts)

#     # Compute area
#     area = 0.5 * np.sum(ypts[:-1] * np.diff(xpts) - xpts[:-1] * np.diff(ypts))

#     # By default, disregard sign
#     if pos:
#         area = np.abs(area)

#     return area


def project_to_line(xpts,
                    ypts,
                    a,
                    b,
                    c):
    """
    Find point on line nearest to coordinate.

    Given the points (`xpts`, `ypts`) and line `d` defined by,

    .. math::

       ax + by + c = 0

    coordinates on line `d` nearest to each point are given by,

    .. math::

       x = \\frac{b(bx_{pts} - ay_{pts}) - ac}{a^2 + b^2}

    and,

    .. math::

       y = \\frac{a(-bx_{pts} - ay_{pts}) - bc}{a^2 + b^2}

    Parameters
    ----------
    xpts, ypts : array_like
        Coordinates to project on line.
    a, b, c : float
        Factors defining line `d`.

    Returns
    -------
    x, y : array_like
        Points on line `d` nearest to each input coordinate.

    """
    x = (b * (b * xpts - a * ypts) - a * c) / (a ** 2 + b ** 2)
    y = (a * (-b * xpts + a * ypts) - b * c) / (a ** 2 + b ** 2)
    return x, y


def _distance(xpts, ypts, x0, y0):
    """
    Return cartesian distance from (xpts, ypts) to (x0, y0).
    """
    return np.sqrt((xpts - x0) ** 2 + (ypts - y0) ** 2)


def perpendicular_line(m, x, y):
    """
    Find slope and intercept of perpendicular line.

    Parameters
    ----------
    m: float
        Slope of original line.
    x, y: float
        Coordinates of one point along original line.

    Returns
    -------
    float
        Perpendicular line slope.
    float
        Perpendicular line intercept.

    """
    slope = -1 / m
    intercept = y + (x / m)
    return slope, intercept


def proj_grid(lon_0, lat_0, dx, nx, ny, bearing, dy=None):
    """
    Define a regular grid over geographical space.

    Parameters
    ----------
    lon_0, lat_0: float
        PlateCarree coordinates of mean grid position.
    dx, dy: float
        Horizontal and vertical resolutions in meters.
    nx, ny: int
        Horizontal and vertical number of grid points.
    bearing: float
        Compass orientation of horizontal dimension.

    Returns
    -------
    2D array
        Horizontal and vertical grid coordinates.
    cartopy.crs
        Coordinate reference system in which grid is defined.

    """
    def _crs2crs(x, y, target, origin=ccrs.PlateCarree()):
        """
        Using the version in convert results in a circular import.
        """
        x, y = np.array([x, x]), np.array([y, y])
        transformed = target.transform_points(origin, x, y)
        x_dest, y_dest = transformed[0, 0], transformed[0, 1]
        return x_dest, y_dest

    # Default to equal spacing in both directions
    dy = dy or dx

    # Get equal area projection
    pj_sine = ccrs.Sinusoidal(central_longitude=lon_0)
    x_g = np.arange(0, (nx + 1) * dx, dx)
    y_g = np.arange(0, (ny + 1) * dy, dy)
    XG, YG = np.meshgrid(x_g, y_g)

    # Set to required center
    x_0, y_0 = _crs2crs(lon_0, lat_0, pj_sine)
    

    XG = XG - XG.mean() + x_0
    YG = YG - YG.mean() + y_0

    # Rotate
    XG, YG = rotate_frame(XG, YG, (90 - bearing) * np.pi / 180, inplace=True) 

    return XG, YG, pj_sine


def proximity_group(array, distance):
    """
    Group consecutive array values by proximity.

    Proximity grouping of values in an irregular but
    monotonously increasing `array`. Adjacent values
    are grouped if the distance between them is less than
    or equal to `distance`.

    Parameters
    ----------
    array : array_like
        Monotonously increasing values.
    distance : same type as array
        Maximum distance to neighboor.

    Returns
    -------
    gid : array_like
        Group ID of each value in growing integers.
    gn : array_like
        Number of values per group.
    gindex : list
        Index values in `array` divided by group.

    """
    # Init
    gindex = []
    k = 0
    gid = np.zeros(array.size)

    # Group ID for each element
    for i in range(1, array.size):
        if array[i] - array[i-1] <= distance:
            gid[i] = k
        else:
            k += 1
            gid[i] = k

    # Number of elements per group
    #             and
    # Index numbers for each group
    gn = np.zeros(int(gid.max()+1))
    for i in np.unique(gid):
        gn[int(i)] = np.sum(gid == int(i))
        gindex.append(np.flatnonzero(gid == int(i)))

    return gid, gn, gindex


def rotate_frame(u, v, angle, units='rad', inplace=False):
    """
    Return 2D data in rotated frame of reference.

    Rotates values of 2D vectors whose component values
    are given by `u` and `v`. This function should be thought
    of as rotating the frame of reference anti-clockwise by
    `angle`.

    Parameters
    ----------
    u, v : array_like
        Eastward and northward vector components.
    angle : float
        Rotate the frame of reference by this value.
    units : str ('deg' or 'rad')
        Units of `angle`.
    inplace : bool
        Rotate around mean instead of origin.

    Returns
    -------
    ur, vr : array_like
        Vector components in rotated reference frame.

    """
    # Size errors
    if u.shape == v.shape:
        (sz) = u.shape
    else:
        raise ValueError("u and v must be of same size")

    # Prepare vectors
    u = u.flatten()
    v = v.flatten()

    if inplace:
        o_u, o_v = np.nanmean(u), np.nanmean(v)
        u -= o_u
        v -= o_v

    # Handle deg/rad opts and build rotation matrix
    angle = angle if units == 'rad' else np.pi * angle / 180
    B = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])

    # Rotate
    ur = (B @ np.array([u, v]))[0, :]
    vr = (B @ np.array([u, v]))[1, :]

    # Reshape
    ur = np.reshape(ur, sz)
    vr = np.reshape(vr, sz)

    if inplace:
        ur += o_u
        vr += o_v

    return ur, vr


def square_dimensions(xx, yy):
    """
    Print euclinian dimensions of polygon `abcd`.

    b -- c
    |    |
    a -- d

    Parameters
    ----------
    xx, yy: 1D array
        Ordered coordinates of points `abcd`.
    """
    x_a, x_b, x_c, x_d = xx[0, 0], xx[0, -1], xx[-1, -1], xx[-1, 0]
    y_a, y_b, y_c, y_d = yy[0, 0], yy[0, -1], yy[-1, -1], yy[-1, 0]

    def _d(_x_a, _y_a, _x_b, _y_b):
        return np.sqrt((_x_b - _x_a) ** 2 + (_y_b - _y_a) ** 2)

    print('From a to b: %.0f' % (_d(x_a, y_a, x_b, y_b)))
    print('From a to d: %.0f' % (_d(x_a, y_a, x_d, y_d)))
    print('From b to c: %.0f' % (_d(x_b, y_b, x_c, y_c)))
    print('From c to d: %.0f' % (_d(x_c, y_c, x_d, y_d)))


def xr_contour(dataarray, levels):
    """
    Calculate contours without plotting.

    Parameters
    ----------
    dataarray: xarray.DataArray
        Assumed to be a 2D array on which the contours are found.
    levels: 1D array
        Levels for which to find coordinates in `dataarray`.

    Returns
    -------
    matplotlib.QuadContourSet
        Contains the contours at levels.

    """
    dummy = plt.axes(label='dummy')
    qcs = dataarray.plot.contour(ax=dummy, levels=levels)
    plt.close(dummy.figure)

    return qcs


def xr_abs(dataset, field):
    """
    Return dataset dataset with field taken absolute.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset on which to operate.
    field : str
        Field to return as absolute.

    Returns
    -------
    xarray.Dataset
        Dataset with field made positive.

    """
    dataset[field] = (dataset[field].dims, np.abs(dataset[field].values))
    return dataset


def xr_time_step(dataset, tname, unit):
    """
    Get median step of the time axis in dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset on which to operate.
    tname : str
        Name of the time variable or coordinate.
    unit: str ('s', 'm' or 'h')
        Unit of the return value.

    Returns
    -------
    float
        Median time step.
    """
    diff = np.array(np.diff(dataset[tname]).tolist())/10**9
    if unit == 's':
        pass
    if unit == 'm':
        diff = diff / 60
    if unit == 'h':
        diff = diff / 3600

    return np.median(diff)


def xr_unique(dataset, dim):
    '''
    Remove duplicates along dimension `dim`.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to operate on.
    dim : str
        Name of dimension to operate along.

    Returns
    -------
    xarray.Dataset
        Dataset with duplicates removed along `dim`.
    '''
    _, index = np.unique(dataset[dim], return_index=True)
    return dataset.isel({dim: index})


def mxy2abc(m, x0, y0):
    """
    For a line defined by,

    .. math::

       y = m (x - h) + b

    reformulate as,

    .. math::

       ax + by + c = 0

    Parameters
    ----------
    m, x0, y0 : float
        Line parameters in form 1.

    Returns
    -------
    a, b, c : float
        Line parameters in form 2.

    """
    return 1, -1/m, y0 / m - x0
