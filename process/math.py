"""
Mathematical, geometrical and simple matrix operations.
"""
import numpy as np
# from math import radians, cos, sin, asin, sqrt
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon
from scipy.interpolate import interp1d

__all__ = ['broadcastable',
           'circular_distance',
           'distance_along_bearing',
           'haversine',
           'increase_resolution',
           'in_polygon',
           'get_contour_xy',
           'project_to_line',
           'proximity_group',
           'polygon_area',
           'rotate_frame',
           'xr_abs',
           'xr_time_step',
           'xr_unique']

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
    if units=='deg':
        a1  =   np.pi*a1/180
        a2  =   np.pi*a2/180

    if np.isscalar(a1) and np.isscalar(a2):
        v1  =   np.array([np.cos(a1),np.sin(a1)])
        v2  =   np.array([np.cos(a2),np.sin(a2)])
        dot =   np.dot(v1,v2)
    elif not np.isscalar(a1) and np.isscalar(a2):
        a2  =   np.tile(a2,a1.size)
        v1  =   np.array([np.cos(a1),np.sin(a1)]).T
        v2  =   np.array([np.cos(a2),np.sin(a2)]).T
#        dot =   np.diag( v1 @ v2.T )
        dot =   (v1 * v2).sum(-1)
    else:
        v1=np.array([np.cos(a1),np.sin(a1)]).T
        v2=np.array([np.cos(a2),np.sin(a2)]).T
#        dot =   np.diag( v1 @ v2.T )
        dot =   (v1 * v2).sum(-1)

    res =   np.arccos( np.clip( dot, -1., 1.) )

    if units=='deg':
        res =   180*res/np.pi

    return res


def _distance(xpts, ypts, x0, y0):
    """
    Return cartesian distance from (xpts, ypts) to (x0, y0).
    """
    return np.sqrt((xpts - x0) ** 2 + (ypts - y0) ** 2)


def distance_along_bearing(xpts,
                           ypts,
                           bearing,
                           x0=0,
                           y0=0,
                           dfunc=None):
    """
    Return distance from (`x0`, `y0`) in an arbitrary direction.

    Summary
    -------

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
    a = -np.tan(bearing * np.pi / 180)
    b = 1
    c = -y0
    xprj, yprj = project_to_line(xpts, ypts, a, b, c)

    # Calculate distance
    if dfunc:
        distance = dfunc(xprj, yprj, x0, y0)
    else:
        distance = _distance(xprj, yprj, x0, y0)

    # Make negative distances where y < y0
    distance[yprj < y0] = -distance[yprj < y0]

    return distance


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
    r = 6371000 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def in_polygon(xpts, ypts, x_poly, y_poly):
    """
    Find points inside an arbitraty polygon.

    Given coordinates of 2D space (`xpts`, `ypts`), returns a boolean array
    of the same size as `xpts` and `ypts` where True values indicate
    coordinates inside the polygon defined by (`x_poly`, `y_poly`).

    Parameters
    ----------
    xpts, ypts : array_like
        Input coordinates.
    x_poly, y_poly: array_like
        Polygon coordinates.

    Returns
    -------
    boolean array
        True for points inside polygon.

    """
    # Polygon border
    poly = Polygon([(xp, yp) for (xp, yp) in zip(x_poly, y_poly)])

    # Bool vector
    return [poly.contains(Point(x_pt, y_pt))
            for (x_pt, y_pt)
            in zip(xpts, ypts)]


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


def polygon_area(xpts, ypts, lonlat=False, pos=True):
    """
    Compute polygon area.

    Use Greene's theorem to compute polygon area. Input vectors
    can be longitudes and latitudes if lonlat is specified. Otherwise
    they are assumed coordinates of linear space.

    Parameters
    ----------
    xpts, ypts : array like
        Polygon coordinates.
    lonlat : bool
        If input is lonlat, convert to CEA projection.
    pos : bool
        Return absolute value of area, defaults to True.

    Returns
    -------
    float
        Polygon area. Units are coordinate dependent.

    """
    # Ensure polygon is closed
    if xpts[0] != xpts[-1] or ypts[0] != ypts[-1]:
        xpts, ypts = np.hstack((xpts, xpts[0])), np.hstack((ypts, ypts[0]))

    # If in longitudes and latitudes, convert to linear 2D space
    if lonlat:
        cea = Basemap(projection='cea',
                      llcrnrlat=ypts.min(),
                      urcrnrlat=ypts.max(),
                      llcrnrlon=xpts.min(),
                      urcrnrlon=xpts.max())
        xpts, ypts = cea(xpts, ypts)

    # Compute area
    area = 0.5 * np.sum(ypts[:-1] * np.diff(xpts) - xpts[:-1] * np.diff(ypts))

    # By default, disregard sign
    if pos:
        area = np.abs(area)

    return area


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


def rotate_frame(u, v, angle, units='rad'):
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

    Returns
    -------
    ur, vr : array_like
        Vector components in rotated reference frame.

    """
    # Size errors
    if u.shape==v.shape:
        (sz) = u.shape
    else:
        raise ValueError("u and v must be of same size")

    # Prepare vectors
    u = u.flatten()
    v = v.flatten()

    # Handle deg/rad opts and build rotation matrix
    angle = angle if units=='rad' else np.pi * angle / 180
    B = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    # Rotate
    ur = (B @ np.array([u, v]))[0, :]
    vr = (B @ np.array([u, v]))[1, :]

    # Reshape
    ur = np.reshape(ur, sz)
    vr = np.reshape(vr, sz)

    return ur, vr


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
