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
           'haversine',
           'increase_resolution',
           'in_polygon',
           'get_contour_xy',
           'proximity_group',
           'polygon_area',
           'rotate_frame',
           'xr_abs',
           'xr_time_step',
           'xr_unique']

def broadcastable(a: '<numpy.ndarray>', b: '<numpy.ndarray>') -> 'list or None':
    """
    Return shape in which b could be broadcasted with a. If arrays
    can not be broadcast None is returned, therefore this function
    can also serve as a boolean test answering:

    Can b be broadcast to a through appending through prepending and
    appending empty dimensions?
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


def get_contour_xy(cs):
    """
    Let cs be the handle to a contour with one line. This returns
    its x,y values.
    """
    nctr = len(cs.collections[0].get_paths())
    v = [cs.collections[0].get_paths()[i].vertices for i in range(nctr)]
    return v


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees). Return value
    in meters.
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


def in_polygon(x_pts, y_pts, x_poly, y_poly):
    """
    Return a boolean array identifying points of x_pts, y_pts inside
    the polygon defined by x_poly,y_poly.
    """
    # Polygon border
    poly = Polygon([(xp, yp) for (xp, yp) in zip(x_poly, y_poly)])

    # Bool vector
    return [poly.contains(Point(x_pt, y_pt))
            for (x_pt, y_pt)
            in zip(x_pts, y_pts)]


def increase_resolution(ptsx, ptsy, N, offset_idx=0):
    """
    Create a higher resolution track described in 2D by ptsx and
    ptsy, 2 numpy 1D arrays, with N linearly equidistant points
    added in between.

    ptsx and ptsy are assumed to be longitudes and latitudes and
    along track distance from the first point is returned in
    km. Example:

    trk_lon, trk_lat, trk_dist = points_to_hrtrack(ptsx, ptsy, 100)
    """
    xout = ptsx[0]
    yout = ptsy[0]
    dout = np.array(0)
    for (x1, y1, x2, y2) in zip(ptsx[:-1], ptsy[:-1], ptsx[1:], ptsy[1:]):
        xout = np.hstack((xout, np.linspace(x1, x2, N)[1:]))
        yout = np.hstack((yout, np.linspace(y1, y2, N)[1:]))
    for (x1, y1, x2, y2) in zip(xout[:-1], yout[:-1], xout[1:], yout[1:]):
        dout = np.hstack((dout, haversine(x1, y1, x2, y2)))
    dout = np.cumsum(dout) / 1000

    # Make the distance with respect to poin offset_idx
    origin = np.flatnonzero(np.logical_and(xout == ptsx[offset_idx],
                                           yout == ptsy[offset_idx]))
    dout -= dout[origin]

    return xout, yout, dout


def polygon_area(vecx, vecy, lonlat=False, pos=True):
    """
    Use Greene's theorem to compute polygon area. Input vectors
    can be longitudes and latitudes if bm is specified. Otherwise
    they are assumed coordinates of linear space.

    vecx:    array like, horizontal vector in linear space
    vecy:    array like, vertical vector in linear space
    lonlat:    bool, if input is lonlat, convert to CEA projection
    pos: bool, return absolute value, defaults to True
    """
    # Ensure polygon is closed
    if vecx[0] != vecx[-1] or vecy[0] != vecy[-1]:
        vecx, vecy = np.hstack((vecx, vecx[0])), np.hstack((vecy, vecy[0]))

    # If in longitudes and latitudes, convert to linear 2D space
    if lonlat:
        cea = Basemap(projection='cea',
                      llcrnrlat=vecy.min(),
                      urcrnrlat=vecy.max(),
                      llcrnrlon=vecx.min(),
                      urcrnrlon=vecx.max())
        vecx, vecy = cea(vecx, vecy)

    # Compute area
    area = 0.5 * np.sum(vecy[:-1] * np.diff(vecx) - vecx[:-1] * np.diff(vecy))

    # By default, disregard sign
    if pos:
        area = np.abs(area)

    return area


def proximity_group(A, D):
    """

    Function proxGroup usage:   G,N,I   =   proxGroup(A,D)

    Summary:

        Proximity grouping of values in an irregular but
        monotonically increasing array 'A'. Adjacent values
        are group if the distance between them is less than
        or equal to 'D'. Returned values are the group ID
        of each element 'G', number of elements per group
        'N', and index values by group 'I'.

    """
    # Init
    I = []
    k = 0
    G = np.zeros(A.size)

    # Group ID for each element
    for i in range(1, A.size):
        if A[i] - A[i-1] <= D:
            G[i] = k
        else:
            k += 1
            G[i] = k

    # Number of elements per group
    #             and
    # Index numbers for each group
    N = np.zeros(int(G.max()+1))
    for i in np.unique(G):
        N[int(i)] = np.sum(G == int(i))
        I.append(np.flatnonzero(G == int(i)))

    return G, N, I


def rotate_frame(u,v,ang,units='rad'):
    """

    Rotates values of 2D vector space whose component values
    are given by u and v.

    u,v:    eastward and northward vector component arrays. Must
            be of same size, of arbitrary dimension.

    ang:    the angle by which you turn the frame of reference
            positive being anti-clockwise.

    units:  units of ang, 'rad' (default) for radians and 'deg'
            for degrees.

    """

    # Size errors
    if u.shape==v.shape:
        (sz)    =   u.shape
    else:
        raise ValueError("u and v must be of same size")

    # Prepare vectors
    u   =   u.flatten()
    v   =   v.flatten()

    # Handle deg/rad opts and build rotation matrix
    ang =   ang if units=='rad' else np.pi*ang/180
    B   =   np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]])

    # Rotate
    ur  =   ( B @ np.array([u,v]) )[0,:]
    vr  =   ( B @ np.array([u,v]) )[1,:]

    # Reshape
    ur  =   np.reshape(ur,sz)
    vr  =   np.reshape(vr,sz)

    return ur, vr


def xr_abs(ds, field):
    """
    Return dataset ds with field taken absolute.
    """
    ds[field] = (ds[field].dims, np.abs(ds[field].values))
    return ds


def xr_time_step(ds, tname, unit):
    diff = np.array(np.diff(ds[tname]).tolist())/10**9
    if unit == 'second':
        pass
    if unit == 'minute':
        diff = diff / 60
    if unit == 'hour':
        diff = diff / 3600

    return np.median(diff)


def xr_unique(ds, sort_label):
    '''
    Returns the xarray dataset or dataarry with duplicate values along
    dimension sort_label have been removed.
    '''
    _, index = np.unique(ds[sort_label], return_index=True)
    return ds.isel({sort_label: index})
