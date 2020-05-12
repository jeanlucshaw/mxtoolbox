"""
Functions to streamline work when using cartopy
"""
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from ..process.math_ import destination_point

__all__ = ['cp_map_ruler',
           'cp_mercator_panel',
           'cp_proj_panel']


def cp_map_ruler(axes,
                 x_start,
                 y_start,
                 length,
                 bearing,
                 interval=10,
                 n_minor=4,
                 size=1,
                 space=1,
                 units='km',
                 text_side='east',
                 text_rot=0,
                 text_kw=None,
                 line_kw=None):
    """
    Draw a distance ruler on a cartopy map.

    Each tick is found iteratively by travelling `interval` along
    a great circle line starting at the previous tick towards
    `heading`.

    Parameters
    ----------
    axes : GeoAxes
        The axes on which to draw.
    x_start, y_start : float
        Longitude and latitude or ruler zero.
    length : float
        Distance covered by the ruler (km).
    bearing : float
        Along which destination is calculated between ticks in degrees
        with zero north and 90 east.
    interval : float
        Number of kilometers between ticks.
    n_minor : int
        Number of minor ticks between major ticks.
    size : float
        Tick length. Smaller size means bigger ticks.
    space : float
        Between text and ruler. Bigger means more space.
    units : str
        Unit description to draw on top of the ruler.
    text_side : str
        Display labels `east` or `west` of the ruler.
    text_rot : float
        Rotate text if not diplaying properly (workaround).
    text_kw : dict
        Keyword arguments passed to `axes.text`.
    line_kw : dict
        Keyword arguments passed to `axes.plot`.

    Note
    ----

       * At large scales the ruler will distort but the distance
         between ticks is always right. Distortions are most visible
         at scales > 1000 km and at absolute latitudes > 35.

       * This routine relies on current axes aspect ratio. If it is
         changed after a call to cp_map_ruler, the ruler will be crooked.
         Call just before showing/plotting for best results.

    See Also
    --------
    
       * math_.destination_point

    """

    # Option switches
    if line_kw is None:
        line_kw = {'color': 'k', 'linestyle': '-', 'transform': axes.transAxes}
    else:
        line_kw = {'color': 'k', 'linestyle': '-', **line_kw, 'transform': axes.transAxes}

    text_override = {'va': 'center', 'transform': axes.transAxes, 'rotation_mode': 'anchor'}
    if text_kw is None:
        text_kw = {}
    if text_side == 'east':
        text_sign = 1
        text_kw = {'ha': 'left',  **text_kw, **text_override}
    else:
        text_sign = -1
        text_kw = {'ha': 'right', **text_kw, **text_override}

    # Transforms between map and axis coordinates
    data_to_axis = axes.transData + axes.transAxes.inverted()
    axis_to_data = data_to_axis.inverted()

    # Parameters and vector initialization
    distance = 0
    step_count = length // interval
    lon = np.ones(step_count + 1) * x_start
    lat = np.ones(step_count + 1) * y_start
    tick_x = np.zeros(step_count + 1)
    tick_y = np.zeros(step_count + 1)

    # Adjust for rectangular axes
    lon_size = np.diff(axes.get_xlim())[0]
    lat_size = np.diff(axes.get_ylim())[0]
    aspect = lat_size / lon_size


    # Generate distance points vector
    for i in range(1, lon.size):
        lon[i], lat[i] = destination_point(lon[i-1], lat[i-1], interval, bearing)

        # Transform to axes coordinates
        x1, y1 = data_to_axis.transform((lon[i - 1], lat[i - 1]))
        x2, y2 = data_to_axis.transform((lon[i], lat[i]))

        # Get perpendicular scaled vector
        norm = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 20 * size
        if (i - 1) % (n_minor + 1) != 0:
            norm *= 3
        u, v = -(y2 - y1) / norm * aspect, (x2 - x1) / norm / aspect

        # Draw barbs
        axes.plot([x1, x1 + u],[y1, y1 + v], **line_kw)
        axes.plot([x1, x1 - u],[y1, y1 - v], **line_kw)

        # Label major ticks
        if (i-1) % (n_minor + 1) == 0:
            angle = np.arctan2(v, u) * 180 / np.pi + 180
            axes.text(x1 - space * u * text_sign,
                      y1 - space * v * text_sign,
                      '%d' % distance,
                      rotation=angle + text_rot,
                      **text_kw)

        # Keep track of distance travelled
        distance += interval

    # Draw last barb
    norm = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 50 * size
    if step_count % (n_minor + 1) != 0:
        norm *= 3
    u, v = -(y2 - y1) / norm * aspect, (x2 - x1) / norm / aspect
    axes.plot([x2, x2 + u],[y2, y2 + v], **line_kw)
    axes.plot([x2, x2 - u],[y2, y2 - v], **line_kw)

    # Label last barb if major
    if step_count % (n_minor + 1) == 0:
        axes.text(x2 - space * u * text_sign,
                  y2 - space * v * text_sign,
                  '%d' % distance,
                  rotation=angle + text_rot,
                  **text_kw)

    axes.text(x2 + space * v,
              y2 + space * u * text_sign,
              units,
              rotation=angle + text_rot,
              **{**text_kw, 'ha': 'center', 'va': 'bottom'})

    
    # Draw center of the ruler
    axes.plot(lon, lat, **{**line_kw, 'transform': axes.transData})


def cp_mercator_panel(axes: 'array of subplots',
                      preset: 'GSL or WA' = None,
                      loc: 'array position of panel as iterable' = None,
                      **m_kwargs) -> 'GeoAxes':
    """
    Add a Mercator projection map panel to an existing multipanel figure.

    Usage:

    Create a multipanel figure using plt.subplots(). Pass the axes array as
    input along with the index of the panel to replace with this map as parameter
    loc.
    """
    # Fail if the input is a single panel
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
        loc = np.array([0])
        # raise TypeError("Parameter axes is not an array of subplots")

    # Default to top left if loc not specified
    if not loc:
        loc = tuple(0 for e in axes.shape)

    # Preset projection parameters for often drawn maps
    if preset == 'GSL':         # Gulf of St. Lawrence
        pass
    if preset == 'WA':          # west Atlantic
        m_kwargs = dict(**m_kwargs, central_longitude=-55, latitude_true_scale=45)
    else:
        pass

    # Get current panel position
    position = axes[(*loc,)].get_position()

    # Whiten current axes
    axes[(*loc,)].set_visible(False)

    # Output
    geoaxes = axes[(*loc,)].figure.add_axes(position, projection=ccrs.Mercator(**m_kwargs))
    map_kw = dict(ax=geoaxes, transform=geoaxes.projection, add_colorbar=False)

    return geoaxes, map_kw


def cp_proj_panel(axes, proj, loc=None, **m_kwargs):
    """
    Transform one panel of subplot array to a cartopy map projection.

    Parameters
    ----------
    axes : array of pyplot.Axes
        As generated by pyplot.subplots.
    proj : str
        Name of cartopy.crs class to use.
    loc : iterable
        Array position of the axes to modify.
    **m_kwargs : keyword arguments
        Passed to the crs constructor.

    Returns
    -------
    pyplot.Axes
        The new projected axes.
    dict
        Keywords to pass when plotting to these axes.
    """
    # Fail if the input is a single panel
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
        loc = np.array([0])
        # raise TypeError("Parameter axes is not an array of subplots")

    # Default to top left if loc not specified
    if not loc:
        loc = tuple(0 for e in axes.shape)

    # Choose projection
    if proj == 'PlateCarree':
        projection = ccrs.PlateCarree(**m_kwargs)
    else:
        raise KeyError('Projection %s not implemented to plot.cp_proj_panel.' % proj)

    # Get current panel position
    position = axes[(*loc,)].get_position()

    # Whiten current axes
    axes[(*loc,)].set_visible(False)

    # Output
    geoaxes = axes[(*loc,)].figure.add_axes(position, projection=projection)
    map_kw = dict(ax=geoaxes, transform=geoaxes.projection, add_colorbar=False)

    return geoaxes, map_kw


