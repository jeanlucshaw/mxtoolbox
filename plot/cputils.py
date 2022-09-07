"""
Functions to streamline work when using cartopy
"""
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import _DEGREE_SYMBOL
from matplotlib.ticker import FormatStrFormatter
from ..process.math_ import destination_point

__all__ = ['cp_map_ruler',
           'cp_mercator_panel',
           'cp_proj',
           'cp_proj_panel',
           'cp_ticks',
           'km_legend']


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


def cp_proj(axes, proj, **m_kwargs):
    """
    Transform axes into a cartopy map projection.

    Parameters
    ----------
    axes : array of pyplot.Axes
        As generated by pyplot.subplots.
    proj : str
        Name of cartopy.crs class to use.
    **m_kwargs : keyword arguments
        Passed to the crs constructor.

    Returns
    -------
    pyplot.Axes
        The new projected axes.
    dict
        Keywords to pass when plotting to these axes.
    """
    # Choose projection
    # if proj == 'PlateCarree':
    #     projection = ccrs.PlateCarree(**m_kwargs)
    # elif proj == 'Mercator':
    #     projection = ccrs.Mercator(**m_kwargs)
    # else:
    #     raise KeyError('Projection %s not implemented to plot.cp_proj_panel.' % proj)
    projection = getattr(ccrs, proj)(**m_kwargs)

    # Get current panel position
    position = axes.get_position()

    # Whiten current axes
    axes.set_visible(False)

    # Output
    geoaxes = axes.figure.add_axes(position, projection=projection)
    map_kw = dict(transform=geoaxes.projection, add_colorbar=False)

    # Reduce frame width
    geoaxes.outline_patch.set_linewidth(0.5)
    
    return geoaxes, map_kw


# def cp_ticks(axes,
#              xlocs,
#              ylocs,
#              labels=False,
#              size=5,
#              lrbt=[True, True, True, True],
#              float_format='%.0f'):
#     """
#     Add tick marks to cartopy map plot.

#     Parameters
#     ----------
#     axes: cartopy.GeoAxes
#         Axes containing the map.
#     xlocs, ylocs: 1D array
#         Longitudes and latidudes of tick marks.
#     labels: bool or 4-list of bool
#         Add text label to tick mark. If list, label [left, right, bottom, top].
#     size: float
#         Length of tick marck.
#     lrbt: 4-list of bool
#         Control addition of ticks to left, right, bottom and top of map.
#     """
#     if isinstance(labels, list):
#         draw_labels = True
#     else:
#         draw_labels = labels

#     # Get map extent
#     xmin, xmax, ymin, ymax = axes.get_extent(crs=ccrs.PlateCarree())

#     # Filter grid locs outside extent
#     xlocs = xlocs[(xlocs >= xmin) & (xlocs <= xmax)]
#     ylocs = ylocs[(ylocs >= ymin) & (ylocs <= ymax)]

#     # Call to Gridliner
#     gl = axes.gridlines(xlocs=xlocs,
#                         ylocs=ylocs,
#                         draw_labels=draw_labels)

#     # Gridliner parameters
#     if isinstance(labels, list):
#         gl.ylabels_left = labels[0]
#         gl.ylabels_right = labels[1]
#         gl.xlabels_bottom = labels[2]
#         gl.xlabels_top = labels[3]
#     else:
#         gl.xlabels_top = False
#         gl.ylabels_right = False

#     gl.xlines = False
#     gl.ylines = False
#     gl.xpadding = 10
#     gl.ypadding = 10
#     gl.xformatter = FormatStrFormatter(float_format+'%s' % _DEGREE_SYMBOL)
#     gl.yformatter = FormatStrFormatter(float_format+'%s' % _DEGREE_SYMBOL)

#     # Get tick positions (top + bottom)
#     xticks_x = gl.xlocator.tick_values(xmin, xmax)
#     bottom_y = np.ones_like(xticks_x) * ymin
#     top_y = np.ones_like(xticks_x) * ymax

#     # Get tick positions (left + right)
#     yticks_y = gl.ylocator.tick_values(ymin, ymax)
#     left_x = np.ones_like(yticks_y) * xmin
#     right_x = np.ones_like(yticks_y) * xmax

#     # Bottom ticks
#     if lrbt[2]:
#         axes.plot(xticks_x, bottom_y, marker=3, ms=size, clip_on=False, transform=ccrs.PlateCarree())

#     # Top ticks
#     if lrbt[3]:
#         axes.plot(xticks_x, top_y , marker=2, ms=size, clip_on=False, transform=ccrs.PlateCarree())

#     # Left ticks
#     if lrbt[0]:
#         axes.plot(left_x, yticks_y, marker=0, ms=size, clip_on=False, transform=ccrs.PlateCarree())

#     # Right ticks
#     if lrbt[1]:
#         axes.plot(right_x, yticks_y, marker=1, ms=size, clip_on=False, transform=ccrs.PlateCarree())
def cp_ticks(geoax,
             xticks,
             yticks,
             labels=True,
             labels_pad_x=3,
             labels_pad_y=3,
             float_fmt='%.0f',
             size=5,
             lrbt=[True, True, True, True]):
    """
    Add tick marks to cartopy map plot.

    Parameters
    ----------
    axes: cartopy.GeoAxes
        Axes containing the map.
    xlocs, ylocs: 1D array
        Longitudes and latidudes of tick marks.
    labels: bool or 4-list of bool
        Add text label to tick mark. If list, label [left, right, bottom, top].
    size: float
        Length of tick marck.
    lrbt: 4-list of bool
        Control addition of ticks to left, right, bottom and top of map.
    """
    if isinstance(labels, bool):
        labels = [labels, labels, labels, labels]
    
    # Get map limits
    xmin, xmax, ymin, ymax = geoax.get_extent(crs=ccrs.PlateCarree())
    xmin, xmax, ymin, ymax = (round(l_, 5) for l_ in [xmin, xmax, ymin, ymax])

    # Remove ticks outside extent
    xticks = xticks[(xticks <= xmax) & (xticks >= xmin)]
    yticks = yticks[(yticks <= ymax) & (yticks >= ymin)]

    # Set label offsets
    labels_lon_pad = (ymax - ymin) / 100 * labels_pad_x
    labels_lat_pad = (xmax - xmin) / 100 * labels_pad_y

    # Get tick positions (top + bottom)
    bottom = np.ones_like(xticks) * ymin
    top = np.ones_like(xticks) * ymax

    # Get tick positions (left + right)
    left = np.ones_like(yticks) * xmin
    right = np.ones_like(yticks) * xmax

    # Bottom ticks
    plot_kw = dict(clip_on=False, ms=size, ls='none', transform=ccrs.PlateCarree())
    text_kw = dict(transform=ccrs.PlateCarree())
    if lrbt[2]:
        geoax.plot(xticks, bottom, marker=3, **plot_kw)
        if labels[2]:
            for x_, y_ in zip(xticks, bottom):
                y_ -= labels_lon_pad
                l_ = (float_fmt+'%s') % (x_, _DEGREE_SYMBOL)
                geoax.text(x_,
                           y_,
                           l_,
                           ha='center',
                           va='top',
                           **text_kw)

    # Top ticks
    if lrbt[3]:
        geoax.plot(xticks, top , marker=2, **plot_kw)
        if labels[3]:
            for x_, y_ in zip(xticks, top):
                y_ += labels_lon_pad
                l_ = (float_fmt+'%s') % (x_, _DEGREE_SYMBOL)
                geoax.text(x_,
                           y_,
                           l_,
                           ha='center',
                           va='bottom',
                           **text_kw)

    # Left ticks
    if lrbt[0]:
        geoax.plot(left, yticks, marker=0, **plot_kw)
        if labels[0]:
            for x_, y_ in zip(left, yticks):
                x_ -= labels_lat_pad
                l_ = (float_fmt+'%s') % (y_, _DEGREE_SYMBOL)
                geoax.text(x_,
                           y_,
                           l_,
                           ha='right',
                           va='center',
                           **text_kw)

    # Right ticks
    if lrbt[1]:
        geoax.plot(right, yticks, marker=1, **plot_kw)
        if labels[1]:
            for x_, y_ in zip(right, yticks):
                x_ += labels_lat_pad
                l_ = (float_fmt+'%s') % (y_, _DEGREE_SYMBOL)
                geoax.text(x_,
                           y_,
                           l_,
                           ha='left',
                           va='center',
                           **text_kw)


def km_legend(geoax,
              lon0,
              lat0,
              distance,
              azimuth,
              wing_size=0.1,
              label='',
              label_offset=0.25,
              **plot_kw):
    """
    Add an size scale to cartopy axes.

    Parameters
    ----------
    geoax: cartopy.Geoaxes
        Where to add the legend.
    lon0, lat0: float
        Left of the legend.
    distance: float
        Size of the legend in km.
    azimuth: float
        Orientation of the legend (0: N, 90: E).
    wing_size: float
        Legend `wing` size as fraction of `distance`.
    label: str
        Label of legend.
    label_offset: float
        As fraction of `distance`.
    **plot_kw: dict
        Passed to pyplot.plot.
    """
    # Manage whether of not the transform is given
    if plot_kw is None:
        plot_kw = dict(color='k', transform=ccrs.PlateCarree())
    elif isinstance(plot_kw, dict) and 'transform' not in plot_kw.keys():
        plot_kw['transform'] = ccrs.PlateCarree()

    # Calculate other end of legend
    lonf, latf = destination_point(lon0, lat0, distance, azimuth)

    # Calculate left wing coords
    _x1, _y1 = destination_point(lon0, lat0, wing_size * distance, azimuth - 90)
    _x2, _y2 = destination_point(lon0, lat0, wing_size * distance, azimuth + 90)
    lonwl, latwl = [_x1, _x2], [_y1, _y2]

    # Calculate left wing coords
    _x1, _y1 = destination_point(lonf, latf, wing_size * distance, azimuth - 90)
    _x2, _y2 = destination_point(lonf, latf, wing_size * distance, azimuth + 90)
    lonwr, latwr = [_x1, _x2], [_y1, _y2]

    # Calculate middle point
    lonm, latm = np.mean([lon0, lonf]), np.mean([lat0, latf])

    # Calculate legend center
    lonl, latl = destination_point(lonm, latm, label_offset * distance, azimuth - 90)

    # Draw
    geoax.plot([lon0, lonf], [lat0, latf], **plot_kw)
    geoax.plot(lonwl, latwl, **plot_kw)
    geoax.plot(lonwr, latwr, **plot_kw)

    # Label
    label_kw = dict(ha='center', va='center', transform=plot_kw['transform'])
    geoax.text(lonl, latl, label, rotation=-azimuth+90, **label_kw)
