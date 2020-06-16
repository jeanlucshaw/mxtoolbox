"""
Tweak details of matplotlib.
"""
import calendar as cd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..process.math_ import xr_abs


__all__ = ['axlabel_doy2months',
           'axlabel_woy2months',
           'colorbar',
           'colorline',
           'labeled_colorbar',
           'move_axes',
           'text_array']


def axlabel_doy2months(axes):
    """
    Change x axis labels from day of year to month abbreviation.

    Parameters
    ----------
    axes : matplotlib.axes
        Matplotlib axes on which to operate.

    """
    mons = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    axes.set(xticks=days, xticklabels=mons, xlabel=None)
    axes.tick_params(axis='x', which='minor', top=False, bottom=False)


def axlabel_woy2months(axes, place='tick', labels=True, ycoord=-10, ltype='abbr', **kwargs):
    """
    Change x axis labels from week of year to month abbreviation.

    Parameters
    ----------
    axes : matplotlib.axes
        Matplotlib axes on which to operate.

    """
    days = np.array([1,32,60,91,121,152,182,213,244,274,305,335, 365])
    if ltype == 'abbr':
        mons = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec', '']
    elif ltype == 'letter':
        mons = ['J','F','M','A','M','J','J','A','S','O','N','D', '']
    weeks = days/7 + 1
    weeks[0] = 1
    axes.set(xticks=weeks, xticklabels=[], xlabel=None)
    if labels:
        for (week, mon, delta) in zip(weeks[:-1], mons[:-1], np.diff(weeks)):
            if place == 'center':
                axes.text(week + delta/2, ycoord, mon, **kwargs)
            else:
                axes.text(week, ycoord, mon, **kwargs)
    # axes.set_xticklabels(mons, ha=align)
    axes.tick_params(axis='x', which='minor', top=False, bottom=False)


def colorbar(axes,
             mappable,
             label=None,
             pad_size=0.05,
             cbar_size=0.05,
             **cbar_kw):
    """
    Add colorbar to axes.

    Out of the box, pyplot's colorbar presents two inconvenients,

    * It changes the axes size to accomodate the colorbar
    * It requires a second call and a new variable for labeling

    This routine's purpose is to hide often used workarounds from
    the main code. It also takes arrays of two axes as input to
    produce colorbars spanning multiple panels. If `cbar_kw`
    sets the `orientation` parameter to 'vertical', axes[0] should
    be the top axes. If it is set to 'horizontal, axes[0] should
    be the left axes.

    Parameters
    ----------
    axes : matplotlib.axes
        Add colorbar to this (these) axes.
    mappable : matplotlib.artist
        Colored object to match.
    label : str
        Text label of colorbar.
    pad_size : float
        Pad width in fraction of axes width.
    cbar_size : float
        Colorbar width in fraction of axes width.
    **cbar_kw : keyword arguments
        Passed on to pyplot.colorbar.

    Returns
    -------
    cbar : pyplot.colorbar
        Reference to created colorbar object.
    cax : matplotlib.axes
        Axes in which the colorbar is drawn.

    """
    # Orientation is vertical if otherwise not specified
    cbar_kw = {**dict(orientation='vertical'), **cbar_kw}

    # Save axes position
    if type(axes) is np.ndarray:
        left0, bottom0, width0, height0 = axes[0].get_position().bounds
        left1, bottom1, width1, height1 = axes[1].get_position().bounds
        if cbar_kw['orientation'] == 'vertical':
            left, bottom, width, height = left0, bottom1, width0, bottom0 + height0 - bottom1
        else:
            left, bottom, width, height = left0, bottom0, left1 + width1 - left0, height0
    else:
        position = axes.get_position()
        left, bottom, width, height = position.bounds

    # Set colorbar position
    if cbar_kw['orientation'] == 'vertical':
        cbar_pos = (left + width + pad_size * width,
                    bottom,
                    cbar_size * width,
                    height)
    else:
        cbar_pos = (left,
                    bottom - height * cbar_size - pad_size * width,
                    width,
                    height * cbar_size)

    # Create axes for colorbar
    cax = plt.axes(cbar_pos)

    # Draw colorbar
    cbar = plt.colorbar(mappable, cax=cax, **cbar_kw)
    # axes.set_position(position)

    # Colorbar label
    if label:
        if cbar_kw['orientation'] == 'vertical':
            cax.set_ylabel(label)
        else:
            cax.set_xlabel(label)

    return cbar, cax


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'),
              norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def labeled_colorbar(position, col_arr, lab_array, fig, txt_w=None):
    """
    Add a discrete colorbar with labels on the colors.

    Parameters
    ----------
    position : tuple size 4
        (left, bottom, width, height).
    col_arr : array
        RGB or RGBA lists.
    lab_array: str array
        Labels to assign colors.
    fig : pyplot.Figure
        Where to add the colorbar.
    txt_w : boolean array
        Labels to print in white (black is default).

    """
    cbar_ax = fig.add_axes(position)
    steps = len(lab_array)
    stepsize = 1 / steps

    # Form text color array
    tcs = np.array(['k' for i in range(steps)])
    if not txt_w is None:
        tcs[txt_w] = 'w'

    # Graphical elements
    for (color, label, i, tc) in zip(col_arr, lab_array, range(steps-1), tcs):
        if position[2] < position[3]:
            # Vertical colorbar
            cbar_ax.axhspan(i * stepsize, (i + 1) * stepsize, color=color)
            cbar_ax.text(0.5, (i + 0.5) * stepsize, label, color=tc, ha='center', va='center')
    if position[2] < position[3]:
        # Vertical colorbar
        cbar_ax.axhspan((steps - 1) * stepsize, 1, color=col_arr[-1])
        cbar_ax.text(0.5, (steps - 0.5) * stepsize, lab_array[-1], color=tcs[-1], ha='center', va='center')

    # Remove ticks and white space
    cbar_ax.set(xlim=(0, 1),
                ylim=(0, 1),
                xticks=[],
                yticks=[])


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def move_axes(axes, ud=0, rl=0):
    """
    Moves axes ud upwards and rl right-wards.

    Handy when plt.subplots gets you 99% to what you want but
    needs a final nudge here and there.

    Parameters
    ----------
    axes : matplotlib.axes
        Axes to operate on.
    ud : float
        Move panel up by `ud` in axes units (0-1).
    rl : float
        Move panel right by `rl` in axes units (0-1).

    """
    box = axes.get_position()
    box.x0 += rl
    box.x1 += rl
    box.y0 += ud
    box.y1 += ud
    axes.set_position(box)


def text_array(axes, xpts, ypts, labels, fmt=None, xoffset=None, yoffset=None, **kwargs):
    """
    Add multiple text annotations to plot in one call.

    To add a single text label, use `axes.text`. This function
    is a wrapper around it that accepts arrays of coordinates
    and corresponding labels. If anything goes wrong with one
    label (e.g. value not printable, coordinate is missing value)
    it is silently skipped.

    Parameters
    ----------
    axes : pyplot.Axes
        The axes on which to draw.
    xpts, ypts : 1D array
        Coordinates of the labels to draw.
    labels : iterable of str or numeric type
        Labels to draw.
    fmt : str
        Numeric format of the labels, e.g. '%d'.
    xoffset, yoffset : float, 1D array or object
        Horizontal or vertical label offsets in coordinates
        of xpts and ypts.
    **kwargs : keyword arguments
        Passed to pyplot.text.

    """
    if xoffset is None:
        xoffset = np.zeros(np.array(xpts).size)
    elif np.array(xoffset).size == 1:
        xoffset = np.tile(xoffset, np.array(xpts).size)
    if yoffset is None:
        yoffset = np.zeros(np.array(xpts).size)
    elif np.array(yoffset).size == 1:
        yoffset = np.tile(yoffset, np.array(xpts).size)

    for x, y, s, xo, yo in zip(xpts, ypts, labels, xoffset, yoffset):
        try:
            if fmt:
                s = fmt % s
            axes.text(x + xo, y + yo, s, **kwargs)
        except ValueError:
            pass

    return None
