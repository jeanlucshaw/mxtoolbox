"""
Tweak details of matplotlib.
"""
import calendar as cd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..process.math import xr_abs


__all__ = ['axlabel_doy2months',
           'axlabel_woy2months',
           'colorbar',
           'colorline',
           'labeled_colorbar',
           'move_axes']


def axlabel_doy2months(ax):
    """
    Change x axis labels of x to show month abbreviation
    instead of day of year.
    """
    mons = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax.set(xticks=days, xticklabels=mons, xlabel=None)
    ax.tick_params(axis='x', which='minor', top=False, bottom=False)


def axlabel_woy2months(ax, align='center', ycoord=-10, ltype='abbr'):
    """
    Change x axis labels of ax to show month abbreviations
    instead of week of year.
    """
    days = np.array([1,32,60,91,121,152,182,213,244,274,305,335, 365])
    if ltype == 'abbr':
        mons = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec', '']
    elif ltype == 'letter':
        mons = ['J','F','M','A','M','J','J','A','S','O','N','D', '']
    weeks = days/7 + 1
    weeks[0] = 1
    ax.set(xticks=weeks, xticklabels=[], xlabel=None)
    for (week, mon, delta) in zip(weeks[:-1], mons[:-1], np.diff(weeks)):
        ax.text(week + delta/2, ycoord, mon, ha=align)
    # ax.set_xticklabels(mons, ha=align)
    ax.tick_params(axis='x', which='minor', top=False, bottom=False)


def colorbar(axes: 'matplotlib axes or subplot',
             mappable: 'color field to quantify',
             label: 'colorbar label' = None,
             pad_size: 'pad width, fraction of axes width' = 0.05,
             cbar_size: 'cb width, fraction of axes width' = 0.05,
             # **cbar_kw: 'arguments passed to pyplot.colorbar') -> 'colorbar object, colorbar axes':
             **cbar_kw) -> 'colorbar object, colorbar axes':
    """
    Wrapper around matplotlib.pyplot.colorbar

    Its purpose is to:

    1- add colorbars without changing axes position
    2- consolidate colorbar labeling into the same call
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

    position:    tuple size 4, (left, bottom, width, height)
    col_arr:     array of RGB or RGBA lists
    lab_array:   array of labels to assign colors
    fig:         pyplot.Figure, where to add the colorbar
    txt_w:       boolean array, labels to print in white
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


def move_axes(axes: 'matplotlib axes or subplot',
               ud: 'move panel up by' = 0,
               rl: 'move panel right by' = 0) -> 'None':
    """
    Moves axes ud upwards and rl right-wards. Handy when
    plt.subplots gets you 99% to what you want but needs a
    final nudge here and there.
    """
    box = axes.get_position()
    box.x0 += rl
    box.x1 += rl
    box.y0 += ud
    box.y1 += ud
    axes.set_position(box)
