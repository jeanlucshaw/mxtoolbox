"""
Functions to streamline work when using cartopy
"""
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

__all__ = ['cp_mercator_panel']

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
    

