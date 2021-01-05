"""
Streamline making typical plots for time series analysis, and
physical oceanography.
"""
import gsw
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter
from .mplutils import text_array, colorbar
from .cputils import cp_proj, cp_ticks
from ..process.math_ import xr_abs, doy_mean, doy_std, date_abb
from ..read.text import list2cm
from ..process.convert import anomaly2rgb, binc2edge, dd2dms

__all__ = ['anomaly_bar',
           'ts_diagram',
           'correlation_text',
           'gebco_bathy',
           'gebco_bathy_contour',
           'gsl_bathy_contourf',
           'gsl_map',
           'gsl_temperature_cast',
           'pd_scorecard',
           'scorecard',
           'scorecard_bottom_monthly',
           'wa_map',
           'xr_plot_pm_patch']


def anomaly_bar(axes,
                coord,
                anomaly,
                box_labels=None,
                bar_label=None,
                bar_label_pos='bottom',
                pad_size=0.1,
                bar_size=0.25,
                orientation='vertical'):
    """
    Add anomaly side bar to current + climatology plot.

    Parameters
    ----------
    axes : matplotlib.axes
        Add anomaly bar to these axes.
    coord : 1D array
        Centers of the anomaly boxes.
    anomaly : 1D array
        Anomaly values.
    box_labels : 1D array or None
        Show these values in boxes when specified, otherwise show `anomaly`.
    bar_label : str
        Add label to bar end (e.g. `box_labels` units).
    bar_label_pos : str
        Placement of `bar_label` either `left`, `right`, `top`, `bottom`
        or first letter of these choices.
    pad_size : float
        Pad width as fraction of axes width.
    bar_size : float
        Bar width as fraction of axes width.
    orientation : str
        Make bar `horizontal` or `vertical`. First letters of these options
        are also accepted values.

    Returns
    -------
    pyplot.Axes
        The axes in which the anomaly bar is drawn.

    """
    # Get axes position
    position = axes.get_position()
    left, bottom, width, height = position.bounds

    # Set anomaly bar position
    if orientation in ['vertical', 'v']:
        bar_pos = (left + width + pad_size * width,
                   bottom,
                   bar_size * width,
                   height)

        # Manage bar and box label position accordingly
        bar_box_x, bar_box_y = 0.5 * np.ones_like(coord), coord

        bar_label_kw = {'ha': 'left', 'va': 'center'}
        if bar_label_pos in ['b', 'bottom']:
            bar_label_x, bar_label_y = 0, -coord[0]
        elif bar_label_pos in ['t', 'top']:
            bar_label_x, bar_label_y = 0, coord[-1] + (coord[-1] - coord[-2])

    elif orientation in ['horizontal', 'h']:
        bar_pos = (left,
                   bottom - height * bar_size - pad_size * width,
                   width,
                   height * bar_size)

        # Manage bar label position accordingly
        bar_box_y, bar_box_x = 0.5 * np.ones_like(coord), coord

        bar_label_kw = {'ha': 'center', 'va': 'center'}
        if bar_label_pos in ['b', 'bottom']:
            bar_label_y, bar_label_x = 0.5, -coord[0]
        elif bar_label_pos in ['t', 'top']:
            bar_label_y, bar_label_x = 0.5, coord[-1] + (coord[-1] - coord[-2])
    else:
        raise ValueError('Unrecognized value %s for paramater `orientation`.' % orientation)

    # Create axes for colorbar
    anomaly_ax = plt.axes(bar_pos)

    # Map anomaly values to colors 
    colors = anomaly2rgb(anomaly)

    # Loop over anomaly boxes and color
    edges = binc2edge(coord)
    for low_lim, up_lim, fc in zip(edges[:-1], edges[1:], colors):

        if orientation in ['v', 'vertical']:
            anomaly_ax.axhspan(up_lim, low_lim, facecolor=fc)
            anomaly_ax.axhline(up_lim, color='k')
            anomaly_ax.axhline(low_lim, color='k')

        elif orientation in ['h', 'horizontal']:
            anomaly_ax.axvspan(up_lim, low_lim, facecolor=fc)
            anomaly_ax.axvline(up_lim, color='k')
            anomaly_ax.axvline(low_lim, color='k')

    # Add box labels or anomaly values if labels not defined
    if box_labels is not None:
        labels = box_labels
    else:
        labels = anomaly
    text_array(anomaly_ax,
               bar_box_x,
               bar_box_y,
               labels,
               fmt='%.1f',
               ha='center',
               va='center')

    # Add bar label
    if bar_label:
        anomaly_ax.text(bar_label_x, bar_label_y, bar_label, **bar_label_kw)

    # Axes parameters
    anomaly_ax.tick_params(which='both', bottom=False, left=False)
    anomaly_ax.set(ylim=axes.get_ylim(), xticklabels=[], yticklabels=[])

    return anomaly_ax


def correlation_text(axes, x, y, b, a, r2, fmt='%.2f', p=None, **text_kw):
    """
    Describe linear fit resutls on graph.

    Parameters
    ----------
    axes: matplotlib.Axes
        On which to plot.
    x, y: float
        Left side of text in data coordinates.
    b, a: float
        Power 0 and 1 coefficients of linear fit.
    r2: float
        Variance explained by independent variable.
    fmt: str
        Text format of floats.
    p: float
        Significance threshold.
    **text_kw: dict
        Passed to axes.text .

    """
    if b > 0:
        b_sign = ' + '
    else:
        b_sign = ' '
        
    interpolator = 'R$^2$ = '+fmt+', y = '+fmt+' x' + b_sign + fmt
    _text = interpolator % (r2, a, b)

    axes.text(x, y, _text, clip_on=False, **text_kw)


def gebco_bathy():
    dataset = xr.open_dataset('/data/atlas/gebco/netCDF/gebco_west_atlantic.nc')
    bathy = dataset.elevation * -1
    return bathy


def gebco_bathy_contour(axes, isobaths, xarray=True, step=1, **ctr_kw):
    """
    Add bathymetric contours to the map in `axes`.

    Parameters
    ----------
    axes: cartopy.GeoAxes or matplotlib.Axes
        Panel to operate on.
    isobaths: 1D array
        Levels of bathymetry to draw.
    step: int
        Coarsen Gebco bathymetry by this factor.
    xarray: bool
        If true read gebco netcdf file. This allows plotting of any depth,
        but is slower than reading precalculated contours for 50, 100, 150
        200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, and
        4500 metres.
    **ctr_kw: dict
        Keyword arguments passed to `contour`.

    Returns
    -------
    qcs: matplotlib.QuadContourSet or None
        The contour collections plotted.

    """
    # Defaul contourf parameters
    ctr_kw = {'transform': ccrs.PlateCarree(), **ctr_kw}

    if xarray:
        # Load data
        dataset = xr.open_dataset('/data/atlas/gebco/netCDF/gebco_west_atlantic.nc')
        bathy = dataset.elevation * -1

        # Limit to current plot area
        lon_min, lon_max, lat_min, lat_max = axes.get_extent(crs=ccrs.PlateCarree())
        bathy = bathy.loc[lat_min: lat_max, lon_min: lon_max]

        # Coarsen
        bathy = bathy[::step, ::step]

        # Plot
        qcs = bathy.plot.contour(levels=isobaths, ax=axes, **ctr_kw)
    else:
        for iso in isobaths:
            ctr = pd.read_csv('/data/atlas/gebco/GSL_%d.dat' % iso,
                              names=['lat', 'lon'],
                              sep='\s+',
                              na_values=-99.99)
            axes.plot(ctr.lon, ctr.lat, **ctr_kw)
        qcs = None

    return qcs


def gsl_temperature_cast(axes, temp, z, lon_lat=None, time=None, **plot_kw):
    """
    Plot a Gulf of St. Lawrence temperature profile.

    For all the information to appear clearly, input an axes instance with
    the `figsize` parameter set to (4, 5).

    Parameters
    ----------
    axes: matplotlib.Axes
        Where to plot the profile.
    temp: 1D array or array-like
        Temperature data to plot.
    z: 1D array or array like.
        Depth coordinate of temperature data.
    lon_lat: 2-tuple
        Position of the CTD cast.
    time: numpy.datetime64
        Date and time of CTD cast

    Returns
    -------
    cartopy.Geoaxes or None
        Geoaxes object of inset map.

    """
    # Profile
    axes.plot(temp, z, **plot_kw)

    # Plot parameters
    axes.set(xlabel=r'T ($^\circ$C)', ylabel='Depth (m)')
    axes.set(xlim=(-2, 22), ylim=(0, z.max() + 10), xticks=np.arange(-2, 24, 2))
    axes.invert_yaxis()

    # Map inset
    if lon_lat is not None:
        inset = axes.figure.add_axes([0.5, 0.2, 0.35, 0.35])
        inset, cp_kw = cp_proj(inset, 'PlateCarree')
        inset.plot(lon_lat[0], lon_lat[1], 'o', mfc='r', mec='r', ms=2)
        inset.set(ylim=(44, 54), xlim=(-69, -56))
        inset.coastlines(resolution='50m', color='gray')

        
        axes.text(0.45, 0.55, 'Lon.: %.0f %.0f\' %.0f"' % dd2dms(lon_lat[0]) , transform=axes.transAxes)
        axes.text(0.45, 0.5, 'Lat.: %.0f %.0f\' %.0f"' % dd2dms(lon_lat[1]) , transform=axes.transAxes)
    else:
        inset = None

    # Annotations
    if time is not None:
        axes.text(0.45, 0.6, 'Date: %s' % str(time)[:10], transform=axes.transAxes)

    return inset


def gsl_bathy_contourf(axes, isobaths, add_cbar=False, step=1, **ctrf_kw):
    """
    Add filled bathymetric contours to the map in `axes`.

    Parameters
    ----------
    axes: cartopy.GeoAxes or matplotlib.Axes
        Panel to operate on.
    isobaths: 1D array
        Levels of bathymetry to draw.
    add_cbar: bool
        Add a colorbar mapped to depth on top of plot.
    step: int
        Coarsen Gebco bathymetry by this factor.
    **ctrf_kw: dict
        Keyword arguments passed to `xarray.DataArray.plot.contourf`.

    Returns
    -------
    qcs: matplotlib.QuadContourSet
        The contour collections plotted.
    cbar: matplotlib.Colorbar
        The colorbar object created.

    """
    # Defaul contourf parameters
    ctrf_kw = {'cmap': 'Blues', 'add_colorbar': False, 'transform': ccrs.PlateCarree(), **ctrf_kw}

    # Load data
    dataset = xr.open_dataset('/data/atlas/gebco/netCDF/gebco_west_atlantic.nc')
    bathy = dataset.elevation * -1

    # Limit to current plot area
    lon_min, lon_max, lat_min, lat_max = axes.get_extent(crs=ccrs.PlateCarree())
    bathy = bathy.loc[lat_min: lat_max, lon_min: lon_max]

    # Coarsen
    bathy = bathy[::step, ::step]

    # Saturate
    bathy = bathy.where(bathy < np.max(isobaths), np.max(isobaths))
    bathy = bathy.where(bathy > np.min(isobaths), np.min(isobaths))

    # Plot
    qcs = bathy.plot.contourf(levels=isobaths, ax=axes, **ctrf_kw)

    # Add colorbar if requested
    if add_cbar:
        cbar_kw = {'orientation': 'horizontal', 'format': FormatStrFormatter('%dm')}
        cbar, _ = colorbar(axes, qcs, pad_size=0.075, loc='top', **cbar_kw)
        cbar.ax.tick_params(labelsize=8)
        cbar.minorticks_off()
    else:
        cbar = None

    return qcs, cbar


def gsl_map(axes,
            resolution='intermediate',
            crs='PlateCarree',
            landcolor='oldlace',
            watercolor='lightgray',
            extent=[-71, -55, 44, 52],
            tick_x_maj=np.arange(-70, -50, 5),
            tick_y_maj=np.arange(42, 54, 1),
            tick_x_min=np.arange(-71, -50, 1),
            tick_y_min=np.arange(42, 54, 0.2)):
    """
    Plot Gulf of St Lawrence map in axes.

    Parameters
    ----------
    axes: matplotlib.Axes
        Axes to replace with GSL map.
    resolution: str
        GSHHS features resolution. Can be one of `coarse`, `low`,
        `intermediate`, `high` or `full`.
    crs: str
        Name of cartopy.crs projection class to use.
    landcolor: str
        Continent and islands filled with this color.
    watercolor: str
        Rivers and lakes filled with this color.
    extent: 4-list
        Longitude and latitude limits of map.
    tick_x_maj, tick_y_maj: 1D array
        Longitude and latidude of major ticks and labels.
    tick_x_min, tick_y_min: 1D array
        Longitude and latitude of minor ticks.

    Returns
    -------
    geoax
        GeoAxes instance of map panel.

    """
    # Replace normal Axes with geoAxes
    geoax, _ = cp_proj(axes, crs)
    geoax.set_extent(extent)

    # Most features are fine in GSHHS
    coast = cfeature.GSHHSFeature(scale=resolution,
                                  levels=[1],
                                  facecolor=landcolor,
                                  edgecolor='k',
                                  linewidth=0.25)
    lakes = cfeature.GSHHSFeature(scale=resolution,
                                  levels=[2],
                                  facecolor=watercolor,
                                  edgecolor=watercolor)
    islands = cfeature.GSHHSFeature(scale=resolution,
                                    levels=[3],
                                    facecolor=landcolor,
                                    edgecolor=landcolor)
    ponds = cfeature.GSHHSFeature(scale=resolution,
                                  levels=[4],
                                  facecolor=watercolor,
                                  edgecolor=watercolor)

    # Add GSHHS features
    geoax.add_feature(coast)
    geoax.add_feature(lakes)
    geoax.add_feature(islands)
    geoax.add_feature(ponds)

    # Rivers are not in GSHHS
    RIVERS = pd.read_csv('/data/atlas/gmt/gulff-rivers-all.lon_lat',
                         names=['lon', 'lat'],
                         sep=r'\s+',
                         na_values=-99.000)
    geoax.plot(-RIVERS.lon.values,
               RIVERS.lat.values,
               color='lightskyblue',
               transform=ccrs.PlateCarree())

    # Ticks
    cp_ticks(geoax, tick_x_maj, tick_y_maj, labels=True, size=3)
    cp_ticks(geoax, tick_x_min, tick_y_min, labels=False, size=1.5)

    return geoax

def scorecard(df, field, ax,
              side_label='',
              color_displays='anomaly',
              value_displays='value',
              anomaly_color_levels=4,
              units_label='',
              cbar_label='',
              clevels=None,
              averages=False,
              colorbar=False,
              hide_xlabels=False,
              hide_xticks=True,
              xlabel_angle=60,
              pad_xlabel=0,
              text_offset=[0., 0.],
              sub_na=None,
              bfmt='%.1f',
              sfmt='%.2f',
              n_cutoff=None,
              fc_thres=None):
    """
    Function scorecard:

    Takes as input a pandas dataframe with one of the columns named
    time and expected to be type numpy datetime64[ns]. Makes a monthly
    scorecard plot modelled on Peter Galbraith's AZMP figures for the
    column field of the dataframe and places it in axes ax.

    Options:

    side_label: Label on the left of the plot

    color_displays: choose from 'anomaly', 'normalized', or 'value' to
    make the color of the boxes reflect anomaly to the mean, value normalized
    by standard deviation of the whole time series, or the actual value.

    value_displays: choose between 'anomaly' and 'value' to set the number to
    be displayed in the box.

    anomaly_color_levels: choose how many color steps to have in both
    directions with respect to zero.

    units_label: units of the average values displayed on the right.

    averages: boolean, display averages by month throughout all years.

    colorbar: boolean, display colorbar if averages is false.

    hide_xlabels: boolean, usefull for stacked plots.
    """
    # Compute card values
    months = df.time.dt.month.sort_values().unique()
    years = df.time.dt.year.unique()
    array = np.zeros((months.size, years.size)) * np.nan
    sub_array = np.zeros((months.size, years.size)) * np.nan
    anomaly_array = np.zeros((months.size, years.size)) * np.nan
    for (year, i) in zip(years, range(years.size)):
        for (month, j) in zip(months, range(months.size)):
            df_month = (df[field]
                        .where(np.logical_and(df.time.dt.year == year,
                                              df.time.dt.month == month))
                           .dropna())
            if n_cutoff is None or df_month.size > n_cutoff:
                array[j, i] = df_month.mean()
            else:
                array[j, i] = np.nan
            # array[j, i] = (df[field]
            #                .where(np.logical_and(df.time.dt.year == year,
            #                                      df.time.dt.month == month))
            #                .dropna()
            #                .mean())

            # Substitute missing values with alternate array
            if np.isnan(array[j, i]) and (not sub_na is None):
                sub_array[j, i] = (df[sub_na]
                                   .where(np.logical_and(df.time.dt.year == year,
                                                         df.time.dt.month == month))
                                   .dropna()
                                   .mean())

    # Compute anomaly
    clim_mean = np.nanmean(array, axis=1)
    clim_std = np.nanstd(array, axis=1)

    for (i, mean, std) in zip(range(array.shape[0]), clim_mean, clim_std):
        anomaly_array[i, :] = (array[i, :] - mean) / std

    # Convert to anomaly if called for
    standard_deviation = df[field].std()
    normalized_array = array / standard_deviation

    # Call style and colormap
    cmap_dir = '/home/jls/Software/anaconda3/lib/python3.7/site-packages/libmx/utils/'
    if color_displays in ['anomaly', 'normalized']:
        cmap = list2cm('%sAnomalyPalette' % cmap_dir, N=int(2 * anomaly_color_levels))
        if anomaly_color_levels == 6:
            vmin, vmax = -3, 3
            cticks = range(-3, 4)
        elif anomaly_color_levels == 5:
            vmin, vmax = -2.5, 2.5
            cticks = range(-2, 3)
        elif anomaly_color_levels == 4:
            vmin, vmax = -2, 2
            cticks = range(-2, 3)
        elif anomaly_color_levels == 3:
            vmin, vmax = -3, 3
            cticks = range(-3, 4)
        elif anomaly_color_levels == 2:
            vmin, vmax = -2, 2
            cticks = range(-2, 3)
    elif color_displays in ['value']:
        if clevels is None:
            cmap = list2cm('%sDivergentPalette' % cmap_dir, N=int(2 * anomaly_color_levels))
            abs_max = np.nanmax(np.abs(array))
            vmax = abs_max - (abs_max % anomaly_color_levels)
            vmin = -vmax
            if anomaly_color_levels % 2 == 0:
                cticks = range(int(vmin), int(vmax+1), 2)
            else:
                cticks = range(int(vmin)+1, int(vmax+1), 2)
        else:
            cmap = list2cm('%sDivergentPalette' % cmap_dir, N=int(clevels.size - 1))
            cticks = clevels
            vmin, vmax = clevels.min(), clevels.max()
    elif color_displays in ['seq_value']:
        cmap = list2cm('%sTsatPalette' % cmap_dir, N=int(anomaly_color_levels))
        vmax = np.ceil(np.nanmax(array)) - (np.ceil(np.nanmax(array)) % 2)
        vmin = np.floor(np.nanmin(array))
        cticks = np.linspace(int(vmin), int(vmax), anomaly_color_levels + 1)[::2]
        # if (vmax - vmin) % 2 == 0:
        #     cticks = np.linspace(int(vmin), int(vmax), anomaly_color_levels)
        # else:
        #     cticks = np.linspace(int(vmin)+1, int(vmax), anomaly_color_levels)


    # Color boxes
    XG = np.hstack((years - 0.5, years[-1] + 0.5))
    YG = np.hstack((months - 0.5, months[-1] + 0.5))
    if color_displays == 'anomaly':
        color_array = anomaly_array
    elif color_displays == 'normalized':
        color_array = normalized_array
    else:
        color_array = array
    caxis = ax.pcolor(XG, YG, color_array,
                      ec='k',
                      linewidth=0.1,
                      fc='gray',
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax)
    ax.set(yticks=np.arange(1, 13),
           yticklabels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'],
           xticks=years,
           xlim=(XG[0], XG[-1]),
           ylim=(YG[0], YG[-1]),
           ylabel='',
           xlabel='',
           facecolor='lightgray')
    if hide_xlabels > 1:
        xticklabels = [year if (year % hide_xlabels) == 0 else '' for year in years]
        ax.plot(years[::hide_xlabels] - 0.5,
                12.5 * np.ones_like(years[::hide_xlabels]),
                marker=3, markersize=6, clip_on=False)
        ax.set(xticklabels=xticklabels)
        ax.tick_params(axis='x', pad=pad_xlabel)
    elif hide_xlabels:
        ax.set(xticklabels=[])
    if not hide_xticks:
        ax.plot(years - 0.5, 12.5 * np.ones_like(years), marker=3, markersize=4, clip_on=False)
    ax.tick_params(which='both', bottom=False, left=False)
    ax.tick_params(axis='x', labelrotation=xlabel_angle)
    ax.set_clip_on(False)
    ax.invert_yaxis()

    # Add side label
    ll = ax.plot([XG[0] - 1, XG[0] - 1], [YG[0], YG[-1]], 'k')
    ll[0].set_clip_on(False)
    tt = ax.text(XG[0] - 1.5, np.mean(YG), side_label,
                 ha='center',
                 va='center',
                 rotation='vertical',
                 fontsize=9)
    tt.set_clip_on(False)

    # Font color change threshold
    if fc_thres is None:
        fc_thres = 0.75 * vmax

    # Add box labels
    if value_displays == 'value':
        text_array = array
    else:
        text_array = anomaly_array
    for i in range(years.size):
        for j in range(months.size):
            if np.isfinite(array[j, i]):
                if color_displays in ['seq_value']:
                    text_color = 'k'
                elif np.abs(color_array[j, i]) >= fc_thres:
                    text_color = 'w'
                else:
                    text_color = 'k'
                ax.text(years[i] + text_offset[0],
                        months[j] + text_offset[1],
                        bfmt % text_array[j, i],
                        ha='center',
                        va='center',
                        fontsize=9,
                        color=text_color)
            elif np.isfinite(sub_array[j, i]):
                ax.text(years[i] + text_offset[0],
                        months[j] + text_offset[1],
                        bfmt % sub_array[j, i],
                        ha='center',
                        va='center',
                        fontsize=9,
                        color='k')

    # Averages on the side
    if averages:
        for (month, i) in zip(months, range(months.size)):
            if np.isfinite(np.nanmean(text_array[i, :])):
                tt = ax.text(years[-1] + 0.75, month + text_offset[1],
                             (sfmt + r' %s $\pm$ ' + sfmt) % (np.nanmean(text_array[i, :]),
                                                     units_label,
                                                     np.nanstd(text_array[i, :])),
                             fontsize=9,
                             va='center',
                             ha='left')
                tt.set_clip_on(False)
        if colorbar:
            _, b, _, h = ax.properties()['position'].bounds
            cbaraxis = ax.figure.add_axes([0.9, b, 0.025, h])
            ax.figure.colorbar(caxis, cax=cbaraxis, label=cbar_label, ticks=cticks).minorticks_off()

    elif colorbar:
        _, b, _, h = ax.properties()['position'].bounds
        cbaraxis = ax.figure.add_axes([0.9, b, 0.025, h])
        ax.figure.colorbar(caxis, cax=cbaraxis, label=cbar_label, ticks=cticks).minorticks_off()

    return array, anomaly_array

# def scorecard(df,
#               field,
#               ax,
#               side_label='',
#               color_displays='anomaly',
#               value_displays='value',
#               anomaly_color_levels=4,
#               units_label='',
#               cbar_label='',
#               averages=False,
#               colorbar=False,
#               hide_xlabels=False,
#               sub_na=None):
#     """
#     Make AZMP-like monthly/yearly scorecard plots.

#     Takes as input a pandas dataframe with one of the columns named
#     time and expected to be type numpy datetime64[ns]. Aesthetics are of
#     the scorecard plot are modelled on Peter Galbraith's AZMP figures.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         Contains the input time series.
#     field : str
#         Name of the column to analyse.
#     ax : pyplot.Axes
#         Where to plot the scorecard
#     color_displays : str ('anomaly', 'normalized' or 'value')
#         Meaning of the color dimension. Make the color of the boxes
#         reflect anomaly to the mean, value normalized by standard
#         deviation of the whole time series, or the actual value.
#     value_displays : str ('anomaly' or 'value' 
#         Meaning of the number to be displayed in the box.
#     anomaly_color_levels : int
#         Number of color steps on both sides of zero.
#     units_label : str
#         Units of the average values displayed on the right.
#     averages : bool
#         Display averages by month throughout all years.
#     colorbar : bool
#         Display colorbar. Requires averages=False.
#     hide_xlabels : bool
#         Hide labels of years on x axis.

#     Returns
#     -------
#     2D array
#         Averaged values.
#     2D array
#         Anomaly values.

#     """
#     # Compute card values
#     months = df.time.dt.month.sort_values().unique()
#     years = df.time.dt.year.unique()
#     array = np.zeros((months.size, years.size)) * np.nan
#     sub_array = np.zeros((months.size, years.size)) * np.nan
#     anomaly_array = np.zeros((months.size, years.size)) * np.nan
#     for (year, i) in zip(years, range(years.size)):
#         for (month, j) in zip(months, range(months.size)):
#             array[j, i] = (df[field]
#                            .where(np.logical_and(df.time.dt.year == year,
#                                                  df.time.dt.month == month))
#                            .dropna()
#                            .mean())

#             # Substitute missing values with alternate array
#             if np.isnan(array[j, i]) and (not sub_na is None):
#                 sub_array[j, i] = (df[sub_na]
#                                    .where(np.logical_and(df.time.dt.year == year,
#                                                          df.time.dt.month == month))
#                                    .dropna()
#                                    .mean())

#     # Compute anomaly
#     clim_mean = np.nanmean(array, axis=1)
#     clim_std = np.nanstd(array, axis=1)

#     for (i, mean, std) in zip(range(array.shape[0]), clim_mean, clim_std):
#         anomaly_array[i, :] = (array[i, :] - mean) / std

#     # Convert to anomaly if called for
#     standard_deviation = df[field].std()
#     normalized_array = array / standard_deviation

#     # Call style and colormap
#     cmap_dir = '/home/jls/Software/anaconda3/lib/python3.7/site-packages/libmx/utils/'
#     if color_displays in ['anomaly', 'normalized']:
#         cmap = list2cm('%sAnomalyPalette' % cmap_dir, N=int(2 * anomaly_color_levels))
#         if anomaly_color_levels == 6:
#             vmin, vmax = -3, 3
#             cticks = range(-3, 4)
#         elif anomaly_color_levels == 5:
#             vmin, vmax = -2.5, 2.5
#             cticks = range(-2, 3)
#         elif anomaly_color_levels == 4:
#             vmin, vmax = -2, 2
#             cticks = range(-2, 3)
#         elif anomaly_color_levels == 3:
#             vmin, vmax = -3, 3
#             cticks = range(-3, 4)
#         elif anomaly_color_levels == 2:
#             vmin, vmax = -2, 2
#             cticks = range(-2, 3)
#     elif color_displays in ['value']:
#         cmap = list2cm('%sDivergentPalette' % cmap_dir, N=int(2 * anomaly_color_levels))
#         abs_max = np.nanmax(np.abs(array))
#         vmax = abs_max - (abs_max % anomaly_color_levels)
#         vmin = -vmax
#         if anomaly_color_levels % 2 == 0:
#             cticks = range(int(vmin), int(vmax+1), 2)
#         else:
#             cticks = range(int(vmin)+1, int(vmax+1), 2)
#     elif color_displays in ['seq_value']:
#         cmap = list2cm('%sTsatPalette' % cmap_dir, N=int(anomaly_color_levels))
#         vmax = np.ceil(np.nanmax(array)) - (np.ceil(np.nanmax(array)) % 2)
#         vmin = np.floor(np.nanmin(array))
#         cticks = np.linspace(int(vmin), int(vmax), anomaly_color_levels + 1)[::2]
#         # if (vmax - vmin) % 2 == 0:
#         #     cticks = np.linspace(int(vmin), int(vmax), anomaly_color_levels)
#         # else:
#         #     cticks = np.linspace(int(vmin)+1, int(vmax), anomaly_color_levels)


#     # Color boxes
#     XG = np.hstack((years - 0.5, years[-1] + 0.5))
#     YG = np.hstack((months - 0.5, months[-1] + 0.5))
#     if color_displays == 'anomaly':
#         color_array = anomaly_array
#     elif color_displays == 'normalized':
#         color_array = normalized_array
#     else:
#         color_array = array
#     caxis = ax.pcolor(XG, YG, color_array,
#                       ec='k',
#                       linewidth=0.1,
#                       fc='gray',
#                       cmap=cmap,
#                       vmin=vmin,
#                       vmax=vmax)
#     ax.set(yticks=np.arange(1, 13),
#            yticklabels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'],
#            xticks=years,
#            xlim=(XG[0], XG[-1]),
#            ylim=(YG[0], YG[-1]),
#            ylabel='',
#            xlabel='',
#            facecolor='lightgray')
#     if hide_xlabels:
#         ax.set(xticklabels=[])
#     ax.tick_params(which='both', bottom=False, left=False)
#     ax.tick_params(axis='x', labelrotation=60)
#     ax.set_clip_on(False)
#     ax.invert_yaxis()

#     # Add side label
#     ll = ax.plot([XG[0] - 1, XG[0] - 1], [YG[0], YG[-1]])
#     ll[0].set_clip_on(False)
#     tt = ax.text(XG[0] - 1.25, np.mean(YG), side_label,
#                  ha='center',
#                  va='center',
#                  rotation='vertical',
#                  fontsize=9)
#     tt.set_clip_on(False)

#     # Add box labels
#     if value_displays == 'value':
#         text_array = array
#     else:
#         text_array = anomaly_array
#     for i in range(years.size):
#         for j in range(months.size):
#             if np.isfinite(array[j, i]):
#                 if color_displays in ['seq_value']:
#                     text_color = 'k'
#                 elif np.abs(color_array[j, i]) >= 0.75 * vmax:
#                     text_color = 'w'
#                 else:
#                     text_color = 'k'
#                 ax.text(years[i], months[j], '%.1f' % text_array[j, i],
#                         ha='center',
#                         va='center',
#                         fontsize=9,
#                         color=text_color)
#             elif np.isfinite(sub_array[j, i]):
#                 ax.text(years[i], months[j], '%.1f' % sub_array[j, i],
#                         ha='center',
#                         va='center',
#                         fontsize=9,
#                         color='k')

#     # Averages on the side
#     if averages:
#         for (month, i) in zip(months, range(months.size)):
#             tt = ax.text(years[-1] + 0.75, month,
#                          r'%.2f%s $\pm$ %.2f' % (np.nanmean(text_array[i, :]),
#                                                  units_label,
#                                                  np.nanstd(text_array[i, :])),
#                          fontsize=9,
#                          va='center',
#                          ha='left')
#             tt.set_clip_on(False)
#         if colorbar:
#             _, b, _, h = ax.properties()['position'].bounds
#             cbaraxis = ax.figure.add_axes([0.9, b, 0.025, h])
#             ax.figure.colorbar(caxis, cax=cbaraxis, label=cbar_label, ticks=cticks).minorticks_off()

#     elif colorbar:
#         _, b, _, h = ax.properties()['position'].bounds
#         cbaraxis = ax.figure.add_axes([0.9, b, 0.025, h])
#         ax.figure.colorbar(caxis, cax=cbaraxis, label=cbar_label, ticks=cticks).minorticks_off()

#     return array, anomaly_array


def scorecard_bottom_monthly(dataset,
                             field,
                             ax,
                             value_displays='anomaly',
                             xlimits=False,
                             xlabels=True,
                             ylabel=False,
                             year_labels=False,
                             y0_adjust=0,
                             minor_ticks=False,
                             major_ticks=False):
    """    
    """
    m_val = (dataset[field]
             .resample(time='1MS')
             .mean())
    m_std = dataset[field].groupby('time.month').std()
    m_clim = dataset[field].groupby('time.month').mean()

    if xlimits:
        m_val = (m_val
                 .where(m_val.time >= xlimits[0], drop=True)
                 .where(m_val.time < xlimits[1], drop=True))

    # Loop over time periods
    for loop_i in range(m_val.size - 1):

        # Calculate anomaly
        month = m_val.time.dt.month.values[loop_i]
        clim = m_clim.where(m_clim.month == month, drop=True).values[0]
        std = m_std.where(m_std.month == month, drop=True).values[0]
        anomaly = ((m_val.values[loop_i] - clim) / std)

        # Select what the numbers mean
        if value_displays == 'diff':
            text_val = m_val.values[loop_i] - clim
        elif value_displays == 'anomaly':
            text_val = anomaly
        else:
            text_val = m_val.values[loop_i]

        # Get time period central time
        ctime = (m_val.time[loop_i] + (m_val.time[loop_i + 1] - m_val.time[loop_i]) * 0.5).values

        # Draw
        ax.axvspan(m_val.time.values[loop_i],
                   m_val.time.values[loop_i + 1],
                   facecolor=anomaly2rgb(anomaly))
        ax.axvline(m_val.time.values[loop_i], color='k')
        ax.text(ctime, 0.5, '%.1f' % text_val, ha='center', va='center', rotation=90)
        if xlabels:
            ax.text(ctime, -0.5, cd.month_abbr[month][0], ha='center', va='center')
        if month == 7 and year_labels:
            this_year = str(m_val.time.dt.year.values[loop_i])
            ax.text(m_val.time.values[loop_i], -1.3, this_year, ha='center', va='center')

    # Draw last time period
    month = m_val.time.dt.month.values[-1]
    clim = m_clim.where(m_clim.month == month, drop=True).values[0]
    std = m_std.where(m_std.month == month, drop=True).values[0]
    anomaly = ((m_val.values[-1] - clim) / std)
    rlim = np.datetime64("%4d-%02d-%02d" % (m_val.time.dt.year.values[-1], month + 1, 1))

    # Select what the numbers mean
    if value_displays == 'diff':
        text_val = m_val.values[-1] - clim
    elif value_displays == 'anomaly':
        text_val = anomaly
    else:
        text_val = m_val.values[-1]

    # Get time period central time
    ctime = (m_val.time[-1] + (rlim - m_val.time[-1]) * 0.5).values
    ax.axvspan(m_val.time.values[-1],
               rlim,
               facecolor=anomaly2rgb(anomaly))
    ax.axvline(m_val.time.values[-1], color='k')
    ax.text(ctime, 0.5, '%.1f' % anomaly, ha='center', va='center', rotation=90, clip_on=True)
    if xlabels:
        ax.text(ctime, -0.5, cd.month_abbr[month][0], ha='center', va='center')

    # Display ylabel if defined (uses inherent booleaness of variables)
    if ylabel:
        ax.set_ylabel(ylabel=ylabel, labelpad=-6)

    # Reposition plot
    x0, y0, width, height = ax.get_position().bounds
    ax.set_position((x0, y0 + y0_adjust, width, height))

    # Manage ticks
    ax.tick_params(which='minor', left=False, bottom=minor_ticks)
    ax.tick_params(which='major', left=False, bottom=major_ticks)
    ax.set(yticklabels=[])


# TS diagram
def ts_diagram(axes,
               t_min=-2,
               t_max=20,
               s_min=32,
               s_max=37,
               lab_curve_exp=1,
               levels=None,
               **clabel_kw):
    """
    Draw density contours as background of a TS plot.

    Parameters
    ----------
    axes : pyplot.Axes
        Axes on which to operate.
    t_min : float
        Minimum temperature to display.
    t_max : float
        Maximum temperature to display.
    s_min : float
        Minimum salinity to display.
    s_max : float
        Maximum salinity to display.
    lab_curve_exp : float
        Move labels towards the top right (>1) or bottom left (<1) corner.
    levels : 1D array
        Density contours to draw. Defaults to (5, 36).
    **clabel_kw : dict
        Arguments passed to `pyplot.clabel`.

    """
    # Parameters
    npts = 100

    if levels is None:
        levels = np.arange(5, 36)

    # Prepare TS grid and calculate density
    sal_grid, temp_grid = np.meshgrid(np.linspace(s_min, s_max, npts),
                                      np.linspace(t_min, t_max, npts))
    sigma_theta_grid = gsw.density.sigma0(sal_grid, temp_grid)

    # Draw density contours
    sigma_ctr = axes.contour(sal_grid,
                             temp_grid,
                             sigma_theta_grid,
                             levels=levels,
                             colors='k',
                             cmap=None)

    # Find label positions
    s_curve = np.linspace(s_min, s_max, npts)
    t_curve = (t_max
               - ((s_curve - s_min) / (s_max - s_min)) ** lab_curve_exp
               * (t_max - t_min))
    r_curve = gsw.density.sigma0(s_curve, t_curve)
    s_locs = interp1d(r_curve, s_curve, bounds_error=False).__call__(levels)
    t_locs = interp1d(r_curve, t_curve, bounds_error=False).__call__(levels)

    # Prepare input to manual label positions
    label_pos = []
    for (s_loc, t_loc) in zip(s_locs, t_locs):
        if np.isfinite(s_loc) and np.isfinite(t_loc):
            label_pos.append((s_loc, t_loc))

    # Draw contour labels
    plt.clabel(sigma_ctr, manual=label_pos, **clabel_kw)

    # Set axis limits to input TS limits
    axes.set(xlim=(s_min, s_max), ylim=(t_min, t_max))


def pd_scorecard(axes,
                 dataframe,
                 units=None,
                 flip_anomaly=None,
                 groups=None,
                 angle_xlabel=60,
                 pad_stat=0.02,
                 pad_xlabel=0.1,
                 pad_ylabel=0.02,
                 pad_group=0.14,
                 pad_group_label=0.02,
                 pad_group_lines = 0.015):
    """
    Generate scorecard plot from pandas dataframe.

    The index of the dataframe is used as the x axis and
    each column becomes one row of the scorecard. Names of
    the dataframe's columns are used as row labels.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Data to use for plot.
    units: list of str
        Ordered units of each column. If set to `dt`, the
        column is processed as `numpy.datetime64`.
    flip_anomaly: list of str
        Names of columns for which to flip the anomaly colormap.
    groups: dict
        Used to group row labels with a line and label. Values are
        lists of size 2 containing the indices of the rows to group
        starting from the bottom. Keys are the labels to place next
        to each line.
    angle_xlabel: float
        Rotate labels of x axis by this amount (deg).
    pad_stat: float
        Space between the scorecard and averages.
    pad_xlabel: float
        Space between the scorecard and x labels.
    pad_ylabel: float
        Space between the scorecard and y labels.
    pad_group: float
        Space between the scorecard and group lines.
    pad_group_label: float
        Space between the group line and its label.
    pad_group_lines: float
        Space between the group lines.

    Returns
    -------
    matplotlib.Axes:
        Axes on which the scorecard is drawn

    """
    # Defaults
    if units is None:
        units = ['' for _ in dataframe.keys()]
    if flip_anomaly is None:
        flip_anomaly = []

    # Get dataframe dimensions
    n_cols, n_rows = dataframe.shape

    # Find grid edges and centers
    x_centers = np.linspace(0, 1, n_cols)
    y_centers = np.linspace(0, 1, n_rows)
    x_edges = binc2edge(x_centers)
    y_edges = binc2edge(y_centers)

    # Set position of averages
    x_stat = x_edges.max() + pad_stat

    # Loop over rows
    for y_l, y_c, y_r, name_, unit_ in zip(y_edges[:-1], y_centers, y_edges[1:], dataframe.keys(), units):

        # Check if this is a date row
        if unit_ == 'dt':
            mean_, mean_label = doy_mean(dataframe[name_])
            std_ = doy_std(dataframe[name_])
        else:
            # Calculate row mean and std
            mean_ = dataframe[name_].mean()
            std_ = dataframe[name_].std()

        # If requirested, flip anomaly colormap
        if name_ in flip_anomaly:
            flip_ = -1
        else:
            flip_ = 1

        # Write statistic on the scorecard right
        if unit_ == 'dt':
            stat_ = r'%s $\pm$ %.0f days' % (mean_label, std_)
        elif unit_:
            stat_ = r'%.0f %s $\pm$ %.0f' % (mean_, unit_, std_)
        else:
            stat_ = r'%.0f $\pm$ %.0f' % (mean_, std_)
        axes.text(x_stat, y_c, stat_, ha='left', va='center')

        # Label row on the scorecard left
        axes.text(x_edges.min() - pad_ylabel, y_c, '%s' % name_, ha='right', va='center')

        # Loop over columns
        for x_l, x_c, x_r, index_ in zip(x_edges[:-1], x_centers, x_edges[1:], dataframe.index):

            # Get value and color
            if unit_ == 'dt':
                if isinstance(dataframe.loc[index_, name_], pd.Timestamp):
                    value_ = date_abb(dataframe.loc[index_, name_])
                    doy_ = dataframe.loc[index_, name_].dayofyear
                else:
                    value_ = np.nan
                    doy_ = np.nan

                # If value is later than October subtract a year for anomaly calculation
                if doy_ > 275:
                    doy_ -= 365

                anomaly_ = flip_ * (doy_ - mean_) / std_
            else:
                value_ = dataframe.loc[index_, name_]
                anomaly_ = (value_ - mean_) / std_
            color_ = anomaly2rgb(anomaly_)

            # Draw grid and color
            x_rect, y_rect = [x_l, x_r, x_r, x_l, x_l], [y_l, y_l, y_r, y_r, y_l]

            # Switch text color over large anomalies for readability
            if abs(anomaly_) >= 1.5:
                font_color = 'w'
            else:
                font_color ='k'

            # Write date value
            if unit_ == 'dt' and isinstance(dataframe.loc[index_, name_], pd.Timestamp):
                axes.text(x_c, y_c, '%s' % value_, ha='center', va='center', fontsize=7)
                axes.fill(x_rect, y_rect, facecolor=color_, edgecolor='k', linewidth=0.5)

            # Or finite numeric value
            elif np.isfinite(value_):
                axes.text(x_c, y_c, '%.0f' % value_, ha='center', va='center')
                axes.fill(x_rect, y_rect, facecolor=color_, edgecolor='k', linewidth=0.5)

            # Or grey out the box if missing
            else:
                axes.fill(x_rect, y_rect, facecolor=color_, edgecolor=None, zorder=-5)

    # Label x axis
    for x_c, index_ in zip(x_centers, dataframe.index):
        axes.text(x_c, y_edges.min() - pad_xlabel, '%s' % index_, ha='center', va='top', rotation=angle_xlabel)

    # Label row groups if requested
    if isinstance(groups, dict):
        for title_, rows_ in groups.items():

            # Set row line positions
            y_pts = np.array([y_edges[rows_[0]] + pad_group_lines, y_edges[rows_[-1] + 1] - pad_group_lines])
            x_pts = (x_edges.min() - pad_group) * np.ones_like(y_pts)

            # Draw row group line
            axes.plot(x_pts, y_pts, 'k', clip_on=False)

            # Draw row group label
            axes.text(x_pts.mean() - pad_group_label, y_pts.mean(), title_, ha='center', va='center', rotation=90)

    # Axes and tick parameters
    axes.set(ylim=(y_edges.min(), y_edges.max()), xlim=(x_edges.min(), x_edges.max()))
    axes.set(xticklabels=[], yticklabels=[])
    axes.tick_params(which='both', bottom=False, left=False)


def wa_map(axes,
           resolution='intermediate',
           crs='PlateCarree',
           landcolor='oldlace',
           watercolor='lightgray',
           extent=[-70, -40, 35, 54],
           tick_x_maj=np.arange(-70, -35, 5),
           tick_y_maj=np.arange(35, 58, 2),
           tick_x_min=np.arange(-71, -39, 1),
           tick_y_min=np.arange(35, 56, 1)):
    """
    Plot West Atlantic map in axes.

    Parameters
    ----------
    axes: matplotlib.Axes
        Axes to replace with GSL map.
    resolution: str
        GSHHS features resolution. Can be one of `coarse`, `low`,
        `intermediate`, `high` or `full`.
    crs: str
        Name of cartopy.crs projection class to use.
    landcolor: str
        Continent and islands filled with this color.
    watercolor: str
        Rivers and lakes filled with this color.
    extent: 4-list
        Longitude and latitude limits of map.
    tick_x_maj, tick_y_maj: 1D array
        Longitude and latidude of major ticks and labels.
    tick_x_min, tick_y_min: 1D array
        Longitude and latitude of minor ticks.

    Returns
    -------
    geoax
        GeoAxes instance of map panel.

    """
    # Replace normal Axes with geoAxes
    geoax, _ = cp_proj(axes, crs)
    geoax.set_extent(extent)

    # Most features are fine in GSHHS
    coast = cfeature.GSHHSFeature(scale=resolution,
                                  levels=[1],
                                  facecolor=landcolor,
                                  edgecolor='k',
                                  linewidth=0.25)
    lakes = cfeature.GSHHSFeature(scale=resolution,
                                  levels=[2],
                                  facecolor=watercolor,
                                  edgecolor=watercolor)
    islands = cfeature.GSHHSFeature(scale=resolution,
                                    levels=[3],
                                    facecolor=landcolor,
                                    edgecolor=landcolor)
    ponds = cfeature.GSHHSFeature(scale=resolution,
                                  levels=[4],
                                  facecolor=watercolor,
                                  edgecolor=watercolor)

    # Add GSHHS features
    geoax.add_feature(coast)
    geoax.add_feature(lakes)
    geoax.add_feature(islands)
    geoax.add_feature(ponds)

    # Rivers are not in GSHHS
    RIVERS = pd.read_csv('/data/atlas/gmt/gulff-rivers-all.lon_lat',
                         names=['lon', 'lat'],
                         sep=r'\s+',
                         na_values=-99.000)
    geoax.plot(-RIVERS.lon.values,
               RIVERS.lat.values,
               color='lightskyblue',
               transform=ccrs.PlateCarree())

    # Ticks
    cp_ticks(geoax, tick_x_maj, tick_y_maj, labels=True, size=3)
    cp_ticks(geoax, tick_x_min, tick_y_min, labels=False, size=1.5)

    return geoax

    
def xr_plot_pm_patch(ds, xcoord, base, interval, ax, color='lightskyblue'):
    """
    Draw a patch of +- interval width around time series.
 
    Parameters
    ----------
    ds : xarray.Dataset
        Contains the data to plot.
    xcoord : str
        Name of time series coordinate.
    base : str
        Name of data to use as vertical center of the patch.
    interval : str
        Name of data to use as vertical half height of the patch.
    ax : pyplot.Axes
        Axes on which to draw.
    color : str
        Name of the patch's color.

    """
    ds = xr_abs(ds, interval)
    BASE = np.hstack((ds[base].values, ds[base].values[::-1]))
    INTER = np.hstack((ds[interval].values, -ds[interval].values[::-1]))
    YVEC = BASE + INTER
    XVEC = np.hstack((ds[xcoord].values, ds[xcoord].values[::-1]))
    ax.fill(XVEC, YVEC, color=color)
