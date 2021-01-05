"""
Inspect ADCP data processed by adcp2nc.

Command line interface description can be shown by entering,

.. code::

   $ adcp_plots -h

For this utility to be available at the command line, add a
file called :code:`adcp_plots` on your shell path, for example
at :code:`/usr/local/bin/` containing the following lines,

.. code::

   #!/path/to/bash
   /path/to/python /path/to/mxtoolbox/plots/adcp_plots.py "$@"

"""
# import mxtoolbox.plot as pt
# import mxtoolbox.process as ps
from mxtoolbox.process.convert import theta2hd, bine2center
from mxtoolbox.process.analyses import sm_pca
from mxtoolbox.process.math_ import rotate_frame
from mxtoolbox.plot.mplutils import colorbar, bshow
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import os
import warnings


# Axis sizes
gs_kw_sq = {'left': 0.25, 'right': 0.9, 'bottom': 0.15, 'top': 0.9}
gs_kw_fd = {'bottom': 0.2, 'left': 0.1, 'right': 0.9}
gs_kw_pf = {'bottom': 0.15, 'left': 0.15, 'top': 0.9, 'wspace': 0.2}
gs_kw_ts = {'left': 0.1, 'bottom': 0.1, 'right': 0.95, 'hspace': 0.2}


def velocity_scatter_figure(dataset,
                            u,
                            v,
                            title,
                            z_average=False,
                            eigenvecs=True):
    """
    Draw ADCP velocites as scatter plot
    """
    # Velocity
    _, ax = plt.subplots(figsize=(4, 3.5), gridspec_kw=gs_kw_sq)
    if z_average:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_ = dataset.mean(dim='z')

        mean_.plot.scatter(x=u, y=v, c='k', ax=ax)
        u_vec = mean_[u].values.flatten()
        v_vec = mean_[v].values.flatten()
    else:
        dataset.plot.scatter(x=u, y=v, c='k', ax=ax)
        u_vec = dataset[u].values.flatten()
        v_vec = dataset[v].values.flatten()

    # Annotate with data range
    ax.text(-0.9, 0.9, '%.2f < u < %.2f' %
            (np.nanmin(u_vec), np.nanmax(u_vec)), fontsize=8)
    ax.text(-0.9, 0.8, '%.2f < v < %.2f' %
            (np.nanmin(v_vec), np.nanmax(v_vec)), fontsize=8)

    # Draw velocity norm circles
    theta = np.arange(0, 2 * np.pi + np.pi / 64, np.pi / 64)
    for r_ in [0.25, 0.5, 0.75]:
        ax.plot(r_ * np.cos(theta), r_ * np.sin(theta), 'r')

    # Draw eigenvectors
    if eigenvecs:
        if u_vec.size > 1000000:
            raise ValueError(
                'Field %s too large for PCA with size: %.0f' % (u, u_vec.size))

        u_1, v_1, u_2, v_2, l_1, l_2, theta = sm_pca(u_vec, v_vec)
        l_1, l_2 = l_1 / l_1, l_2 / l_1

        quiver_kw = dict(angles='xy', scale_units='xy', scale=1)
        ax.quiver(0, 0, l_1 * u_1, l_1 * v_1, color='b', **quiver_kw)
        ax.quiver(0, 0, l_2 * u_2, l_2 * v_2, color='r', **quiver_kw)

        # Write angle of 1st eigenvec and pcent variance explained
        variance = 100 * l_1 / (l_1 + l_2)
        heading = theta2hd(theta)
        text_ = r'%.0f%% variance at %.0f$^\circ$' % (variance, heading)
        ax.text(-0.9, -0.9, text_, fontsize=8)

    else:
        heading = None

    # Set axes parameters
    ax.set(xlim=(-1, 1),
           ylim=(-1, 1),
           xlabel='%s [%s]' % (dataset[u].long_name, dataset[u].units),
           ylabel='%s [%s]' % (dataset[v].long_name, dataset[v].units))
    ax.set_aspect('equal')
    ax.set_title(title)

    return heading


def velocity_field_figure(dataset, strt, wdth, cres=0.2):
    """
    Draw ADCP velocity fields
    """
    stop = strt + wdth
    levels = np.arange(-1, 1 + cres, cres)
    _, ax = plt.subplots(2, figsize=(12, 7), sharex=True, gridspec_kw=gs_kw_fd)
    ctr = dataset.u[:, strt:stop].plot.pcolormesh(ax=ax[0],
                                                  levels=levels,
                                                  cmap='RdBu',
                                                  add_colorbar=False)
    ax[0].set(facecolor='k', xlabel='', ylim=(0, 120))
    cbar, _ = colorbar(ax[0],
                       ctr,
                       label='%s [%s]' % (
                           dataset.u.long_name, dataset.u.units),
                       cbar_size=0.01,
                       pad_size=0.02,
                       ticks=levels[::2])
    cbar.minorticks_off()

    ctr = dataset.v[:, strt:stop].plot.pcolormesh(ax=ax[1],
                                                  levels=levels,
                                                  cmap='RdBu',
                                                  add_colorbar=False)
    ax[1].set(facecolor='k', xlabel='', ylim=(0, 120))
    cbar, _ = colorbar(ax[1],
                       ctr,
                       label='%s [%s]' % (
                           dataset.v.long_name, dataset.v.units),
                       cbar_size=0.01,
                       pad_size=0.02,
                       ticks=levels[::2])
    cbar.minorticks_off()

    ax[0].invert_yaxis()
    ax[0].tick_params(which='minor', top=False, bottom=False)
    ax[1].invert_yaxis()
    ax[1].tick_params(which='minor', top=False, bottom=False)


def velocity_depth_profiles(phases, dataset, rotation=None):
    """
    Draw tidal phase velocity profiles against depth
    """
    _, axes = plt.subplots(1,
                           2,
                           figsize=(5, 4),
                           gridspec_kw=gs_kw_pf,
                           sharey=True)

    # Copies of input arrays
    dataset_ = dataset.copy()
    phases_ = [p_.copy() for p_ in phases]

    # Rotate if requested
    if rotation is not None:
        u_, v_ = rotate_frame(
            dataset.u.values, dataset.v.values, rotation, units='deg')
        dataset_['u'].values = u_
        dataset_['v'].values = v_
        for p_ in phases_:
            u_, v_ = rotate_frame(
                p_.u.values, p_.v.values, rotation, units='deg')
            p_['u'].values = u_
            p_['v'].values = v_

        # Set xlabels
        u_label = r'Velocity to %.0f$^\circ$ [%s]' % (
            theta2hd(rotation), dataset['u'].units)
        v_label = r'Velocity to %.0f$^\circ$ [%s]' % (
            theta2hd(rotation + 90), dataset['v'].units)
    else:
        # Set xlabels
        u_label = '%s [%s]' % (dataset_['u'].long_name, dataset_['u'].units)
        v_label = '%s [%s]' % (dataset_['v'].long_name, dataset_['v'].units)

    # Line labels and colors
    colors_ = ['k', 'r', 'k', 'b']
    labels_ = ['H', 'E', 'L', 'F']

    # Helper function to avoid code duplication
    def _draw_profiles(axes, field, xlabel):

        # Draw tide phases
        for c_, l_, p_ in zip(colors_, labels_, phases_):
            axes.plot(p_[field], p_.z, c_)
            axes.text(p_[field].dropna(dim='z').isel(z=0),
                      -5,
                      l_,
                      color=c_,
                      ha='center')

        # Draw net average
        axes.plot(dataset_[field].mean(dim='time'), dataset_.z, 'k', lw=1)
        axes.axvline(0, color='lightgray', zorder=-5)

        # Label x axis
        axes.set(xlabel=xlabel)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _draw_profiles(axes[0], 'u', u_label)
        _draw_profiles(axes[1], 'v', v_label)

    axes[0].set(ylim=(0, 120),
                ylabel='%s [%s]' % (dataset_.z.long_name, dataset_.z.units))
    axes[0].invert_yaxis()


if __name__ == '__main__':
    # Handle input arguments
    parser = ap.ArgumentParser(usage=__doc__)

    # identifies files
    parser.add_argument('adcpnc',
                        metavar='adcp netcdf file(s)',
                        help='Expression identifying adcp netcdf file(s).',
                        nargs='+')
    parser.add_argument('-l', '--tidenc',
                        metavar='tide file',
                        help='Expression identifying tide netcdf file.')
    parser.add_argument('-t', '--timeseries',
                        help='''Draw time series of u,v [and more vars]
                        at level.''',
                        metavar='level var_1 var_2 ...',
                        nargs='+')
    parser.add_argument('-s', '--scatters',
                        action='store_true',
                        help='Draw u,v scatter plots.')
    parser.add_argument('-f', '--fields',
                        action='store_true',
                        help='Draw u,v field plots.')
    parser.add_argument('-p', '--profiles',
                        action='store_true',
                        help='Draw u,v profile plots.')
    parser.add_argument('-i', '--show',
                        action='store_true',
                        help='''Show plots interactively as they are
                        drawn (requires X11).''')
    args = parser.parse_args()

    if args.timeseries is None:
        timeseries = []
    else:
        timeseries = args.timeseries

    # Defaul values
    if args.scatters or args.timeseries or args.fields or args.profiles:
        plot_all_ = False
    else:
        plot_all_ = True

        # Timeseries defaults must be specified for plot all
        if args.timeseries:
            timeseries = args.timeseries
        else:
            timeseries = [0]    # u,v plot of the first bin

    # Style of plots to use
    plt.style.use('~/.config/matplotlib/gri_xy.mplstyle')

    # Open adcp dataset
    if len(args.adcpnc) == 1:
        adcp = xr.open_dataset(*args.adcpnc)

        # Paths and names
        a_path_ = os.path.abspath(*args.adcpnc)
        path_ = os.path.dirname(a_path_)
        name_ = os.path.basename(a_path_).split('_')[0]

    # Multiple files
    else:
        adcp = xr.open_mfdataset(
            args.adcpnc, combine='by_coords', concat_dim='time')

        # Paths and names
        a_path_ = os.path.abspath(args.adcpnc[0])
        path_ = os.getcwd()
        name_ = os.path.basename(a_path_).split('_')[0]

    # Apply quality control
    adcp['u_raw'] = adcp.u.copy()
    adcp['v_raw'] = adcp.v.copy()
    adcp['w_raw'] = adcp.w.copy()
    adcp['u'] = adcp.u.where(adcp.flags < 2)
    adcp['v'] = adcp.v.where(adcp.flags < 2)
    adcp['w'] = adcp.w.where(adcp.flags < 2)

    # Tide related processing
    if args.tidenc:
        # Open tide dataset
        tide = xr.open_dataset(args.tidenc)

        # Time after high tide vector
        adcp['aht'] = tide.aht.interp(time=adcp.time)

        # Tide average bin sizes
        edges = np.arange(-0.5, 13.5, 1)
        cntrs = bine2center(edges)

        # Tide averaging
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            # Average
            adcp_t_a = adcp.groupby_bins('aht',
                                         bins=edges,
                                         labels=cntrs,
                                         restore_coord_dims=True).mean()

            # Make 0 and 12 h AHT identical
            for var_ in adcp_t_a.data_vars:
                if var_ in ['u', 'v', 'w']:
                    mean_ = (adcp_t_a[var_]
                             .isel(aht_bins=[0, -1])
                             .mean(dim='aht_bins')
                             .values)
                    adcp_t_a.isel(aht_bins=0)[var_].values = mean_
                    adcp_t_a.isel(aht_bins=0)[var_].values = mean_

                    adcp_t_a[var_] = adcp_t_a[var_].transpose('z', 'aht_bins')

                # Keep variable attributes for labelling
                adcp_t_a[var_].attrs = adcp[var_].attrs

        # Divide dataset by tide phase
        def _phase(x): return x.mean(dim='aht_bins').interpolate_na(dim='z')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            adcp_high = _phase(adcp_t_a.isel(aht_bins=[-1, 0, 1]))
            adcp_ebb = _phase(adcp_t_a.isel(aht_bins=[2, 3, 4]))
            adcp_low = _phase(adcp_t_a.isel(aht_bins=[5, 6, 7]))
            adcp_flood = _phase(adcp_t_a.isel(aht_bins=[8, 9, 10]))

    # Velocity scatter plots
    if plot_all_ or args.scatters:
        # All velocities scattered
        velocity_scatter_figure(
            adcp, 'u', 'v', 'Horizontal velocities', eigenvecs=False)

        # Save and show
        bshow('%s/%s_u_v_scatter.png' % (path_, name_), show=args.show)

        # Depth averaged velocities scattered
        velocity_scatter_figure(
            adcp, 'u', 'v', 'Depth averaged', z_average=True)

        # Save and show
        bshow('%s/%s_u_v_depth_averaged_scatter.png' %
              (path_, name_), show=args.show)

        # Platform velocities scattered
        velocity_scatter_figure(adcp, 'uship', 'vship', 'Platform velocities')

        # Save and show
        bshow('%s/%s_uship_vship_scatter.png' %
              (path_, name_), show=args.show)

        # Tide averaged velocities scattered
        if args.tidenc:
            azimuth = velocity_scatter_figure(
                adcp_t_a, 'u', 'v', 'Tide averaged')

            # Keep azimuth in two right quadrants
            if azimuth > 180:
                azimuth -= 180

            # Save azimuth for comparison to other years
            np.savetxt('%s/tide.averaged.azimuth' %
                       path_, [azimuth], fmt='%.2f')

            # Save and show
            bshow('%s/%s_u_v_tide_averaged_scatter.png' %
                  (path_, name_), show=args.show)

    # Velocity field plots
    if plot_all_ or args.fields:

        # Suppresses a deprecation warning independed of input
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Velocity fiels (a few weeks)
            strt = int(adcp.time.size / 2)
            velocity_field_figure(adcp, strt, 3000)
            bshow('%s/%s_u_v_fields_month.png' % (path_,
                                                  name_), show=args.show)

            # Velocity fields (a few days)
            velocity_field_figure(adcp, strt, 200)

            # Save and show
            bshow('%s/%s_u_v_fields_week.png' %
                  (path_, name_), show=args.show)

    # Profile agains depth plots
    if plot_all_ or args.profiles:
        if args.tidenc:
            # Tidal phase profiles (east and north)
            velocity_depth_profiles(
                (adcp_high, adcp_ebb, adcp_low, adcp_flood), adcp)
            bshow('%s/%s_u_v_profiles.png' % (path_, name_), show=args.show)

            # Tidal phase profiles (principal direction of variation)
            _, _, _, _, _, _, theta_t = sm_pca(adcp_t_a.u.values.flatten(),
                                               adcp_t_a.v.values.flatten())

            # Keep angle in right side quadrants
            if theta_t < 0:
                theta_t += 180

            # Draw profiles
            velocity_depth_profiles((adcp_high,
                                     adcp_ebb,
                                     adcp_low,
                                     adcp_flood), adcp, rotation=theta_t)

            # Save and show
            bshow('%s/%s_u_v_profiles_rotated.png' %
                  (path_, name_), show=args.show)

        # Warn user if this plot can not be drawn
        else:
            warnings.warn(
                "Plotting profiles requires a specified sea level file (-l).")

    # Timeseries plots
    if plot_all_ or (len(timeseries) >= 1):

        # Manage variable input formats
        if len(timeseries) > 1:
            z_level = int(timeseries[0])
            fields = ['u', 'v', *timeseries[1:]]
        else:
            z_level = int(timeseries[0])
            fields = ['u', 'v']

        # Initialize plot
        _, ax = plt.subplots(len(fields), figsize=(
            8, 4 + len(fields)), sharex=True, gridspec_kw=gs_kw_ts)

        # Get percentage of values dropped
        n_u_cln = adcp.isel(z=z_level).u.dropna(dim='time').size
        n_u_raw = adcp.isel(z=z_level).u_raw.dropna(dim='time').size
        pc_u = 100 * n_u_cln / n_u_raw
        ax[0].text(0.015,
                   0.9,
                   '%.1f%% data good' %
                   pc_u, transform=ax[0].transAxes)

        # Draw raw velocities
        adcp.isel(z=z_level).plot.scatter(x='time',
                                          y='u_raw',
                                          c='lightblue',
                                          s=0.5,
                                          ax=ax[0],
                                          zorder=-5)
        adcp.isel(z=z_level).plot.scatter(x='time',
                                          y='v_raw',
                                          c='lightblue',
                                          s=0.5, ax=ax[1],
                                          zorder=-5)

        # Add additionnal timeseries if requested
        for n_, (f_, a_) in enumerate(zip(fields, ax)):

            # 2D variable
            if 'z' in adcp[f_].dims:
                adcp.isel(z=z_level).plot.scatter(x='time',
                                                  y=f_,
                                                  c='k',
                                                  s=0.5,
                                                  ax=a_,
                                                  zorder=10)
                adcp[f_].isel(z=z_level).plot(ax=a_, color='lightgray')

            # 1D variable
            else:
                adcp[f_].plot(ax=a_)

            # Remove title and minor ticks
            if n_ > 1:
                a_.set_title('')
            a_.tick_params(which='minor', bottom=False)
            a_.set(xlabel='')

        # Save and show
        bshow('%s/%s_u_v_timeseries_zlevel_%d.png' % (path_,
                                                      name_,
                                                      z_level),
              show=args.show)
