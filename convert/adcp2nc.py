"""
Read and quality control binary ADCP data.

Uses the CODAS library to read Teledyne RDI ADCP
files (.000, .ENX, .ENS, .LTA, etc), arranges data
into an xarray Dataset, performs QC, and saves to
netCDF in the current directory.

Rowetech files are also accepted but reading is not handled
by CODAS, and processing is much slower. Forcing processing
as upward/downward looking is not yet implemented for this
type of input. Neither are the minimum required depth or
time offset options.

It is best to inlcude as much information as possible
through the option flags as this will improve the quality
control.

This module is meant to be called from command line. A full
list of options can be displayed by calling,

.. code::

   $ adcp2nc -h

For this utility to be available at the command line, add a
file called :code:`adcp2nc` on your shell path, for example
at :code:`/usr/local/bin/` containing the following lines,

.. code::

   #!/path/to/bash
   /path/to/python /path/to/mxtoolbox/convert/adcp2nc.py "$@"

See Also
--------

   * mxtoolbox.read.adcp
   * mxtoolbox.read.rtitools
   * pycurrents.adcp.rdiraw.Multiread

"""
from mxtoolbox.read.adcp import *
from mxtoolbox.read.rtitools import load_rtb_binary
import numpy as np
import os
import argparse


__all__ = list()


# Command line interface
if __name__ == '__main__':
    # # Not ideal but until pandas/xarray sort eachother out..
    # warnings.simplefilter('ignore')

    # Handle input arguments
    parser = argparse.ArgumentParser(usage=__doc__)

    # identifies files
    parser.add_argument('files',
                        metavar='1 - files',
                        help='Expression identifying adcp files',
                        nargs='+')
    # adcp type
    parser.add_argument('adcptype',
                        metavar='2 - sonar',
                        help='''String designating type of adcp. This
                        is fed to CODAS Multiread. Must be one
                        of `wh`, `os`, or `bb`''')
    # deployment nickname
    parser.add_argument('name',
                        metavar='3 - name',
                        help='''Mission, mooring, or station name to
                        prepend to the output file name.''')

    parser.add_argument('-a', '--amp-thres',
                        metavar='',
                        type=float,
                        help='Amplitude threshold (0-255). Defaults to 0.')
    parser.add_argument('-c', '--corr-thres',
                        metavar='',
                        type=float,
                        help='Correlation threshold (0-255). Defaults to 64.')
    parser.add_argument('-d', '--depth',
                        metavar='',
                        type=float,
                        help='Water depth (scalar)')
    parser.add_argument('-D', '--force-dw',
                        action='store_true',
                        help='Force downward looking processing.')
    parser.add_argument('-g', '--gps-file',
                        metavar='',
                        help='GPS netcdf file path and name.')
    parser.add_argument('-i', '--include-temp',
                        action='store_true',
                        help='''Include temperature. Otherwise, default behavior
                        is to save it to a difference netcdf file.''')
    parser.add_argument('-k', '--clip',
                        metavar='',
                        type=int,
                        help='Number of ensembles to clip from the end of the dataset.')
    parser.add_argument('-m', '--motion-correction',
                        metavar='',
                        help='''Motion correction mode. Defaults to no motion correction.
                        If given 'bt', will use bottom track data to correct for
                        instrument motion. If given 'gps', will use gps data to
                        correct for instrument motion but fail if no gps file is
                        provided. See vkdat2vknetcdf.py for gps file details.''')
    parser.add_argument('-o', '--t-offset',
                        metavar='',
                        type=int,
                        help='''Offset by which to correct time in hours. May for
                        example be used to move dataset from UTC to local
                        time.''')
    parser.add_argument('-p', '--pg-thres',
                        metavar='',
                        type=float,
                        help='Percentage of 4 beam threshold (0-100). Defaults to 80.')
    parser.add_argument('-P', '--pitch-thres',
                        metavar='',
                        type=float,
                        help='Pitch threshold (0-180). Defaults to 20.')
    parser.add_argument('-q', '--no-qc',
                        action='store_true',
                        help='Omit quality control.')
    parser.add_argument('-r', '--roll-thres',
                        metavar='',
                        type=float,
                        help='Roll threshold (0-180). Defaults to 20.')
    parser.add_argument('-R', '--rot-ang',
                        type=float,
                        metavar='',
                        help='''Anti-clockwise positive angle in degrees by which to
                        rotate the frame of reference.
                        (-360 to 360).''')
    parser.add_argument('-s', '--sl-mode',
                        metavar='',
                        help='''Side lobe rejection mode. Default is None. If given `bt`,
                        will use range to boundary from bottom track data. If given
                        `dep` will use a constant depth but fail if depth is not
                        provided for downward looking data. If data is upward looking,
                        the average depth of the instrument is used as distance to
                        boundary.''')
    parser.add_argument('-T', '--mindep',
                        metavar='',
                        type=float,
                        help='''Minimum instrument depth threshold. Keep only data for
                        which the instrument was below the provided depth in
                        meters.''')
    parser.add_argument('-U', '--force-up',
                        action='store_true',
                        help='Force upward looking processing.')
    parser.add_argument('-v', '--velocity-thres',
                        type=float,
                        metavar='',
                        help='''Reject velocities whose absolute value is greather than
                        this value in m/s.''')
    parser.add_argument('-z', '--zgrid',
                        metavar='',
                        help='''Interpolate depths to grid defined by the single column
                        depth in meters file given in argument.''')
    args = parser.parse_args()


    # Option switches
    if args.force_up and args.force_dw:
        raise ValueError("Cannot force downwards AND upwards processing")

    # Brand independent quality control defaults
    qc_defaults = dict(iv=None,
                       gpsfile=None,
                       pitch_th=20,
                       roll_th=20,
                       R=0,
                       sl=None,
                       depth=None,
                       pg_th=80)

    # Brand dependent quality control defaults
    rti_qc_defaults = dict(amp_th=20,
                           corr_th=0.6)
    rdi_qc_defaults = dict(amp_th=0,
                           corr_th=64)

    # Quality control options
    user_qc_kw = {}
    if args.motion_correction:
        user_qc_kw['iv'] = args.motion_correction
    if args.gps_file:
        user_qc_kw['gpsfile'] = args.gps_file
    if args.corr_thres:
        user_qc_kw['corr_th'] = args.corr_thres
    if args.pg_thres:
        user_qc_kw['pg_th'] = args.pg_thres
    if args.amp_thres:
        user_qc_kw['amp_th'] = args.amp_thres
    if args.pitch_thres:
        user_qc_kw['pitch_th'] = args.pitch_thres
    if args.roll_thres:
        user_qc_kw['roll_th'] = args.roll_thres
    if args.roll_thres:
        user_qc_kw['vel_th'] = args.velocity_thres
    if args.rot_ang:
        user_qc_kw['R'] = np.pi * args.rot_ang / 180
    if args.sl_mode:
        user_qc_kw['sl'] = args.sl_mode
    if args.depth:
        user_qc_kw['depth'] = args.depth

    # Other options
    t_offset = args.t_offset / 24 if args.t_offset else 0
    clip = args.clip or 0
    mindep = args.mindep or 0
    gridf = args.zgrid          # expected default is None anyway
    qc = not args.no_qc
    selected = None             # not used for now.

    # Get output path
    path = (os.path.dirname(args.files[0])
            if isinstance(args.files, list)
            else os.path.dirname(args.files))
    if not path == '':
        path = path + '/'  # avoids trying to write at /

    # Read ADCP data
    if args.adcptype in ['wh', 'bb', 'os']:
        ds = load_rdi_binary(args.files,
                             args.adcptype,
                             force_dw=args.force_dw,
                             force_up=args.force_up,
                             mindep=mindep)
        brand = 'RDI'
        qc_kw = {**qc_defaults, **rdi_qc_defaults, **user_qc_kw}
    elif args.adcptype == 'sw':
        ds = load_rtb_binary(args.files)
        brand = 'RTI'
        qc_kw = {**qc_defaults, **rti_qc_defaults, **user_qc_kw}
    else:
        raise ValueError('Sonar type %s not recognized' % args.adcptype)

    # Quality control
    if qc:
        ds = adcp_qc(ds, **qc_kw)

    # Interpolate to z grid
    if gridf:
        # Find maximum depth where 10% of data is good
        Ng = np.asarray([np.isfinite(ds.u.values[ii, :]).sum()
                         for ii in range(ds.z.size)])
        z_max = ds.z.values[np.argmin(np.abs(Ng.max() * 0.1 - Ng))]

        # Bin to z grid
        z_grid = np.loadtxt(gridf)
        ds = xr_bin(ds, 'z', z_grid)

        # Remove sparse data
        cond = ds.z < z_max if ds.looking == 'down' else ds.z > z_max
        ds.u.values = ds.u.where(cond)
        ds.v.values = ds.v.where(cond)
        ds.w.values = ds.w.where(cond)
        ds.e.values = ds.e.where(cond)

    # Save to netcdf
    strt = str(ds.time.values[0])[:10]
    stop = str(ds.time.values[-1])[:10]

    # Manage temperature data
    if args.include_temp:
        ds['temp'].values = np.asarray([data.temperature[selected]])
    # else:
    #     temp = seabird_init(t)
    #     temp['temp'] = ('time', np.array(data.temperature[selected]))
    #     temp['z'] = ('time', np.array(data.XducerDepth[selected]))
    #     temp.to_netcdf(path+args.name+'_'+strt+'_'+stop+'_RDI_TEMP.nc')

    ds.to_netcdf('%s%s_%s_%s_%s_ADCP.nc' % (path, args.name, strt, stop, brand))
