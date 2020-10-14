from pycurrents.adcp.rdiraw import Multiread
import xarray as xr
import numpy as np
import mxtoolbox.process as ps
from scipy.stats import circmean
# from libmx.utils.time import yb2dt


__all__ = ['adcp_init',
           'adcp_qc',
           'load_rdi_binary']


def adcp_init(Nz, Nt, Nb):
    """
    Return empty xarray shell in standardized format.

    Parameters
    ----------
    Nz : int
        Number of vertical bins.
    Nt : int
        Number of time steps.
    Nb : int
        Number of ADCP beams.

    Returns
    -------
    xarray.Dataset
        An empty dataset ready for data input.

    """
    ds = xr.Dataset(
        data_vars = {'u': (['z', 'time'], np.nan * np.ones((Nz, Nt))),
                     'v': (['z', 'time'], np.nan * np.ones((Nz, Nt))),
                     'w': (['z', 'time'], np.nan * np.ones((Nz, Nt))),
                     'e': (['z', 'time'], np.nan * np.ones((Nz, Nt))),
                     'flags': (['z', 'time'], np.zeros((Nz, Nt))),
                     'temp': (['time'], np.nan * np.ones(Nt)),
                     'dep': (['time'], np.nan * np.ones(Nt)),
                     'Roll': (['time'], np.nan * np.ones(Nt)),
                     'pitch': (['time'], np.nan * np.ones(Nt)),
                     'heading': (['time'], np.nan * np.ones(Nt)),
                     'u_bt': (['time'], np.nan * np.ones(Nt)),
                     'v_bt': (['time'], np.nan * np.ones(Nt)),
                     'w_bt': (['time'], np.nan * np.ones(Nt)),
                     'e_bt': (['time'], np.nan * np.ones(Nt)),
                     'corr': (['z', 'time'], np.nan * np.ones((Nz, Nt))),
                     'amp': (['z', 'time'], np.nan * np.ones((Nz, Nt))),
                     'pg': (['z', 'time'], np.nan * np.ones((Nz, Nt))),
                     'pg_bt': (['Nbeam', 'time'], np.nan * np.ones((Nb, Nt))),
                     'corr_bt': (['Nbeam', 'time'], np.nan * np.ones((Nb, Nt))),
                     'range_bt': (['Nbeam', 'time'], np.nan * np.ones((Nb, Nt)))},
        coords = {'z' : np.arange(Nz),
                  'time': np.empty(Nt, dtype='datetime64[ns]'),
                  'Nbeam': np.arange(0, Nb)},
        attrs = {'freqHz': '',
                 'angle': '',
                 'binSize': '',
                 'looking': '',
                 'serial': ''})

    return ds


def adcp_qc(di,
            amp_th=30,
            pg_th=90,
            corr_th=0.6,
            roll_th=20,
            pitch_th=20,
            vel_th=5,
            iv=None,
            gpsfile=None,
            sl=None,
            depth=None,
            R=None,
            C=None):
    """
    Perform ADCP quality control.

    Parameters
    ----------
    di : xarray.Dataset
        ADCP dataset formatted as done by adcp_init.
    amp_th : float
        Require more than this amplitude values.
    pg_th : float
        Require more than this percentage of good 4-beam transformations.
    corr_th : float
        Require more than this beam correlation value.
    roll_th : float
        Require roll values be smaller than this value (degrees).
    pitch_th : float
        Require pitch values be smaller than this value (degrees).
    iv : None
        Unknown.
    gpsfile : str
        GPS dataset formatted as by gps_init.
    sl : str
        Use fixed depth or bottom track range to remove side lobe
        contamination. Set to either `dep` or `bt`.
    depth : float
        Fixed depth used for removing side lobe contamination.
    R : float
        Horizontally rotate velocity after motion correction by this value.
    C : float
        Horizontally rotate velocity before motion correction by this value.


    Note
    ----

       Quality control flags follow those used by DAISS for ADCP
       data at Maurice-Lamontagne Institute. Meaning of the flagging
       values is the following.

       * 0: no quality control
       * 1: datum seems good
       * 3: datum seems questionable
       * 4: datum seems bad
       * 9: datum is missing

       Data are marked as questionable if they fail only the 4beam
       transformation test. If they fail the 4beam test and any other
       non-critical tests they are marked as bad. Data likely to be
       biased from sidelobe interference are also marked as bad.

    """
    # Work on copy
    ds = di.copy(deep=True)

    # Check for gps file if required
    if iv=='gps':
        try:
            gps = xr.open_dataset(gpsfile).interp(time=ds.time)
            ds['lon'] = gps.lon
            ds['lat'] = gps.lat
        except:
            raise NameError("GPS file not found...!")

    # Acoustics conditions
    corr_condition = np.abs( ds.corr ) < corr_th
    pg_condition = np.abs( ds.pg ) < pg_th
    amp_condition = np.abs( ds.amp ) < amp_th

    # Motion conditions
    roll_mean = circmean(ds.Roll.values, low=-180, high=180)
    roll_condition = np.abs(ps.circular_distance(ds.Roll.values, roll_mean, units='deg')) > roll_th
    pitch_mean = circmean(ds.pitch.values, low=-180, high=180)
    pitch_condition = np.abs(ps.circular_distance(ds.pitch.values, pitch_mean, units='deg')) > pitch_th
    motion_condition = roll_condition & pitch_condition

    # Outiler conditions
    horizontal_velocity = np.sqrt(ds.u ** 2 + ds.v ** 2)
    # velocity_condition = (np.greater(abs(ds.u.values), vel_th, where=np.isfinite(ds.u.values)) |
    #                       np.greater(abs(ds.v.values), vel_th, where=np.isfinite(ds.v.values)))
    velocity_condition = np.greater(horizontal_velocity.values, vel_th, where=np.isfinite(ds.u.values))
    bottom_track_condition = (np.greater(abs(ds.u_bt.values), vel_th, where=np.isfinite(ds.u_bt.values)) |
                              np.greater(abs(ds.v_bt.values), vel_th, where=np.isfinite(ds.v_bt.values)))

    # Missing condition
    missing_condition = (~np.isfinite(ds.u) | ~np.isfinite(ds.v) | ~np.isfinite(ds.w)).values

    # Boolean summary of non-critical tests
    ncrit_condition = corr_condition | amp_condition | motion_condition | velocity_condition

    # Remove side lob influence according to a fixed depths (e.g. Moorings)
    if sl == 'dep':
        # Dowward looking
        if ds.attrs['looking'] == 'down':
            if depth != None:
                sidelobe_condition = (ds.z > ds.dep.values.mean()
                                      * (1 - np.cos(np.pi * ds.attrs['angle'] / 180))
                                      + depth * np.cos(np.pi * ds.attrs['angle'] / 180))
            else:
                raise Warning("Can not correct for side lobes, depth not provided.")

        # Upward looking
        elif ds.attrs['looking']=='up':
            sidelobe_condition = ds.z < ds.dep.values.mean()*(1-np.cos(np.pi*ds.attrs['angle']/180))

        # Orientation unknown 
        else:
            raise Warning("Can not correct for side lobes, looking attribute not set.")
            sidelobe_condition = np.ones(ds.z.values.size, dtype='bool')

    # Remove side lobe influence ping by ping
    elif sl=='bt':
        print("Per ping side lobe correction using bottom track range not yet implemented. Doing nothing.")

    # Do not perform side lobe removal
    else:
        sidelobe_condition = np.zeros_like(ds.u.values, dtype='bool')

    # Apply condition to bottom track velocities
    for field in ['u_bt', 'v_bt', 'w_bt']:
        ds[field] = ds[field].where( bottom_track_condition )

    # Determine quality flags
    ds['flags'] = 1
    ds['flags'] = xr.where(pg_condition, 3, ds.flags)
    ds['flags'] = xr.where(pg_condition & ncrit_condition, 4, ds.flags)
    ds['flags'] = xr.where(sidelobe_condition, 4, ds.flags)
    ds['flags'] = xr.where(missing_condition, 9, ds.flags)

    # First optional rotation to correct compass misalignment
    if C not in [None, 0]:
        u, v = ps.rotate_frame(ds.u.values, ds.v.values, C, units='deg')
        ds['u'], ds['v'] = (('z', 'time'), u), (('z', 'time'), v)
        u_bt, v_bt = ps.rotate_frame(ds.u_bt.values, ds.v_bt.values, C, units='deg')
        ds['u_bt'], ds['v_bt'] = (('z', 'time'), u_bt), (('z', 'time'), v_bt)

    # Correct for platform motion
    for field in ['u', 'v', 'w', 'e']:

        # No bottom track for error velocity
        if field in ['u', 'v', 'w']:
            # Bottom track correction in 3D
            if iv=='bt':
                ds[field] -= ds['%s_bt' % field].values

            # GPS velocity correction in 2D
            elif iv=='gps' and (field in ['u', 'v']):
                ds[field] += np.tile(gps[field].where(np.isfinite(gps.lon.values), 0), (ds.z.size, 1))

    # Second optional rotation to place in chosen reference frame
    if R not in [None, 0]:
        u, v = ps.rotate_frame(ds.u.values, ds.v.values, R, units='deg')
        ds['u'], ds['v'] = (('z', 'time'), u), (('z', 'time'), v)

    return ds


def load_rdi_binary(files,
                     adcptype,
                     force_dw=False,
                     force_up=False,
                     mindep=0,
                     selected=None,
                     clip=0,
                     t_offset=0):
    """
    Read Teledyne RDI binary ADCP data to xarray.

    Parameters
    ----------
    files : str or list of str
        File(s) to read in.
    adcptype : str
        Sensor type passed to pycurrents.Multiread. ('wh', 'os')
    force_dw : bool
        Process as downward looking ADCP.
    force_up : bool
        Process as upward looking ADCP.
    mindep : float
        Require instrument depth be greater that this value in meters.

    Returns
    -------
    xarray.Dataset
        ADCP data.
    
    """
    # Read
    data = Multiread(files, adcptype).read()

    # Check coordinate system
    if not data.trans['coordsystem'] == 'earth':
        raise Warning("Beams 1-4 are in %s coordinate system"
                      % data.trans['coordsystem'])

    # Get data set size and configuration
    if selected is None:
        selected = range(0, len(data.dday) - clip)
    t = ps.dayofyear2dt(data.dday[selected] + t_offset, data.yearbase)

    # Configure depth of bin centers
    if force_dw or force_up:
        if force_dw:
            # downwards processing
            selected = data.XducerDepth > mindep
            z = data.dep + np.nanmean(data.XducerDepth[selected])  # depth of the bins
            looking = 'down'
        else:
            # upwards processing
            selected = data.XducerDepth > mindep
            qc_kw['depth'] = np.nanmean(data.XducerDepth[selected])
            z = qc_kw['depth'] - data.dep  # depth of the bins
            looking = 'up'
    elif not data.sysconfig['up']:
        # downwards processing
        selected = data.XducerDepth > mindep
        z = data.dep + np.nanmean(data.XducerDepth[selected])  # depth of the bins
        looking = 'down'
    elif data.sysconfig['up']:
        # upwards processing
        selected = data.XducerDepth > mindep
        qc_kw['depth'] = np.nanmean(data.XducerDepth[selected])
        z = qc_kw['depth'] - data.dep  # depth of the bins
        looking = 'up'
    else:
        raise ValueError('Could not determine ADCP orientation!')

    # Init xarray
    t = t[selected]
    ds = adcp_init(z.size, t.size, 4)

    # Set up xarray
    ds['u'].values = np.asarray(data.vel1[selected][:].T)
    ds['v'].values = np.asarray(data.vel2[selected][:].T)
    ds['w'].values = np.asarray(data.vel3[selected][:].T)
    ds['e'].values = np.asarray(data.vel4[selected][:].T)
    ds['corr'].values = np.asarray(np.mean(data.cor, axis=2)[selected][:].T)
    ds['amp'].values = np.asarray(np.mean(data.amp, axis=2)[selected][:].T)
    ds['pg'].values = np.float64(np.asarray(data.pg4[selected][:].T))
    ds['dep'].values = np.asarray(data.XducerDepth[selected])
    ds['heading'].values = np.asarray(data.heading[selected])
    ds['Roll'].values = np.asarray(data.roll[selected])
    ds['pitch'].values = np.asarray(data.pitch[selected])

    # Set up coordinates
    ds = ds.assign_coords(time=t, z=z, Nbeam=np.arange(4))

    # Bottom track data if it exists
    if not (data.bt_vel.data == 0).all():
        ds['u_bt'].values = data.bt_vel.data[selected, 0]
        ds['v_bt'].values = data.bt_vel.data[selected, 1]
        ds['w_bt'].values = data.bt_vel.data[selected, 2]
        ds['e_bt'].values = data.bt_vel.data[selected, 3]
        ds['range_bt'].values = data.bt_depth.data[selected, :].T

    # Attributes
    ds.attrs['angle'] = data.sysconfig['angle']
    ds.attrs['freqHz'] = data.sysconfig['kHz'] * 1000
    ds.attrs['binSize'] = data.CellSize
    ds.attrs['looking'] = looking
    ds.attrs['serial'] = str(data.FL['Inst_SN'])

    # Flip z if upward looking
    if ds.attrs['looking'] == 'up':
        ds = ds.sortby('z')

    # Sort according to time
    ds = ds.sortby('time')

    return ds
