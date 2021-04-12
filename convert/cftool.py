import xarray as xr
import pandas as pd
import numpy as np
import warnings
from datetime import datetime


__all__ = ['pd_dataframe_2_ragged',
           'xr_cf_attributes',
           'xr_concatenate_ragged',
           'xr_cf_del_cast',
           'xr_cf_get_cast']


# Common oceanographic pairings of long_name, standard_name, units (vars)
var_dict = {'CT': ('Conservative temperature',
                   'sea_water_conservative_temperature',
                   'Celsius'),
            'temperature': ('In situ temperature',
                            'sea_water_temperature',
                            'Celsius'),
            'temp': ('In situ temperature',
                     'sea_water_temperature',
                     'Celsius'),
            'SA': ('Absolute salinity',
                   'sea_water_absolute_salinity',
                   'g kg-1'),
            'salinity': ('Practical salinity',
                         'sea_water_practival_salinity',
                         '1'),
            'sal': ('Practical salinity',
                    'sea_water_practival_salinity',
                    '1'),
            'density': ('In situ density',
                        'sea_water_density',
                        'kg m-3'),
            'rho': ('In situ density',
                    'sea_water_density',
                    'kg m-3'),
            'ST': ('Potential density',
                   'sea_water_sigma_theta',
                   'kg m-3'),
            'sigma_theta': ('Potential density',
                            'sea_water_sigma_theta',
                            'kg m-3'),
            'pden': ('Potential density',
                     'sea_water_sigma_theta',
                     'kg m-3'),
            'rho': ('In situ density',
                    'sea_water_density',
                    'kg m-3'),
            'u': ('Velocity east',
                  'eastward_sea_water_velocity',
                  'm s-1'),
            'v': ('Velocity north',
                  'northward_sea_water_velocity',
                  'm s-1'),
            'w': ('Velocity up',
                  'upward_sea_water_velocity',
                  'm s-1')
            }

# Common oceanographic pairings of long_name, standard_name, units (coords)
crd_dict = {'longitude': ('Longitude',
                          'longitude',
                          'degrees_east',
                          'X'),
            'lon': ('Longitude',
                    'longitude',
                    'degrees_east',
                    'X'),
            'latitude': ('Latitude',
                         'latitude',
                         'degrees_nort',
                         'Y'),
            'lat': ('Latitude',
                    'latitude',
                    'degrees_north',
                    'Y'),
            'z': ('Depth',
                  'depth',
                  'm',
                  'Z',
                  'down'),
            'depth': ('Depth',
                      'depth',
                      'm',
                      'Z',
                      'down'),
            'time': ('Time',
                     'time',
                     '',
                     'T')
            }


def pd_dataframe_2_ragged(dataframe,
                          groupby,
                          var_names_,
                          coord_names_,
                          coord_var_names_,
                          var_types_=None,
                          coord_types_=None,
                          coord_var_types_=None):
    """
    Convert profile data(x, y, z, t) to ragged format.

    Ragged format is as described by CF conventions. Default
    types of coordinates and variables are `numpy.float64`.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Columns are a list of obs and coords.
    groupby: list of str
        Input col names to use for grouping sets of obs.
    var_names_: list of str
        Input col names to extract.
    coord_names : list of str
        Output coord names in order matching `groupby`.
    coord_var_names_: list of str
        Input col names of coords same size as obs.
    var_types_: list of objects
        Variable types with order matching `var_names_`.
    coord_types_: list of objects
        Coord types with order matching `coord_names_`.
    coord_var_types_: list of objects
        Coord variable types with order matching `coord_var_names_`.

    Returns
    -------
    xr.Dataset
        Input dataframe in ragged format.

    """
    # Manage default types
    if var_types_ is None:
        var_types_ = [np.float64 for _ in var_names_]
    if coord_types_ is None:
        coord_types_ = [np.float64 for _ in coord_names_]
    if coord_var_types_ is None:
        coord_var_types_ = [np.float64 for _ in coord_var_names_]

    # Select T_max by profile
    gp = dataframe.groupby(groupby)
    index_ = gp.first().index

    # Set up master dimension
    profiles_ = np.arange(index_.size)

    # Set up coordinates
    coords_ = dict()
    for c_, t_ in zip(coord_names_, coord_types_):
        coords_[c_] = ('profiles', np.empty(index_.size, dtype=t_))

    # Set up variables
    vars_ = dict()
    for v_, t_ in zip(var_names_, var_types_):
        vars_[v_] = (['obs'], np.empty(
            dataframe[v_].values.size, dtype=np.float64))
        vars_['%s_row_size' % v_] = (
            ['profiles'], np.zeros(index_.size, dtype=np.int64))

    # Set up coordinate variables
    coord_vars_ = dict()
    if coord_var_names_:
        for c_, t_ in zip(coord_var_names_, coord_var_types_):
            coord_vars_[c_] = ('obs', np.empty(
                dataframe[v_].values.size, dtype=np.float64))
            vars_['%s_row_size' % c_] = (
                ['profiles'], np.zeros(index_.size, dtype=np.int64))

    # Loop over groups
    for i_, name_ in enumerate(index_):

        # Find individual profile
        profile = gp.get_group(name_).copy(deep=True)

        # Reintegrate coordinates to this profile
        for j_, c_ in enumerate(coords_):
            coords_[c_][1][i_] = name_[j_]

        # Get downcast size
        rs_ = profile.shape[0]

        # Store coordinate variable data and size
        for c_ in coord_var_names_:
            row_start_ = np.nansum(vars_['%s_row_size' % c_][1])
            coord_vars_[c_][1][row_start_:row_start_ +
                               rs_] = profile[c_].values
            vars_['%s_row_size' % c_][1][i_] = rs_

        # Store variable data and size
        for v_ in var_names_:
            row_start_ = np.nansum(vars_['%s_row_size' % v_][1])
            vars_[v_][1][row_start_:row_start_ + rs_] = profile[v_].values
            vars_['%s_row_size' % v_][1][i_] = rs_

    # Form xarray dataset
    dataset = xr.Dataset(vars_, coords={'profiles': profiles_,
                                        **coords_,
                                        **coord_vars_})

    return dataset


def xr_cf_attributes(dataset, crd_attr=None, var_attr=None, user_globals=None):
    """
    Label xarray dataset with CF compliant attributes.

    A few common oceanographic variable names are pre-mapped
    to CF long and standard names and units. Likewise for typical
    geographical coordinales in longitude, latitude, depth and time.

    Parameters
    ----------
    dataset: xr.Dataset
        Label vars and coords of this object.
    crd_attr: dict
        Name--tuple mapping for coords (long, std, units, axis, positive). If
        time is of type `np.datetime64` do not define the units field. Set as
        empty string. `The positive` field only applies to the depth coord. The
        other mappings should be 4-tuples.
    var_attr: dict
        Name--tuple mapping for vars (long, std, units).
    user_globals: dict
        Name--str mapping for global dataset attributes.

    Returns
    -------
    xr.Dataset
        Labelled xarray dataset.

    """

    # Combine presets and user variable attributes
    if isinstance(var_attr, dict):
        var_dict_ = {**var_dict, **var_attr}
    else:
        var_dict_ = var_dict

    # Only rely on preset coord name to CF conversions
    if isinstance(crd_attr, dict):
        crd_dict_ = {**crd_dict, **crd_attr}
    else:
        crd_dict_ = crd_dict

    # Preset global attributes
    preset_globals = {'Conventions': 'CF-1.8',
                      'history': 'Created: %s' % datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                      'institution': 'Maurice Lamontagne Institute',
                      'contact': 'Jean-Luc.Shaw@dfo-mpo.gc.ca'}

    # Specified global attributes
    if user_globals is None:
        global_attrs = preset_globals
    else:
        global_attrs = {**preset_globals, **user_globals}

    # Manage coordinates
    for c_ in dataset.coords:

        # Coordinate name has preset or user definition
        if c_ in crd_dict_.keys():
            dataset[c_].attrs['long_name'] = crd_dict_[c_][0]
            dataset[c_].attrs['standard_name'] = crd_dict_[c_][1]
            dataset[c_].attrs['axis'] = crd_dict_[c_][3]

            # Do not encode time units
            if crd_dict_[c_][2]:
                dataset[c_].attrs['units'] = crd_dict_[c_][2]

            # Positive attribute for depth coordinate
            if len(crd_dict_[c_]) == 5:
                dataset[c_].attrs['positive'] = crd_dict_[c_][4]

        # This is the sample coordinate
        elif 'profile' in c_:
            dataset[c_].attrs['cf_role'] = 'profile_ID'

        # No match found for coordinate name
        else:
            msg = 'No attributes found for coord: %s' % c_
            warnings.warn(msg)

    # Data variable attributes
    for n_ in dataset.data_vars:

        # Variable name has a preset or user definition
        if n_ in var_dict_.keys():
            dataset[n_].attrs['long_name'] = var_dict_[n_][0]
            dataset[n_].attrs['standard_name'] = var_dict_[n_][1]
            dataset[n_].attrs['units'] = var_dict_[n_][2]

        # This is a sample dimension variable
        elif 'row_size' in n_:
            dataset[n_].attrs['long_name'] = 'Number of obs for profile'
            dataset[n_].attrs['sample_dimension'] = 'obs'

        # No match found for variable name
        else:
            msg = 'No attributes found for var: %s' % n_
            warnings.warn(msg)

    # Data ranges
    for n_ in {**dataset.data_vars, **dataset.coords}:

        # Get max values
        min_val_ = dataset[n_].min().values
        max_val_ = dataset[n_].max().values

        # Convert to str if datetime
        """
        Without this step, saving to netCDF is not possible.
        """
        if isinstance(min_val_, np.datetime64):
            min_val_ = str(min_val_)
        if isinstance(max_val_, np.datetime64):
            max_val_ = str(max_val_)

        # Set attributes
        dataset[n_].attrs['data_min'] = min_val_
        dataset[n_].attrs['data_max'] = max_val_

    # Global attributes
    dataset.attrs = global_attrs

    return dataset


def xr_concatenate_ragged(file_list, concat_dims):
    """
    Concatenate ragged array netCDF files (CF).

    Parameters
    ----------
    file_list : list of str or list of xr.Dataset
        Path and name of files to concatenate. Also accepts lists of datasets.
    concat_dims : list of str
        Dimensions along which to independently concatenate.

    Returns
    -------
    xr.Dataset
        Merged netCDF files.

    """
    # Init dictionnary to contain size of each concat dimension
    dim_size = dict()
    for cd_ in concat_dims:
        dim_size[cd_] = 0

    # Check each input file for dimension sizes
    for f_ in file_list:
        # load file
        if isinstance(f_, xr.Dataset):
            ds = f_
        else:
            ds = xr.open_dataset(f_)

        # Determine size of each concat dimension
        for d_ in concat_dims:
            dim_size[d_] += ds[d_].size

    # Pre-allocate
    vars_ = dict()
    crds_ = dict()

    # --- variables
    for v_ in ds.data_vars:
        dim_, = ds[v_].dims
        vars_[v_] = ([dim_], np.empty(dim_size[dim_], dtype=ds[v_].dtype))

    # --- coordinates
    for c_ in ds.coords:
        dim_, = ds[c_].dims
        crds_[c_] = ([dim_], np.empty(dim_size[dim_], dtype=ds[c_].dtype))

    # Init indexing dictionnaries
    dim_start = dict()
    row_width = dict()

    # --- start index along each concat dim
    for cd_ in concat_dims:
        dim_start[cd_] = 0

    # --- row size along each concat dim
    for cd_ in concat_dims:
        row_width[cd_] = 0

    # Loop over input files
    for f_ in file_list:

        # Load file
        if isinstance(f_, xr.Dataset):
            ds = f_
        else:
            ds = xr.open_dataset(f_)

        # Loop over coordinates
        for c_ in crds_.keys():

            # Determine alon which dim
            d_, = ds[c_].dims

            # Determine start and stop indices
            row_start = dim_start[d_]
            row_width[d_] = ds[c_].size
            row_stop = row_start + row_width[d_]

            # Insert new values
            if c_ in concat_dims:

                # Offset by row start to keep unique value sample dims
                offset = row_start

            else:
                offset = 0
            crds_[c_][1][row_start:row_stop] = ds[c_].values + offset

        # Loop over variables
        for v_ in vars_.keys():

            # Determine along which dim
            d_, = ds[v_].dims

            # Determine start and stop indices
            row_start = dim_start[d_]
            row_width[d_] = ds[v_].size
            row_stop = row_start + row_width[d_]

            # Insert new values
            vars_[v_][1][row_start:row_stop] = ds[v_].values

        # Update row start indices
        for d_ in dim_start.keys():
            dim_start[d_] = dim_start[d_] + row_width[d_]

    # Form dataset
    dataset = xr.Dataset(vars_, coords=crds_)

    # Add CF attributes
    dataset = xr_cf_attributes(dataset)

    return dataset


def xr_cf_del_cast(dataset,
                   delete,
                   prf_dim='profiles',
                   obs_dim=['obs'],
                   depth_name='z'):
    """
    Delete specific casts from a CF ragged array profiles dataset.

    Parameters
    ----------
    dataset: xarray.Dataset
        Ragged array dataset.
    delete: 1D array
        Number ids of the profiles to delete.
    prf_dim: str
        Name of profiles coordinate in `dataset`.
    obs_dim: iterable of str
        Names of observations coordinate in `dataset`.
    depth_name: str
        Name of depth coordinate in `dataset`.

    Returns
    -------
    xarray.Dataset:
        Data structure containing the undeleted casts.

    Note
    ----

       This function is not yet fully generalized to multiple
       different observations dimensions, though this feature
       is an anticipated necessity.

    """
    sel_dict = dict()

    # Make a boolean vector of the casts to keep
    prf_bool = np.ones_like(dataset[prf_dim], dtype=bool)
    prf_bool[delete] = False
    sel_dict[prf_dim] = prf_bool

    # Make a boolean vector of the observations to keep
    for od_ in obs_dim:
        sel_dict[od_] = np.ones_like(dataset[od_], dtype=bool)

    # Loop over each cast to delete
    for n_ in delete:

        # --- Loop over observations dimensions
        for od_ in obs_dim:
            
            # Get index values of the cast to delete in this dimension
            c_strt = int(dataset['%s_row_size' % depth_name][:n_].values.sum())
            c_stop = c_strt + int(dataset['%s_row_size' % depth_name][n_].values)

            # Set values at these indices to false in the boolean vector
            sel_dict[od_][c_strt:c_stop] = False

    # Make a call to the sel method
    return dataset.sel(**sel_dict)


def xr_cf_get_cast(rag_arr, n, depth_name='z'):
    """
    Get a single cast from a CF ragged array profiles dataset.

    Parameters
    ----------
    rag_arr: str or xarray.Dataset
        Path and name to nc file or ragged dataset.
    n: int
        Number id of the profile to get.
    depth_name: str
        Name of depth coordinate in `rag_arr`.

    Returns
    -------
    xarray.Dataset:
        Data structure containing one cast only.

    """
    # Read netcdf or pass xarray
    if isinstance(rag_arr, str):
        dset = xr.open_dataset(rag_arr)
    elif isinstance(rag_arr, xr.Dataset):
        dset = rag_arr
    else:
        raise TypeError('rag_arr is not string or xarray Dataset')

    # Variables along obs dimension
    var_names = [v_
                 for v_ in rag_arr.data_vars
                 if 'obs' in rag_arr[v_].dims]

    # Get cast depth information

    c_strt = int(dset['%s_row_size' % depth_name][:n].values.sum())
    c_stop = c_strt + int(dset['%s_row_size' % depth_name][n].values)
    depth = dset[depth_name][c_strt:c_stop].values

    # Add requested variables
    cast = xr.Dataset(coords={depth_name: depth}, attrs=dset.attrs)

    # Loop over requested variables
    for variable in var_names:
        # Check this variable is not empty for this cast
        if dset['%s_row_size' % variable][n] > 0:
            # Get variable index values
            c_strt = int(dset['%s_row_size' % variable][:n].values.sum())
            c_stop = c_strt + int(dset['%s_row_size' % variable][n].values)

            # Assign
            cast[variable] = (depth_name, dset[variable][c_strt:c_stop])

    # Variables along obs dimension
    var_names = [v_
                 for v_ in rag_arr.data_vars
                 if 'profiles' in rag_arr[v_].dims]

    # Loop over requested variables
    for variable in var_names:
        # Assign
        cast.attrs[variable] = dset[variable].values[n]

    # Loop over requested variables
    for c_ in dset.coords:
        if c_ != 'z':
            # Assign
            cast = cast.assign_coords({c_: dset[c_].values[n]})

    return cast
