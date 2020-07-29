"""
Python functions to access/manipulate world ocean database ragged arrays.
"""
import xarray as xr
from numpy import isfinite, nan, ones_like
import gsw

__all__ = ['wod_cast_n']

def wod_cast_n(rag_arr,
               n,
               var_names=['Temperature', 'Salinity'],
               anc_names=None,
               do_qc=True,
               do_teos10=True):
    """
    Get an individual cast from WOD ragged array.

    If do_qc is true, data are filtered keeping only those with
    quality flags of 0 or 1. Refused data are returned as NaN. Some
    profiles do not have quality flags. There are three possible cases
    and here are the meaning of the quality flags they produce.

    Profile quality flag missing
    -> Profile flag = -1
    -> value flags = -1

    Profile quality flag exists but is not accepted
    -> Profile flag = passed from original file
    -> Value flags = -2

    Profile quality flag exists and is accepted, but value flags are missing
    -> Profile flag = passed from original file
    -> Value flags = -3

    Parameters
    ----------
    rag_arr: xarray.Dataset or straight
        Path to a WOD netCDF file containing a CTD ragged array or named
        it is read into.
    n: int
        Cast number to return as xarray Dataset.
    var_names: list of str
        Names of the variables to extract. Defaults to ['Temperature', 'Salinity'].
    anc_names: list of str
        Names of the ancillary data variables to extract. Defaults to None.
    do_qc: bool
        If True keep only data with WOD quality flags 0 or 1. Defaults to True.
        This also passes the WOD quality flags to the child cast.
    do_teos10: bool
        If True calculate CT, SA and sigma0 using the gsw package, implementing
        TEOS10. Defaults to True.

    Returns
    -------
    xarray.Dataset
        The isolated nth cast of the ragged array.
    """
    # Read netcdf or pass xarray
    if isinstance(rag_arr, str):
        dset = xr.open_dataset(rag_arr)
    elif isinstance(rag_arr, xr.Dataset):
        dset = rag_arr
    else:
        raise TypeError('rag_arr is not string or xarray Dataset')

    # Replace VAR_row_size NaN values with 0
    for variable in var_names + ['z']:
        field = '%s_row_size' % variable
        dset[field] = dset[field].where(isfinite(dset[field]), 0)

    # Get cast depth information
    depth_name = 'z'
    time_name = 'time'          # For ease if ever necessary
    lon_name = 'lon'            # to make more flexible
    lat_name = 'lat'

    c_strt = int(dset['%s_row_size' % depth_name][:n].values.sum())
    c_stop = c_strt + int(dset['%s_row_size' % depth_name][n].values)
    depth = dset[depth_name][c_strt:c_stop].values

    # Add requested variables
    cast = xr.Dataset(coords={depth_name: depth}, attrs=dset.attrs)

    # Default of TEOS10 switches
    has_temp = False
    has_sal = False
    
    # Loop over requested variables
    for variable in var_names:
        # Check this variable is not empty for this cast
        if dset['%s_row_size' % variable][n] > 0:
            # Get variable index values
            c_strt = int(dset['%s_row_size' % variable][:n].values.sum())
            c_stop = c_strt + int(dset['%s_row_size' % variable][n].values)

            # Assign
            cast[variable] = (depth_name, dset[variable][c_strt:c_stop])

            # Switches for TEOS10
            if 'Temperature' == variable:
                has_temp = True
            if 'Salinity' == variable:
                has_sal = True

            # Do quality control (keeps flags 0 and 1)
            if do_qc:
                # Logic switches
                pf_exists = '%s_WODprofileflag' % variable in dset.data_vars.keys()
                vl_exists = '%s_WODflag' % variable in dset.data_vars.keys()

                # Check quality flag exists for cast
                if pf_exists:
                    # Pass value of cast quality flag
                    cast.attrs['%s_WODprofileflag' % variable] = dset['%s_WODprofileflag' % variable].values[n]

                    # Check quality flag is accepted for cast
                    if cast.attrs['%s_WODprofileflag' % variable] in [0, 1]:
                        # Check existence of observation flags
                        if vl_exists:
                            cast['%s_WODflag' % variable] = (depth_name,
                                                             dset['%s_WODflag' %
                                                                  variable][c_strt:c_stop])
                            condition = ((cast['%s_WODflag' % variable] == 0) |
                                         (cast['%s_WODflag' % variable] == 1))
                            cast[variable] = cast[variable].where(condition)

                        # Value flags do not exist
                        else:
                            print('Warning: No flags for variable %s' % variable)
                            cast[variable] *= nan
                            cast['%s_WODflag' % variable] = (depth_name,
                                                             -3 * ones_like(depth, dtype=int))

                    # Profile quality flag is not accepted
                    else:
                        # Pass cast quality flag
                        cast[variable] *= nan
                        cast['%s_WODflag' % variable] = (depth_name,
                                                         -2 * ones_like(depth, dtype=int))

                # Profile quality flag does not exist
                else:
                    cast.attrs['%s_WODprofileflag' % variable] = -1
                    cast[variable] *= nan
                    cast['%s_WODflag' % variable] = (depth_name,
                                                     -1 * ones_like(depth, dtype=int))

        # Variable exists but profile is empty
        else:
            print('Warning: No data for variable %s' % variable)

    # Convert other coordinates to attributes
    for coord in [time_name, lon_name, lat_name]:
        cast.attrs[coord] = dset[coord].values[n]

    # Gather ancillary data if requested
    if anc_names is not None:
        for anc in anc_names:
            cast.attrs[anc] = dset[anc].values[n]

    # Use TEOS10 to get CT, SA and sigma0 if requested
    if has_temp and has_sal:
        has_rfields = (all(cast.attrs['%s_WODprofileflag' % field]
                           in [0, 1]  # Has the required fields
                           for field in ['Temperature', 'Salinity']))
    else:
        has_rfields = False
    has_rattrs = (all(attr in cast.attrs  # Has the required attributes
                      for attr in ['lon', 'lat']))

    if do_teos10 and has_rattrs and has_rfields:
        # Remove NaN values
        cast = cast.where((isfinite(cast.Temperature)) &
                          (isfinite(cast.Salinity)), drop=True)
        try:
            cast['Sea_Pres'] = (depth_name,
                                gsw.p_from_z(-cast.z.values,
                                             cast.lat))
            cast['SA'] = (depth_name,
                          gsw.SA_from_SP(cast.Salinity.values,
                                         cast.Sea_Pres.values,
                                         cast.lon,
                                         cast.lat))
            cast['CT'] = (depth_name,
                          gsw.CT_from_t(cast.SA.values,
                                        cast.Temperature.values,
                                        cast.Sea_Pres.values))
            cast['RHO'] = (depth_name,
                           gsw.rho(cast.SA.values,
                                   cast.CT.values,
                                   cast.Sea_Pres.values))
            cast['SIGMA_THETA'] = (depth_name,
                                    gsw.density.sigma0(cast.SA.values,
                                                       cast.CT.values))
        except:
            print('Could not do TEOS 10 conversions')

    # Gather ancillary data if requested
    if anc_names is not None:
        for anc in anc_names:
            cast.attrs[anc] = dset[anc].values[n]

    # Output result
    return cast
