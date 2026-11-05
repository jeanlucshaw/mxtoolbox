"""
Low level reader of Workhorse ADCP metadata
"""
import csv


def hex_str_n_bytes(fobj, n, bl):
    """
    Read `n` bytes from stream and return as string of `hex`.

    Parameters
    ----------
    fobj: file object
        Byte stream to read.
    n: int
        Number of bytes to read.
    bl: str
        `big` or `little` endian.

    Returns
    -------
    str:
        The hexadecimal message.

    """
    str_ = ''
    for i_ in range(n):
        str_ += '%s ' % format(int.from_bytes(fobj.read(1), bl), 'x')
    return str_


def read_ens_md_info(fobj):
    """
    Read metadata of one ADCP ensemble from byte stream.

    Returns
    -------
    dict:
        Metadata of the ensemble.

    """
    # Header
    hd_id = int.from_bytes(fobj.read(1), 'little')
    if hd_id:
        ens = dict()
        ens['hd_id'] = format(hd_id, 'x') 
        ens['ds_id'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['n_bytes_ens'] = int.from_bytes(fobj.read(2), 'little')
        _ = int.from_bytes(fobj.read(1), 'little')
        ens['n_data_types'] = int.from_bytes(fobj.read(1), 'little')
        for i_ in range(ens['n_data_types']):
            ens['offset_data_%d' % i_] = int.from_bytes(fobj.read(2), 'little')

        # Fixed leader
        ens['fl_id_1'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['fl_id_2'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['cpu_fw_ver'] = int.from_bytes(fobj.read(1), 'little')
        ens['cpu_fw_rev'] = int.from_bytes(fobj.read(1), 'little')
        ens['sys_config_lsb'] = '{0:0=8b}'.format(int.from_bytes(fobj.read(1), 'little'))
        ens['sys_config_msb'] = '{0:0=8b}'.format(int.from_bytes(fobj.read(1), 'little'))
        ens['flag'] = int.from_bytes(fobj.read(1), 'little')
        ens['lag_length'] = int.from_bytes(fobj.read(1), 'little')
        ens['n_beams'] = int.from_bytes(fobj.read(1), 'little')
        ens['n_cells'] = int.from_bytes(fobj.read(1), 'little')
        ens['pings_per_ens'] = int.from_bytes(fobj.read(2), 'little')
        ens['depth_cell_cm'] = int.from_bytes(fobj.read(2), 'little')
        ens['blank_after_transmit_cm'] = int.from_bytes(fobj.read(2), 'little')
        ens['sig_proc_mode'] = int.from_bytes(fobj.read(1), 'little')
        ens['low_corr_thres'] = int.from_bytes(fobj.read(1), 'little')
        ens['code_reps'] = int.from_bytes(fobj.read(1), 'little')
        ens['pg_min'] = int.from_bytes(fobj.read(1), 'little')
        ens['error_vel_thres_mm_s'] = int.from_bytes(fobj.read(2), 'little')

        time_between_pings = dict()
        time_between_pings['minutes'] = int.from_bytes(fobj.read(1), 'little')
        time_between_pings['seconds'] = int.from_bytes(fobj.read(1), 'little')
        time_between_pings['hundredths'] = int.from_bytes(fobj.read(1), 'little')
        ens['time_between_pings'] = time_between_pings
        ens['coord_sys'] = '{0:0=8b}'.format(int.from_bytes(fobj.read(1), 'little'))
        ens['heading_alignment'] = int.from_bytes(fobj.read(2), 'little', signed=True) / 100
        ens['heading_bias'] = int.from_bytes(fobj.read(2), 'little', signed=True) / 100
        ens['sensor_source'] = '{0:0=8b}'.format(int.from_bytes(fobj.read(1), 'little'))
        ens['sensor_avail'] = '{0:0=8b}'.format(int.from_bytes(fobj.read(1), 'little'))

        ens['bin_1_dist_cm'] = int.from_bytes(fobj.read(2), 'little')
        ens['xmit_pulse_length_cm'] = int.from_bytes(fobj.read(2), 'little')
        ens['ref_layer_start'] = int.from_bytes(fobj.read(1), 'little')
        ens['ref_layer_end'] = int.from_bytes(fobj.read(1), 'little')
        ens['false_target_thres'] = int.from_bytes(fobj.read(1), 'little')
        _ = int.from_bytes(fobj.read(1), 'little')
        ens['transmit_lag_dist'] = int.from_bytes(fobj.read(2), 'little')
        ens['board_serial'] = int.from_bytes(fobj.read(8), 'big')

        ens['sys_bandwidth'] = int.from_bytes(fobj.read(2), 'little')
        ens['sys_power'] = int.from_bytes(fobj.read(1), 'little')
        _ = int.from_bytes(fobj.read(1), 'little')

        ens['instrument_serial'] = int.from_bytes(fobj.read(4), 'little')
        ens['beam_angle'] = int.from_bytes(fobj.read(1), 'little')

        # Variable leader (time)
        ens['var_leader_id_1'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['var_leader_id_2'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['ens_number'] = int.from_bytes(fobj.read(2), 'little')
        y_ = int.from_bytes(fobj.read(1), 'little') + 2000
        m_ = int.from_bytes(fobj.read(1), 'little')
        d_ = int.from_bytes(fobj.read(1), 'little')
        H_ = int.from_bytes(fobj.read(1), 'little')
        M_ = int.from_bytes(fobj.read(1), 'little')
        S_ = int.from_bytes(fobj.read(1), 'little')
        ens['time'] = '{0}-{1:02}-{2:02}T{3:02}:{4:02}:{5:02}'.format(y_, m_, d_, H_, M_, S_)

        # Skip to next ensemble
        n_bytes_skip = (ens['n_bytes_ens']
                        - (6 + 2 * ens['n_data_types'])  # header
                        - 59                             # Fixed leader
                        - 10                             # part of variable leader
                        + 2)                             # checksum
        _ = fobj.read(n_bytes_skip)

    else:
        ens = None

    return ens


def read_ens_md_hex(fobj):
    """
    Read metadata of one ADCP ensemble from byte stream.

    Returns
    -------
    dict:
        Metadata of the ensemble in hex values.

    """
    # Header
    hd_id = int.from_bytes(fobj.read(1), 'little')
    if hd_id:
        ens = dict()
        ens['hd_id'] = format(hd_id, 'x') 
        ens['ds_id'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['n_bytes_ens'] = int.from_bytes(fobj.read(2), 'little')
        _ = int.from_bytes(fobj.read(1), 'little')
        n_data_types = int.from_bytes(fobj.read(1), 'little')
        ens['n_data_types'] = format(n_data_types, 'x')
        for i_ in range(n_data_types):
            ens['offset_data_%d' % i_] = hex_str_n_bytes(fobj, 2, 'little')

        # Fixed leader
        ens['fl_id_1'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['fl_id_2'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['cpu_fw_ver'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['cpu_fw_rev'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['sys_config_lsb'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['sys_config_msb'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['flag'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['lag_length'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['n_beams'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['n_cells'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['pings_per_ens'] =  hex_str_n_bytes(fobj, 2, 'little') 
        ens['depth_cell_cm'] =  hex_str_n_bytes(fobj, 2, 'little') 
        ens['blank_after_transmit_cm'] =  hex_str_n_bytes(fobj, 2, 'little') 
        ens['sig_proc_mode'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['low_corr_thres'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['code_reps'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['pg_min'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['error_vel_thres_mm_s'] =  hex_str_n_bytes(fobj, 2, 'little') 

        time_between_pings = dict()
        time_between_pings['minutes'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        time_between_pings['seconds'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        time_between_pings['hundredths'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['time_between_pings'] = time_between_pings
        ens['coord_sys'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['heading_alignment'] =  hex_str_n_bytes(fobj, 2, 'little')
        ens['heading_bias'] =  hex_str_n_bytes(fobj, 2, 'little')
        ens['sensor_source'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['sensor_avail'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')

        ens['bin_1_dist_cm'] =  hex_str_n_bytes(fobj, 2, 'little')
        ens['xmit_pulse_length_cm'] = hex_str_n_bytes(fobj, 2, 'little') 
        ens['ref_layer_start'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['ref_layer_end'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['false_target_thres'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        _ = int.from_bytes(fobj.read(1), 'little')
        ens['transmit_lag_dist'] =  hex_str_n_bytes(fobj, 2, 'little') 
        ens['board_serial'] =  hex_str_n_bytes(fobj, 8, 'big')

        ens['sys_bandwidth'] =  hex_str_n_bytes(fobj, 2, 'little') 
        ens['sys_power'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        _ = int.from_bytes(fobj.read(1), 'little')

        ens['instrument_serial'] =  hex_str_n_bytes(fobj, 4, 'little')
        ens['beam_angle'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')

        # Variable leader (time)
        ens['var_leader_id_1'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['var_leader_id_2'] = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['ens_number'] =  hex_str_n_bytes(fobj, 2, 'little')
        y_ = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        m_ = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        d_ = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        H_ = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        M_ = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        S_ = format(int.from_bytes(fobj.read(1), 'little'), 'x')
        ens['time'] = '{0}-{1}-{2}T{3}:{4}:{5}'.format(y_, m_, d_, H_, M_, S_)

        # Skip to next ensemble
        n_bytes_skip = (ens['n_bytes_ens']
                        - (6 + 2 * n_data_types)         # header
                        - 59                             # Fixed leader
                        - 10                             # part of variable leader
                        + 2)                             # checksum
        _ = fobj.read(n_bytes_skip)

    else:
        ens = None

    return ens


def metadata_csv(fname):
    """
    Generate human readable and hex summary of pd0 metadata.

    Parameters
    ----------
    fname: str
        Path and name to pd0 file.

    """
    # Open output csv file
    with open('%s_dat.csv' % fname[:-4], 'w') as csvfile:

        # Open ADCP binary file
        with open(fname, 'rb') as binfile:

            # Read first ensemble
            ens = read_ens_md_info(binfile)

            # Init output file
            csv_writer = csv.DictWriter(csvfile, fieldnames=ens.keys())
            csv_writer.writeheader()
            csv_writer.writerow(ens)

            while ens:
                # Read one ensemble's data
                ens = read_ens_md_info(binfile)
                if ens:
                    csv_writer.writerow(ens)

    # Open output hex file
    with open('%s_hex.csv' % fname[:-4], 'w') as csvfile:

        # Open ADCP binary file
        with open(fname, 'rb') as binfile:

            # Read first ensemble
            ens = read_ens_md_hex(binfile)

            # Init output file
            csv_writer = csv.DictWriter(csvfile, fieldnames=ens.keys())
            csv_writer.writeheader()
            csv_writer.writerow(ens)

            while ens:
                # Read one ensemble's data
                ens = read_ens_md_hex(binfile)
                if ens:
                    csv_writer.writerow(ens)

