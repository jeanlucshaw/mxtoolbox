"""
Convert Canadian ice service (CIS) ESRI shapefiles to ascii format (dex).

CIS daily ice analysis charts (dailys) and regional analysis charts (weeklys) are
collected every day on mixing and processed to provide information such as,

* Ice thickness maps.
* First occurence maps.
* Total gulf, Newfoundland shelf and Scotian Shelf ice volumes.
* Comparisons to climatology.

These analyses are performed by a combination of Perl/awk routines and are
facilitated by first transforming the shapefiles to gridded ascii plain text
in a format called dex, containing geographical coordinates and egg code data.
Time information is carried by the file name (YYYYMMDD) and the columns in
the file are ordered as follows,

* Longitude (west)
* Latitude
* Total ice concentration
* Partial ice concentration (thickest ice)
* Stage of developpment (thickest ice)
* Form of ice (thickest ice)
* Partial ice concentration (second thickest ice)
* Stage of developpment (second thickest ice)
* Form of ice (second thickest ice)
* Partial ice concentration (third thickest ice)
* Stage of developpment (third thickest ice)
* Form of ice (third thickest ice)

This module performs the conversion and is meant to be called from command
line. The options are the following,
"""
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import shapefile
from warnings import warn
from mpl_toolkits.basemap import Basemap
from os.path import exists
from mxtoolbox.process.math import in_polygon


__all__ = ['load_cis_shp',
           '_shp2dex',
           '_parse_prj',
           '_manage_shapefile_types',
           '_separate_wrapping_polygons',
           'plot_cis_shp']


def load_cis_shp(name):
    """
    Read CIS shapefile to dataframe and polygon list.

    Creates a pandas DataFrame with records from the `.dbf`
    companion file as rows and the field names as columns. The
    dataframe is also sorted by polygon area as this is needed
    further down the processing chain. For uniformity, empty
    strings in records are replaced by 'X'. A reference to the
    shape object of each record is found in column `shapes`.

    Parameters
    ----------
    name : str
        Path and name to input shapefile (.shp).

    Returns
    -------
    dataframe : pandas.DataFrame
        Shapefile record and field information.
    empty : bool
        True if missing fields essential for processing.
    
    """
    sf = shapefile.Reader(name)
    fld = np.array(sf.fields)[:, 0]
    shp = np.array(sf.shapes())
    rcd = np.array(sf.records())

    # Empty strings become X
    rcd[rcd == ''] = 'X'
    
    # Load to pandas dataframe
    dataframe = pd.DataFrame(rcd, columns = fld[1:])

    # Convert area to numeric and sort
    dataframe['shapes'] = shp
    dataframe['AREA'] = np.float64(dataframe.AREA.values)
    dataframe = dataframe.sort_values('AREA', ignore_index=True)

    # Flag as empty if not enough fields
    empty = dataframe.shape[1] < 11

    return dataframe, empty


def _manage_shapefile_types(dataframe):
    """
    Funnel shapefiles types to uniform labels.

    Labels of the information inside CIS shapefiles
    that this module processes have changed several times
    in the past. To avoid cluttered error handling in the
    main code from future new shapefile types, all files
    are converted to type Z before processing. Important
    label differences between types are summarized below.

    Shapefile types

        * Type A:

           | Legend string = A_LEGEND
           | Legend strings = ['Fast ice', 'Ice free', 'Land', 'Open water', 'Remote egg']
           | Old egg code = E_*
           | Area string = AREA
           | Missing concentration = ['']
           | Missing form = ['', 'X']
           | Missing stage = ['']
           | Example = "data/GEC_H_19740102.shp"

        * Type B:

           | Legend string = POLY_TYPE
           | Legend strings = ['I', 'L', 'N', 'W']
           | New egg code = *
           | Area string = AREA
           | Missing concentration = ['', '-9', '99']
           | Missing form = ['', '-9', '99']
           | Missing stage = ['', '-9', '99']
           | Example = "data/GEC_D_20150108.shp"

        * Type C:

           | Legend string = SGD_POLY_T
           | Legend strings = ['I', 'L', 'W']
           | New egg code = SGD_*
           | Old egg code = E_*
           | Area string = AREA
           | Missing concentration = ['']
           | Missing form = ['', 'X']
           | Missing stage = ['']
           | Example = "data/GEC_H_20200120.shp"

        * Type D:

           | Legend string = POLY_TYPE
           | Legend strings = ['I', 'L', 'W']
           | New egg code = *
           | Old egg code = E_*
           | Area string = AREA
           | Missing concentration = ['']
           | Missing form = ['', 'X']
           | Missing stage = ['']
           | Example = "data/GEC_H_20200309.shp"

        * Type Z:

           | Legend string = LEGEND
           | Legend strings = ['Ice', 'Fast-ice', 'Land', 'IF', 'missing']
           | Old egg code = E_*
           | Area string = AREA
           | Missing concentration = ['X']
           | Missing form = ['X']
           | Missing stage = ['X']
           | Example = None

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Output from load_cis_shp of type A, B, C or D.

    Returns
    -------
    dataframe : pandas.DataFrame
        Type Z.

    """
    fields = ['E_CT',
              'E_CA',
              'E_SA',
              'E_FA',
              'E_CB',
              'E_SB',
              'E_FB',
              'E_CC',
              'E_SC',
              'E_FC']

    # Type A
    if 'A_LEGEND' in dataframe.keys():
        # Rename columns
        mapper = {'A_LEGEND': 'LEGEND'}
        dataframe = dataframe.rename(mapper, axis=1)

        # Convert legend labels
        dataframe.at[(dataframe.LEGEND == 'Fast ice'), 'LEGEND'] = 'Fast-ice'
        dataframe.at[(dataframe.LEGEND == 'Ice free') |
                      (dataframe.LEGEND == 'Open water'), 'LEGEND'] = 'IF'

    # Type B
    elif ('POLY_TYPE' in dataframe.keys()) and ('E_CT' not in dataframe.keys()):
        # Rename egg code columns
        mapper = {'CT': 'E_CT',
                  'CA': 'E_CA',
                  'SA': 'E_SA',
                  'FA': 'E_FA',
                  'CB': 'E_CB',
                  'SB': 'E_SB',
                  'FB': 'E_FB',
                  'CC': 'E_CC',
                  'SC': 'E_SC',
                  'FC': 'E_FC',
                  'POLY_TYPE': 'LEGEND'}
        dataframe = dataframe.rename(mapper, axis=1)

        # Translate to old egg code for each entry
        for i in dataframe.index.values:
            raw = dataframe.iloc[i][fields]
            translated = _newegg_2_oldegg(raw, 'bla', i)
            dataframe.at[i, translated.keys()] = translated.values

        # Convert legend labels
        dataframe.at[(dataframe.LEGEND == 'W'), 'LEGEND'] = 'IF'
        dataframe.at[(dataframe.LEGEND == 'I'), 'LEGEND'] = 'Ice'
        dataframe.at[(dataframe.LEGEND == 'L'), 'LEGEND'] = 'Land'
        dataframe.at[(dataframe.LEGEND == 'N'), 'LEGEND'] = 'missing'

    # Type C
    elif 'SGD_POLY_T' in dataframe.keys():
        # Rename columns
        mapper = {'SGD_POLY_T': 'LEGEND'}
        dataframe = dataframe.rename(mapper, axis=1)

        # Convert legend labels
        dataframe.at[(dataframe.LEGEND == 'W'), 'LEGEND'] = 'IF'
        dataframe.at[(dataframe.LEGEND == 'I'), 'LEGEND'] = 'Ice'
        dataframe.at[(dataframe.LEGEND == 'L'), 'LEGEND'] = 'Land'
        dataframe.at[(dataframe.LEGEND == 'N'), 'LEGEND'] = 'missing'

    # Type D
    elif all([key in DF2.keys() for key in ['POLY_TYPE', 'CT', 'E_CT']]):
    # Rename columns
        mapper = {'POLY_TYPE': 'LEGEND'}
        dataframe = dataframe.rename(mapper, axis=1)

        # Convert legend labels
        dataframe.at[(dataframe.LEGEND == 'W'), 'LEGEND'] = 'IF'
        dataframe.at[(dataframe.LEGEND == 'I'), 'LEGEND'] = 'Ice'
        dataframe.at[(dataframe.LEGEND == 'L'), 'LEGEND'] = 'Land'
        dataframe.at[(dataframe.LEGEND == 'N'), 'LEGEND'] = 'missing'

    # Type not recognized
    else:
        raise TypeError('Shapefile type not in [A, B, C]')

    return dataframe[['AREA', *fields, 'LEGEND', 'shapes']]


def _newegg_2_oldegg(egg_dict, sname, i):
    """
    Convert new more precise egg code to older more general values.

    Two different systems of egg code values exist in the CIS files. The
    most recent offers the possibility of increased precision but is
    this precision is rarely used. As it makes little difference for now
    and as the next processing step expects values in the old format,
    new format egg code is translated back via this routine.

    Parameters
    ----------
    egg_dict : dict
        New egg code keys and values.
    sname : str
        Shapefile name.
    jj : int
        Polygon index.

    Returns
    -------
    translated : dict
        Input translated to old egg code.
    """
    dcon = {'X': 'X',
            '00': 'X',
            '91': '9+',
            '10': '1',
            '20': '2',
            '30': '3',
            '40': '4',
            '50': '5',
            '60': '6',
            '70': '7',
            '80': '8',
            '90': '9',
            '92': '10',
            '99': 'X',
            '-9': 'X',
            '9-': 'X',
            '98': 'X',
            '01': 'X',
            '02': 'X'}

    dsta = {'X' : 'X',
            '-9': 'X',
            '9-': 'X',
            '99': 'X',
            '01': '1',
            '02': '1',
            '03': '1',
            '04': '1',
            '05': '1',
            '06': '1',
            '07': '1',
            '08': '1',
            '09': '1',
            '10': '4',
            '11': '4',
            '12': '4',
            '13': '4',
            '14': '4',
            '15': '3',
            '16': '3',
            '17': '3',
            '18': '3',
            '19': '3',
            '20': '3',
            '21': '3',
            '22': '3',
            '23': '3',
            '24': '3',
            '25': '3',
            '26': '3',
            '27': '3',
            '28': '3',
            '29': '3',
            '30': '8',
            '31': '8',
            '32': '8',
            '33': '8',
            '34': '8',
            '35': '8',
            '36': '8',
            '37': '8',
            '38': '8',
            '39': '8',
            '40': '8',
            '41': '8',
            '42': '8',
            '43': '8',
            '44': '8',
            '45': '8',
            '46': '8',
            '47': '8',
            '48': '8',
            '49': '8',
            '50': '9',
            '51': '9',
            '52': '9',
            '53': '9',
            '54': '9',
            '55': '1.',
            '56': '1.',
            '57': '1.',
            '58': '1.',
            '59': '1.',
            '60': '1.',
            '61': '1.',
            '62': '1.',
            '63': '4.',
            '64': '4.',
            '65': '4.',
            '66': '4.',
            '67': '4.',
            '68': '4.',
            '69': '4.',
            '70': '4.',
            '71': '4.',
            '72': '4.',
            '73': '4.',
            '74': '4.',
            '75': '4.',
            '76': '4.',
            '77': '4.',
            '78': '4.',
            '81': '1',
            '82': '2',
            '83': '3',
            '84': '4',
            '85': '5',
            '86': '6',
            '87': '7',
            '88': '8',
            '89': '9',
            '91': '1.',
            '93': '4.',
            '95': '7.',
            '96': '8.',
            '97': '9.',
            '98': '98'}  #  Kept from SIGRID-3, leads to icebergs exception

    dfor = {'X' : 'X',
            '-9': 'X',
            '9-': 'X',
            '99': 'X',
            '22': '0',
            '01': '1',
            '02': '2',
            '03': '3',
            '04': '4',
            '05': '5',
            '06': '6',
            '07': '7',
            '08': '8',
            '09': '9',
            '10': '10'}  #  Kept from SIGRID-3, leads to icebergs exception

    # Translate
    for key in egg_dict.keys():
        try:
            if key in ['E_CT', 'E_CA', 'E_CB', 'E_CC']:
                egg_dict[key] = dcon[egg_dict[key]]
            elif key in ['E_SA', 'E_SB', 'E_SC']:
                egg_dict[key] = dsta[egg_dict[key]]
            elif key in ['E_FA', 'E_FB', 'E_FC']:
                egg_dict[key] = dfor[egg_dict[key]]
        except KeyError as e:
            print("KeyError : %s for key %s, file: %s, polygon number %d" % (e, key, sname, i))
            egg_dict[key] = 'X'

    return egg_dict

def plot_cis_shp(sname, savepath=None, hl=None, lab=None):
#     """
#     Plot polygons of a CIS ice shapefile
#     """
#     trans = exists(sname[0: -4] + ".prj")
#     if trans:
#         # Read projection file
#         proj, lat0, lon0, std1, std2, a, ifp = _parse_prj(sname[0:-4] + ".prj")

#         # Datum
#         b = a * (ifp - 1) / ifp

#         # Basemap
#         urlon = -38
#         urlat = 50
#         m = Basemap(urcrnrlon=urlon,
#                     urcrnrlat=urlat,
#                     llcrnrlon=lon0,
#                     llcrnrlat=lat0,
#                     projection=proj,
#                     rsphere=(a, b),
#                     resolution='l',
#                     lat_1=std1,
#                     lat_2=std2,
#                     lon_0=lon0,
#                     lat_0=lat0)

#     # Read shapefile
#     sf = shapefile.Reader(sname) ;
#     shp = np.asarray(sf.shapes())
#     fld = np.asarray(sf.fields)[:, 0]
#     rcd = np.asarray(sf.records())

#     # Get size and polygon legend
#     # roll because of deletion flag
#     lgdi = np.flatnonzero([(fld=='A_LEGEND') | (fld=='POLY_TYPE') | (fld=='SGD_POLY_T')])[0] - 1
# #    lgdi = np.where(np.roll(np.logical_or(fld[:, 0] == 'A_LEGEND',
# #                                          fld[:, 0] == 'POLY_TYPE'), -1))
#     sfci = np.where(np.roll(fld == 'AREA', -1))
#     lgd = np.asarray([jj[lgdi] for jj in rcd])
#     sfc = np.asarray([float(jj[sfci[0][0]]) for jj in rcd])

#     # Sort shapes according to size largest first
#     shp = shp[np.argsort(sfc)[::-1]]
#     rcd = rcd[np.argsort(sfc)[::-1]]
#     lgd = lgd[np.argsort(sfc)[::-1]]

#     # Place land elements last
#     I = np.append(np.nonzero(np.logical_and(lgd != 'L', lgd != 'Land'))[0],
#                   np.nonzero(np.logical_or(lgd == 'L', lgd == 'Land'))[0])

#     shp = shp[I]
#     rcd = rcd[I]
#     lgd = lgd[I]

#     fig, ax = plt.subplots(1, figsize=(15, 10))
#     for jj in range(shp.size):

#         if lab == 'lgd':
#             prt = lgd[jj]
#         elif lab == 'num':
#             prt = jj
#         elif isinstance(lab, int):
#             prt = rcd[jj, lab]
#         else:
#             prt = None

#         # Get polygon coordinates
#         P = shp[jj].points
#         x = np.nan * np.ones(len(P))
#         y = np.nan * np.ones(len(P))
#         for ii in np.arange(0, len(P)):
#             y[ii] = P[ii][1]
#             x[ii] = P[ii][0]
#         if trans:
#             lon, lat = m(x, y, inverse=True)
#         else:
#             lon, lat = x, y
#         llon, llat, I = _separate_wrapping_polygons(lon, lat, decimals=5)

#         if (lgd[jj] == 'Land' or lgd[jj] == 'L'):
#             ax.fill(llon[0], llat[0], edgecolor='k', facecolor='w', alpha=0.5)
#         else:
#             ax.fill(llon[0], llat[0], edgecolor='k', alpha=0.5)
#             ax.text(llon[0].mean(), llat[0].mean(), prt)
#             if jj == hl:
#                 ax.fill(llon[0], llat[0], edgecolor='r', facecolor='r')
#                 ax.text(llon[0].mean(), llat[0].mean(), prt)

#     ax.set_title(sname)
#     if savepath != None:
#         plt.savefig(savepath + sname[0: -4] + '.png')
#     plt.show()
    return None


def _parse_prj(fname):
    """
    Parse shapefile (.prj) for projection parameters.

    Geographical projection information for shapefile data is
    contained in a companion file with the extension `.prj`. The
    Basemap class instance needs this information to convert
    polygon coordinates from map units (m) to longitudes and
    latitudes.

    Parameters
    ----------
    fname : str
        Name of the projection file.

    Returns
    -------
    proj : str
        Projection name abbreviated for input to Basemap.
    lat0 : float
        Latitude of origin.
    lon0 : float
        Longitude of origin.
    std1 : float
        Standard parallel 1 used by LCC projection.
    std2 : float
        Standard parallel 2 used by LCC projection.
    a : float
        Datum semi-major radius.
    ifp : float
        Inverse flattening parameter. Used to obtain the Datum
        semi-minor radius.

    Note
    ----
        For the moment, only Lambert conformal conic projections
        are supported.

    """

    # Init output
    proj = None
    lat0 = None
    lon0 = None
    std1 = None
    std2 = None
    a = None
    ifp = None

    # Read file
    file = open(fname, 'r')
    string = file.readline()

    # Set up regex
    rx_dict = {'proj': re.compile(r'PROJECTION\["(.*)"\],',
                                  re.IGNORECASE),
               'lat0': re.compile(r'PARAMETER\["latitude_of_origin",([-\.\d]*)\]',
                                  re.IGNORECASE),
               'lon0': re.compile(r'PARAMETER\["central_meridian",([-\.\d]*)\]',
                                  re.IGNORECASE),
               'std1': re.compile(r'PARAMETER\["standard_parallel_1",([-\.\d]*)\]',
                                  re.IGNORECASE),
               'std2': re.compile(r'PARAMETER\["standard_parallel_2",([-\.\d]*)\]',
                                  re.IGNORECASE),
               'rsph': re.compile(r'SPHEROID\[.*,([-\.\d]*),([-\.\d]*)\]',
                                  re.IGNORECASE)}

    # Match regex
    for key, rx in rx_dict.items():
        match = rx.search(string, re.IGNORECASE)
        if match:
            if key == "proj":
                if match.group(1) == "Lambert_Conformal_Conic":
                    proj = 'lcc'
            if key == "lat0":
                lat0 = float(match.group(1))
            if key == "lon0":
                lon0 = float(match.group(1))
            if key == "std1":
                std1 = float(match.group(1))
            if key == "std2":
                std2 = float(match.group(1))
            if key == "rsph":
                a = float(match.group(1))
                ifp = float(match.group(2))

    return proj, lat0, lon0, std1, std2, a, ifp


def _separate_wrapping_polygons(x, y, decimals=5):
    """
    Find wrapping points of polygon sequence stored in vectors `x` and `y`.

    The CIS shapefiles contain complex polygons with 'holes', which are often other
    polygons of the data set. The encompassing polygon is the first to be defined in
    vectors `x` and `y`, and it 'wraps' to its first coordinate before the start of
    the smaller polygon. This routine separates the these polygons by finding the
    wrapping points.

    Parameters
    ----------
    x, y : 1D array
        Coordinates of the polygon sequence.
    decimals : int
        Number of decimals to keep in float comparison.

    Returns
    -------
    polygons_x, polygons_y : list of 1D arrays
        Coordinates of the separated polygons.
    wrap_index : 1D int array
        Index value of the wrapping points.
    """

    # Make array and sort
    a = np.vstack((x, y)).T
    a = np.round(a, decimals=decimals)

    # Find wrapping points indices
    arr, inv, cnt = np.unique(a, axis=0, return_inverse=True, return_counts=True)
    wrap_index = np.nonzero(cnt[inv] > 1)[0]

    # Remove consecutive repeating coordinates
    # This may look unnecessarily complicated but it solves a non-trivial problem and works.
    Iu, Id = np.hstack((0, wrap_index)), np.hstack((wrap_index, wrap_index[-1]))
    if wrap_index.size % 2 == 0:
        cond = np.logical_or(np.logical_and(np.diff(Iu) == 1,
                                            np.diff(x[Iu]) == 0,
                                            np.diff(y[Iu]) == 0),
                             np.logical_and(np.diff(Id[::-1]) == -1,
                                            np.diff(x[Id[::-1]]) == 0,
                                            np.diff(y[Id[::-1]]) == 0)[::-1])
    else:
        cond = np.logical_and(np.diff(Iu) == 1,
                              np.diff(x[Iu]) == 0,
                              np.diff(y[Iu]) == 0)

    wrap_index = wrap_index[np.invert(cond)]

    # Make list of polygons
    N = int(wrap_index.size)
    polygons_x, polygons_y = [],[]
    for ii in range(0, N-2, 2):
        polygons_x.append(x[wrap_index[ii]: wrap_index[ii + 2]])
        polygons_y.append(y[wrap_index[ii]: wrap_index[ii + 2]])

    # Last polygon
    polygons_x.append(x[wrap_index[-2]:])
    polygons_y.append(y[wrap_index[-2]:])

    return polygons_x, polygons_y, wrap_index


def _show_cis_field(fname, field):
    """
    Print possible values of shapefile field.

    Parameters
    ----------
    fname : str
        Shapefile path and name (.shp).
    field : str
        Name of field to detail.

    """
    sf = shapefile.Reader(fname)
    fld = np.array(sf.fields)[:, 0]
    rcd = np.array(sf.records())

    output = {}
    for record in rcd:
        dicto = dict(zip(fld[1:], record))
        output = {*output, dicto[field]}

    print(field, ': possible values :', output)


def _show_cis_summary(fname):
    """
    Print possible values of egg code data for CIS (.shp).

    Parameters
    ----------
    fname : str
        Shapefile path and name (.shp).
    """
    fields = ['A_LEGEND', 'POLY_TYPE', 'SGD_POLY_T',
              'CT', 'CA', 'CB', 'CC',
              'SA', 'SB', 'SC',
              'FA', 'FB', 'FC', 
              'E_CT', 'E_CA', 'E_CB', 'E_CC',
              'E_SA', 'E_SB', 'E_SC',
              'E_FA', 'E_FB', 'E_FC', 
              'SGD_CT', 'SGD_CA', 'SGD_CB', 'SGD_CC',
              'SGD_SA', 'SGD_SB', 'SGD_SC',
              'SGD_FA', 'SGD_FB', 'SGD_FC']
    print(fname)
    for f in fields:
        try:
            _show_cis_field(fname, f)
        except:
            pass

def _shp2dex(sname,
             gname,
             urlon=-38,
             urlat=52,
             lwest=True,
             skipland=True,
             pip='mpl'):
    """
    Extract egg code data from CIS ESRI shapefiles.

    Conversion from map coordinates to decimal degrees
    is carried out by Basemap. Point in polygon querying
    is done using the shapely or matplotlib libraries.

    Parameters
    ----------
    sname : string
        The shapefile (.shp) path and name.
    gname : string
        Path and name of the text file containing
        the queried coordinates. This is expected to be
        a two column file containing [lon lat] points
        separated by spaces.
    urlon : float
        Longitude of projection upper right corner.
        Default value of -38 works for shapefiles
        in Eastern Canada.
    urlat : float
        Latitude of projection upper right corner.
        Default value of 52 works for
        shapefiles in Eastern Canada.
    lwest : bool
        Return longitude as negative west, defaults
        to True.
    skipland : bool
        Skip polygons marked as land if this is True.
        Defaults to True. Skipping land will be faster.
        If it is not known if the grid has points
        on land and you wish to mark land points, set
        to false.

    Returns
    -------
    pandas.Dataframe
        A dataframe with columns,

            * Longitude (west)
            * Latitude
            * Legend
            * Total ice concentration
            * Partial ice concentration (thickest ice)
            * Stage of developpment (thickest ice)
            * Form of ice (thickest ice)
            * Partial ice concentration (second thickest ice)
            * Stage of developpment (second thickest ice)
            * Form of ice (second thickest ice)
            * Partial ice concentration (third thickest ice)
            * Stage of developpment (third thickest ice)
            * Form of ice (third thickest ice)

    """
    # Option management
    if lwest:
        sign = -1
    else:
        sign = 1

    # Parameters
    egg_strs = ['E_CT',
                'E_CA', 'E_SA', 'E_FA',
                'E_CB', 'E_SB', 'E_FB',
                'E_CC', 'E_SC', 'E_FC']

    # Initialize output
    columns = ['lon', 'lat', 'LEGEND', *egg_strs, 'assigned']
    df_output = pd.read_csv(gname, names=columns, na_filter=False)
    df_output['assigned'] = False

    # Read projection file
    if exists(sname[0:-4]+".prj"):
        proj, lat0, lon0, std1, std2, a, ifp = _parse_prj(sname[0:-4] + ".prj")

        # Datum
        b = a * (ifp - 1) / ifp

        # Basemap
        m = Basemap(urcrnrlon=urlon,
                    urcrnrlat=urlat,
                    llcrnrlon=lon0,
                    llcrnrlat=lat0,
                    projection=proj,
                    rsphere=(a, b),
                    resolution='l',
                    lat_1=std1,
                    lat_2=std2,
                    lon_0=lon0,
                    lat_0=lat0)
        prjfile = True
    else:
        prjfile = False

    # Read shapefile
    df_records, empty = load_cis_shp(sname)

    # Only process if the required fields are present
    if not empty:

        # Convert to uniform naming conventions
        df_records = _manage_shapefile_types(df_records)

        # Interpolate, loops over shapfile polygons
        for (i, shape) in enumerate(df_records.shapes.values):

            # Skip land polygons if skipland is True
            if (df_records.LEGEND[i] == 'Land') and skipland:
                pass
            else:

                # Get polygon coordinates
                x, y = np.split(np.array(shape.points), 2, axis=1)
                if prjfile:
                    lon, lat = m(x.flatten(), y.flatten(), inverse=True)
                else:
                    lon, lat = x.flatten(), y.flatten()

                # Only keep outside polygon
                polygons_lon, polygons_lat, _ = _separate_wrapping_polygons(lon,
                                                                            lat,
                                                                            decimals=7)
                lon, lat = polygons_lon[0], polygons_lat[0]

                # Points to check
                condition = '%f < lon < %f & %f < lat < %f & not assigned'
                df_candidates = df_output.query(condition % (lon.min(),
                                                       lon.max(),
                                                       lat.min(),
                                                       lat.max())) 

                # Find pip with Sloan Algorithm
                if pip == 'sloan':
                    Po = Polygon(lon, lat)

                    # Find grid points in polygon
                    mindst = Po.is_inside(df_candidates.lon.values,
                                          df_candidates.lat.values)
                    target_index = df_candidates.loc[mindst >= 0].index.values
                else:
                    inside = in_polygon(df_candidates.lon.values,
                                        df_candidates.lat.values,
                                        lon,
                                        lat)
                    target_index = df_candidates.loc[inside].index.values
                
                # Index of points to write in egg
                if target_index.size > 0:
                    # For improved readability
                    r = df_records.iloc[i]
                    
                    " Legend "
                    # Fast-ice
                    if ("Fast-ice" == r.LEGEND) or r.E_FA == '8':
                        df_output.at[target_index, 'LEGEND'] = 'Fast-ice'
                    # On land
                    elif r.LEGEND == "Land":
                        df_output.at[target_index, 'LEGEND'] = 'Land'
                    # Missing data
                    elif r.LEGEND == 'missing':
                        df_output.at[target_index, 'LEGEND'] = 'missing'
                    # Open water
                    elif (r.LEGEND == 'IF') or (r.E_CT == 'X' and r.E_CA == 'X'):
                        df_output.at[target_index, 'LEGEND'] = 'IF'
                    # Icebergs
                    elif (r.E_SA == '98' and r.E_FA == '10'):
                        df_output.at[target_index, 'LEGEND'] = 'Icebergs'
                    # Ice
                    else:
                        df_output.at[target_index, 'LEGEND'] = 'Egg'

                    " Egg code "
                    # Assign egg code
                    for egg in egg_strs:
                        df_output.at[target_index, egg] = r[egg]

                    # No need to check these grid points again
                    df_output.at[target_index, 'assigned'] = True

        # Unassigned points are considered missing
        for egg in egg_strs:
            df_output.at[df_output['LEGEND'] == '', egg] = 'X'
        df_output.at[df_output['LEGEND'] == '', 'LEGEND'] = 'missing'
    else:
        # This happens when the shapefile is empty
        warn('Shapefile is empty: returning "missing" for all grid points in dex')
        for egg in egg_strs:
            df_output.at[:, egg] = 'X'
        df_output.at[:, 'LEGEND'] = 'missing'

    # Reverse longitude sign if specified at input
    df_output['lon'] *= sign
    
    return df_output[['lon', 'lat', 'LEGEND', *egg_strs]]


# Command line interface
if __name__ == '__main__':
    from time import perf_counter

    # Set up parser
    parser  = argparse.ArgumentParser(usage=__doc__)

    # Define arguments
    parser.add_argument('shapefiles',
                        metavar='',
                        help='Name and path of shapefile, or * expression',
                        nargs='+')
    parser.add_argument('-g',
                        '--gridfile',
                        metavar='',
                        help='Name and path of grid file')
    parser.add_argument('-l',
                        '--urlat',
                        metavar='',
                        help='Latitude of upper right projection corner')
    parser.add_argument('-L',
                        '--urlon',
                        metavar='',
                        help='Longitude of upper right projection corner')
    parser.add_argument('-E',
                        '--least',
                        action='store_true',
                        help='Make longitude positive towards the east')
    parser.add_argument('-e','--earth',
                        action='store_true',
                        help='Check land polygons for grid points')
    args    = parser.parse_args()

    # Parameters
    fields = ['E_CT',
              'E_CA', 'E_SA', 'E_FA',
              'E_CB', 'E_SB', 'E_FB',
              'E_CC', 'E_SC', 'E_FC']
    
    # Option switches
    if args.gridfile != None:
        grid = args.gridfile
    else:
        grid = "/data/SeaIce/scripts/CIS_grid_lon015_lat01.csv"

    kwargs  = {}
    if args.urlon:
        kwargs['urlon'] = args.urlon
    if args.urlat:
        kwargs['urlat'] = args.urlat
    if args.least:
        kwargs['lwest'] = False
    if args.earth:
        kwargs['skipland'] = False

    # Convert and write to file
    for shp in args.shapefiles:
        # Perform conversion
        df_dex = _shp2dex(shp, grid, **kwargs)

        # Format to match required character lengths, spacing and
        df_dex['printable'] = (df_dex['lon'].apply(lambda x : '%07.3f' % x) + " " +
                               df_dex['lat'].apply(lambda x : '%05.2f' % x) + " " +
                               df_dex['E_CT'].apply(lambda x : x.rjust(2)) + "  " +
                               df_dex['E_CA'] + " " +
                               df_dex['E_SA'] + " " +
                               df_dex['E_FA'] + "  " +
                               df_dex['E_CB'] + " " +
                               df_dex['E_SB'] + " " +
                               df_dex['E_FB'] + "  " +
                               df_dex['E_CC'] + " " +
                               df_dex['E_SC'] + " " +
                               df_dex['E_FC'])
        df_dex.at[df_dex.E_CC == 'X', 'printable'] = df_dex['printable'].apply(lambda x : x[:30])
        df_dex.at[df_dex.E_CB == 'X', 'printable'] = df_dex['printable'].apply(lambda x : x[:23])
        df_dex.at[df_dex.E_CA == 'X', 'printable'] = df_dex.LEGEND

        # Write to output
        df_dex.to_csv('%s.dex2' % shp[0:-4], columns=['printable'], header=False, index=False)