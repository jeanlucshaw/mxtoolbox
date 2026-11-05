"""
Get geographical outer limit of CIS SIGRID3 polygons.

Setting the `-o` option to plot visualizes the domain
over the file polygons instead of printing to file.

Other options and this help message can be displayed by calling,

.. code::

   $ ExtractCIS-Domain -h

The following example would create a domain files in the directory
called masks,

.. code::

   $ ExtractCIS-Domain GEC_H_YYYYMMDD.shp -o ./masks

Command line functionality depends on a simple wrapper script,

.. code::

   /usr/local/bin/ExtractCIS-Domain

"""
import shapefile
import numpy as np
import argparse
import mxtoolbox.convert as cv
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


# Command line interface
if __name__ == '__main__':

    # Set up parser
    parser  = argparse.ArgumentParser(usage=__doc__)

    # Define arguments
    parser.add_argument('source',
                        metavar='',
                        help='Shapefile to extract domain from.')
    parser.add_argument('-f',
                        '--float-format',
                        metavar='',
                        help='String format of mask coordinate.')
    parser.add_argument('-s',
                        '--separator',
                        metavar='',
                        help='Output file delimiter.')
    parser.add_argument('-o',
                        '--output',
                        metavar='',
                        help='Name of output directory.')
    parser.add_argument('-r',
                        '--shrink',
                        metavar='',
                        type=float,
                        help='Shrink domain towards center by this value in meters.')
    args = parser.parse_args()

    # Manage defauls
    separator = args.separator or ' '
    fmt = args.float_format or '%.3f'

    # Get lon lat converter function
    to_lon_lat = cv._get_lon_lat_converter(args.source)

    # Read shape data
    shapes = shapefile.Reader(args.source).shapes()

    # Increase resolution
    def increase_polygon_resolution(x_vert, y_vert, N=100):
        """
        Add `N` steps between each vertex of a polygon.
        """
        x, y = np.array([]), np.array([])
        for x_1, x_2, y_1, y_2 in zip(x_vert[:-1],
                                      x_vert[1:],
                                      y_vert[:-1],
                                      y_vert[1:]):
            x = np.hstack((x, np.linspace(x_1, x_2, N)))
            y = np.hstack((y, np.linspace(y_1, y_2, N)))

        return x, y


    # Make a convex hull with all polygon information
    for shape in shapes:
        # Add this polygon to the hull
        newpoints = np.array(shape.points)
        try:
            points = np.vstack((points, newpoints))
            hull = ConvexHull(points)
        except:
            points = newpoints

        if args.output == 'plot':
            lon, lat = cv._get_polygon_lon_lat(shape, to_lon_lat, separate=True)
            plt.plot(lon, lat)

    # Get vertex points from hull
    x_dom = hull.points[hull.vertices, 0]
    y_dom = hull.points[hull.vertices, 1]

    # Shrink polygon
    if args.shrink:
        i_left = x_dom < np.nanmean(x_dom)
        x_dom[i_left] += args.shrink
        i_right = x_dom >= np.nanmean(x_dom)
        x_dom[i_right] -= args.shrink
        i_bottom = y_dom < np.nanmean(y_dom)
        y_dom[i_bottom] += args.shrink
        i_top = y_dom >= np.nanmean(y_dom)
        y_dom[i_top] -= args.shrink

    # Tighten resolution before conversion to lat lon
    x_dom, y_dom = increase_polygon_resolution(x_dom, y_dom)
    lon_dom, lat_dom = to_lon_lat(x_dom, y_dom)


    # Set output domain file name and path
    prefix = args.source.split('/')[-1].split('_')[0]
    output_name = '%s.domain' % prefix
    if args.output:
        if args.output[-1] != '/':
            out_path = args.output + '/'
        else:
            out_path = args.output
        output_name = out_path + output_name.split('/')[-1]

    # Output as plot
    if args.output == 'plot':
        plt.plot(lon_dom, lat_dom, 'k--', lw=2)
        plt.show()
    # Output as text file
    else:
        text_ = np.array([lon_dom, lat_dom]).T
        np.savetxt(output_name, text_, delimiter=separator, fmt=fmt)

