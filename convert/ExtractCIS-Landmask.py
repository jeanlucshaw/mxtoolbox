"""
Make column file for each closed land polygon in CIS SIGRID3 shapefile.

Specify one shapefile as positional argument 1 to make mask files. If
argument 1 is an expression for multiple files, they are assumed to be
mask files and are plotted.

Mask files have longitude east in column 1 and latitude in column 2
separated by spaces. The size scale of each polygon is shown in km
before .mask extension. Scale is determined by the square root of
the polygon area. For simplicity, scales are discretized to smaller
than 10, 100, 1000 and 10000 km. Features larger than this are marked
.LARGE.mask .

To specify mask float format use the following syntax,

    (percent)(pad char).(number decimals)f

when setting the -f option.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mxtoolbox.process as ps
import mxtoolbox.convert as cv

# Command line interface
if __name__ == '__main__':

    # Set up parser
    parser  = argparse.ArgumentParser(usage=__doc__)

    # Define arguments
    parser.add_argument('source',
                        metavar='',
                        help='Shapefile or mask files * expression.',
                        nargs='+')
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
    args = parser.parse_args()

    # View mask files
    if len(args.source) > 1:
        for mask in args.source:
            lon, lat = np.loadtxt(mask, unpack=True)
            plt.plot(lon, lat)
        plt.show()

    # Create mask files
    else:
        source = args.source[0]
        # Manage defauls
        separator = args.separator or ' '
        fmt = args.float_format or '%.3f'

        # Load to type Z dataframe
        df, _ = cv.load_cis_shp(source)
        df = cv._manage_shapefile_types(df)

        # Get projection function
        to_lon_lat = cv._get_lon_lat_converter(source)

        # Keep only land polygons
        df = df.loc[df.LEGEND == 'L']

        # Convert area field to km^2
        df['SCALE'] = np.sqrt(df.AREA / 1000 / 1000)

        # Loop over polygons
        for i, (scale, shape) in enumerate(zip(df.SCALE, df.shapes)):

            # Determine polygon scale class
            if 0 < scale <= 10:
                scale_class = '10km'
            elif 10 < scale <= 100:
                scale_class = '100km'
            elif 100 < scale <= 1000:
                scale_class = '1000km'
            elif 1000 < scale <= 10000:
                scale_class = '10000km'
            else:
                scale_class = 'LARGE'

            # Set output mask file name and path
            prefix = source.split('/')[-1].split('_')[0]
            output_name = '%s_%04d.%s.mask' % (prefix, i, scale_class)
            if args.output:
                if args.output[-1] != '/':
                    out_path = args.output + '/'
                else:
                    out_path = args.output
                output_name = out_path + output_name.split('/')[-1]

            # Get polygon points
            lon, lat = cv._get_polygon_lon_lat(shape, to_lon_lat, separate=True)

            # Write to text
            if args.output == 'plot':
                plt.plot(lon, lat)
            else:
                text_ = np.array([lon, lat]).T
                np.savetxt(output_name, text_, delimiter=separator, fmt=fmt)

        # Plot if requested
        if args.output == 'plot':
            plt.show()

