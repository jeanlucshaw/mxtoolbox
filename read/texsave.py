"""
Save variable values for LaTeX document writing.
"""
import os
import re
from termcolor import colored, cprint


__all__ = ['_int_to_roman',
           'dataframe2latex',
           'tex_write',
           'tex_print']


def _int_to_roman(input):
    """ Convert an integer to a Roman numeral. """

    if not isinstance(input, type(1)):
        raise TypeError("expected integer, got %s" % type(input))
    if not 0 < input < 4000:
        raise (ValueError, "Argument must be between 1 and 3999")
    ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
    nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
    result = []
    for i in range(len(ints)):
        count = int(input / ints[i])
        result.append(nums[i] * count)
        input -= ints[i] * count
    return ''.join(result)


def dataframe2latex(df, fmt, cols, buf, show=False):
    """
    Write dataframe rows formatted for latex

    The string `fmt` a sequence of 'x.x' strings separated
    by space. Each space marks a column separation. To insert
    an empty column write two spaces.

    Parameters
    ----------
    df: pandas.DataFrame
        From which to write.
    fmt: str
        Float/string format of the variables.
    cols: list of int or str
        Columns to use.
    buf: str
        Name of output file.
    show: bool
        Print results to python shell.

    """
    # Format for latex
    fmt = re.sub(r'([A-Za-z_0-9.]+)', r'%\1', fmt)
    fmt = re.sub(' ', ' & ', fmt) + ' \\\\'

    # Open file buffer
    with open(buf, 'w') as ofile:

        # Write line by line
        for i_, n_ in enumerate(df.index):
            if isinstance(cols[0], str):
                if show:
                    print(fmt % tuple(df.loc[n_, cols]))
                ofile.write((fmt + '\n') % tuple(df.loc[n_, cols]))
            elif isinstance(cols[0], int):
                if show:
                    print(fmt % tuple(df.iloc[i_, cols]))
                ofile.write((fmt + '\n') % tuple(df.iloc[i_, cols]))
            else:
                raise TypeError('`cols must be a list of int or str')


def tex_write(signature, results, mode='w'):
    """
    Write variable values as TeX commands.

    Parameters
    ----------
    signature : str
        Nickname of the generating script.
    results : sequence of tuple
        Sequence of (value, comment, format) triplets.
    mode : str
        Passed to `open`. Set `a` for append, or `w` for overwrite.

    """

    # Automated file name
    fname = '%s_tsv.tex' % signature

    # Write output
    with open(fname, mode) as ofile:

        # Transform numbers in fname for TeX command compatibility
        regex = re.compile(r"\d+")
        signature = os.path.basename(signature)
        signature = regex.sub(lambda x: _int_to_roman(int(x.group(0))), signature)

        # Write saved values to file
        for i, (value, comment, fmt, unit) in enumerate(results):
            cname = 'tsv_%s_%s' % (signature, _int_to_roman(i + 1))
            cname = ''.join(word.title() for word in cname.split('_'))

            line = '\\newcommand{\\%s}{'+ fmt + '} %% %s : units %s\n'
            ofile.write(line % (cname, value, comment, unit))


def tex_print(results):
    """
    Print `results` to standard out as saved for TeX.

    Parameters
    ----------
    results : tuple of tuples
        Sequence of (value, comment, format) triplets.

    """
    # Find length of longest comment
    max_string_length = 0
    max_value_length = 0
    max_unit_length = 0
    for value, string, fmt, unit in results:
        if len(string) > max_string_length:
            max_string_length = len(string)
        if len(unit) > max_unit_length:
            max_unit_length = len(unit)
        if len(fmt % value) > max_value_length:
            max_value_length = len(fmt % value)

    # Print saved values
    print('\n\n')
    for i, (value, comment, fmt, unit) in enumerate(results):
        fmt = '%%%d' % max_value_length + fmt[1:]
        if i % 2 == 0:
            tc, bc = 'white', 'red'
        else:
            tc, bc = 'red', 'white'

        line = ('%%-%ds  :  ' % max_string_length + fmt +
                '  :  %%%ds' % max_unit_length)
        cprint(colored(line % (comment, value, unit), '%s' % tc, 'on_%s' % bc))
    print('\n\n')

