"""
Tools for developing new code
"""


class CLIArguments:
    """
    Mimic the output of `argparse.ArgumentParser.parse_args()`.

    Useful to develop command line scripts interactively. Keyword arguments
    become the attributes of the instance, as in `argparse.Namespace`.

    Examples
    --------
    >>> args = CLIArguments(input='file.nc', verbose=True)
    >>> args.input
    'file.nc'
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
