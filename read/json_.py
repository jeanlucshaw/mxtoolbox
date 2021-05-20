import json
from types import SimpleNamespace


__all__ = ['jdump', 'jload']


def jload(name, namespace=True):
    """
    Load JSON file as namespace or dictionnary.

    Parameters
    ----------
    name: str
        Path and name of json file.
    namespace: bool
        Return contents as namespace or dictionnary.

    Returns
    -------
    SimpleNamespace or dict:
        The contents of the json file.

    """
    with open(name, 'r') as file_:
        if namespace:
            return SimpleNamespace(**json.load(file_))
        else:
            return json.load(file_)


def jdump(name, dictionnary):
    """
    Load JSON file as namespace.

    Parameters
    ----------
    name: str
        Path and name of json file.
    dictionnary: dict
        Contents to write to json file.

    """
    with open(name, 'w') as file_:
        json.dump(dictionnary, file_, indent=4)
