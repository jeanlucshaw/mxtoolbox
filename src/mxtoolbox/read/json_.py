import json
from types import SimpleNamespace


__all__ = ['jdump', 'jload']


def jload(name, namespace=True, keytype=None):
    """
    Load JSON file as namespace or dictionnary.

    Parameters
    ----------
    name: str
        Path and name of json file.
    namespace: bool
        Return contents as namespace or dictionnary.
    keytype: None or Type
        Convert keys of output to this type. Only active with `namespace` False.

    Returns
    -------
    SimpleNamespace or dict:
        The contents of the json file.

    """
    with open(name, 'r') as file_:
        if namespace:
            return SimpleNamespace(**json.load(file_))
        else:
            output = json.load(file_)
            if keytype is not None:
                output = {keytype(k_): v_ for k_, v_ in output.items()}
            return output


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
