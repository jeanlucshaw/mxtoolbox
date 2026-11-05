# mxtoolbox

Python code for time series analyses and physical oceanography: readers and
converters (`read`, `convert`), numerical tools (`process`) and plotting
helpers (`plot`).

## Install

```
uv pip install -e /opt/mxtoolbox          # or: pip install -e /opt/mxtoolbox
```

Requires Python 3.11 or newer. The colormap files `TsatPalette` and
`DivergentPalette` are package data, read with `importlib.resources`.
ADCP readers need the optional dependencies `tqdm`, `rti_python` and
`pycurrents` (the last one is not on PyPI).
