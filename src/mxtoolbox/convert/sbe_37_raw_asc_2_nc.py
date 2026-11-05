#!/usr/bin/env python
"""

Function sbe37tonetcdf:

    Summary:

        Mean for use with clean_sbe37.sh and uses the automatically
        generated file extension to determine variable columns. Extensions
        mean:

            t   :   temperature
            c   :   conductivity
            p   :   pressure
            s   :   salinity
            h   :   this column and the 3 next contain time

    Usage:

        $ clean_sbe37.sh filename.RAW
        $ sbe37tonetcdf.py filename.tcpsh stationname

    Options:

        -d  :   Minimum depth. Discard data shallower than this.
        -p  :   Approximate pressure in dBars, for salinity calculation from C

"""
import numpy as np
import xarray as xr
from gsw import SP_from_C
from libmx.physics.xri import seabird_init
from dateutil.parser import parse
import argparse

# Parse input arguments
parser  =   argparse.ArgumentParser()
parser.add_argument("fname")
parser.add_argument("sname")
parser.add_argument('-d','--mindep',
                    metavar='',type=float,
                    help='discard data shallower than this depth in meters')
parser.add_argument('-t','--maxtemp',
                    metavar='',type=float,
                    help='discard data hotter than this (dC)')
parser.add_argument('-s','--minsal',
                    metavar='',type=float,
                    help='discard data less salty than this (PSU)')
parser.add_argument('-p','--pressure',
                    metavar='',type=float,
                    help='pressure to use for salinity calculation (dbar)')
args    =   parser.parse_args()

# options
mindep = 0 if args.mindep == None else args.mindep
minsal = 0 if  args.minsal == None else args.minsal
maxtemp = 0 if args.maxtemp == None else args.maxtemp

fname   =   args.fname
save    =   '/'.join(fname.split('/')[:-1])
ext     =   fname.split('.')[-1]

# load time
tpos    =   ext.find('h')
day,mon,year,hms    =   np.loadtxt(fname,unpack=True,usecols=range(tpos,tpos+4),dtype='u8,|U32,u8,|U32')
time    =   np.array( [parse("%s %s %s %s" % (day[i],mon[i],year[i],hms[i])) for i in range(day.size)] )

# init ds
ds      =   seabird_init(time)

# get data
for i in range(len(ext)):
    if ext[i] != 'h':
        if ext[i] in 't': fmt = 'f8';   key =   'temp'
        if ext[i] in 'c': fmt = 'f8';   key =   'cond'
        if ext[i] in 'p': fmt = 'f8';   key =   'p'
        if ext[i] in 's': fmt = 'f8';   key =   'sal'

        ds[key].values =  np.loadtxt(fname,usecols=i,dtype=fmt)

# If salinity was not output by the device, calculate it
if 'c' in ext and 'p' in ext and not 's' in ext:
    print("A : %s" % fname)
    # Multiply by 10 for S/m  ->  mS/cm
    ds['sal'].values    =   SP_from_C(ds.cond.values*10,ds.temp.values,ds.p.values)
    ext += 's'
# If salinity was not output by the device and pressure unavailable
if 'c' in ext and not 'p' in ext and not 's' in ext:
    print("B : %s" % fname)
    # Multiply by 10 for S/m  ->  mS/cm
    ds['sal'].values    =   SP_from_C(ds.cond.values*10,ds.temp.values,args.pressure)
    ext += 's'

# quality control
print(minsal, mindep, maxtemp, ext)
if mindep != 0 and 'p' in ext:
    ds = ds.where(ds.p > mindep, drop=True)
if minsal != 0 and 's' in ext:
    print("QC by salinity")
    ds['sal'].values = ds.sal.where(ds.sal > minsal)
if maxtemp != 0 and 't' in ext:
    ds['temp'].values = ds.temp.where(ds.temp < maxtemp)

strt    =   str(ds.time.values[0])[:10]
stop    =   str(ds.time.values[-1])[:10]
ds.to_netcdf(save+"/"+args.sname+'_'+strt+'_'+stop+'_SBE37.nc')
