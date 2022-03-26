#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:33:42 2021

@author: jason
"""

ly='x' ;plt_map=0

import datetime
# import rasterio
# import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# from numpy.polynomial.polynomial import polyfit
import geopandas as gpd
from matplotlib.pyplot import figure
from PIL import Image
import matplotlib.dates as mdates
from pyproj import Proj, transform
from datetime import datetime

if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS/stats'
elif os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS/stats'

os.chdir(base_path)


sites=['SWC','CP1','NUK_U','SDM','NSE']

# for i,site in enumerate(sites):
cc=0
fn='/Users/jason/Dropbox/rain_optics_SICE_AWS/stats/'+sites[cc]+'_event.csv'
df1=pd.read_csv(fn)
cc+=1
fn='/Users/jason/Dropbox/rain_optics_SICE_AWS/stats/'+sites[cc]+'_event.csv'
df2=pd.read_csv(fn)
cc+=1
fn='/Users/jason/Dropbox/rain_optics_SICE_AWS/stats/'+sites[cc]+'_event.csv'
df3=pd.read_csv(fn)
cc+=1
fn='/Users/jason/Dropbox/rain_optics_SICE_AWS/stats/'+sites[cc]+'_event.csv'
df4=pd.read_csv(fn)
cc+=1
fn='/Users/jason/Dropbox/rain_optics_SICE_AWS/stats/'+sites[cc]+'_event.csv'
df5=pd.read_csv(fn)
cc+=1

frames = [df1, df2, df3, df4, df5]
df = pd.concat(frames)
print(df.columns)

df.reset_index(drop=True, inplace=True)

# cols=['']
# for col in cols[1:]:
#         df[col] = pd.to_numeric(df[col])

for i in range(len(df)):
    # print('melt',df.site[i],df.t0_melt[i],df['hours melt'][i])
    print('rain',df.site[i],df.t0rain[i],'hours rain',df['hours rain'][i],'delay',pd.to_datetime(df.t0rain[i])-pd.to_datetime(df.t0_melt[i]))
