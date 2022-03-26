#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zurich, Switzerland

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_retrievals/data/'
                   + 'SICE_BIA_Greenland_2021.csv')

data.index = pd.to_datetime(data.time)

sub_data = data['2021-08-11': '2021-08-22']

print((np.nanmax(sub_data.bare_ice_area_gris) 
       - np.nanmin(sub_data.bare_ice_area_gris)) 
      / np.nanmax(data.bare_ice_area_gris))

# %% compare to other years 

years = np.arange(2017, 2022).astype(str)

plt.figure() 

for year in years:
    
    annual_data = pd.read_csv('/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_retrievals/data/'
                   + f'SICE_BIA_Greenland_{year}.csv')
    
    plt.plot(pd.to_datetime(annual_data.time), 
             annual_data.bare_ice_area_gris, 'o-')
    
    # plt.plot(pd.to_datetime(annual_data.time)[:-1], 
    #           np.diff(annual_data.bare_ice_area_gris), 'o-')
    
    # plt.plot(pd.to_datetime(annual_data.time), 
    #          annual_data.bare_ice_area_gris)
    
    print(np.nanmax(np.diff(annual_data.bare_ice_area_gris)) / 
          np.nanmax(annual_data.bare_ice_area_gris))