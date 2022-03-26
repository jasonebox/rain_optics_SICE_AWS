#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:31:10 2021

@author: jeb@geus.dk

"""
import cdsapi
import os

if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS'
elif os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS'

os.chdir(base_path)


varnams=['rf','tp','sf']
j=0 # select rf or tp

choices=['turb_fluxes','tcwv','z']

choice_index=0
choice=choices[choice_index]


yearx=[]
monx=[]
dayx=[]
Gt=[]

for choice in choices[0:1]:
    year=2021
    month=8

    ofile='./ERA5/ERA5_6hourly_10-31_Aug_2021_'+choice+'.grib'

    c = cdsapi.Client()

    if choice=='turb_fluxes':
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'surface_latent_heat_flux', 'surface_sensible_heat_flux',
                ],                
                'year': str(year),
                'month': str(month).zfill(2),
                'day': [
                    '10','11','12','13','14','15','16','17','18','19','20',\
                    '21','22','23','24','25','26','27','28','29','30','31'
                            ],
                'time': [
                    # '00:00', '03:00', '06:00',
                    # '09:00', '12:00', '15:00',
                    # '18:00', '21:00',
                    '00:00', '06:00',
                    '12:00', '18:00',
                ],
                'format': 'grib',
            },
            ofile)

    if choice=='tcwv':
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': 'total_column_water_vapour',
                'year': str(year),
                'month': str(month).zfill(2),
                'day': [
                    '10','11','12','13','14','15','16','17','18','19','20',\
                    '21','22','23','24','25','26','27','28','29','30','31'
                            ],
                'time': [
                    # '00:00', '03:00', '06:00',
                    # '09:00', '12:00', '15:00',
                    # '18:00', '21:00',
                    '00:00', '06:00',
                    '12:00', '18:00',
                ],
                'format': 'grib',
            },
            ofile)

    if choice=='z':        
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'geopotential', 'u_component_of_wind', 'v_component_of_wind',
                ],        'pressure_level': '850',
                'year': str(year),
                'month': str(month).zfill(2),
                'day': [
                    '10','11','12','13','14','15','16','17','18','19','20',\
                    '21','22','23','24','25','26','27','28','29','30','31'                ],
                'time': [
                    '00:00', '06:00',
                    '12:00', '18:00',
                ],
                'format': 'grib',
            },
            ofile)
