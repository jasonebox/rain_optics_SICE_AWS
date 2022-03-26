# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""

import os
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import SHConfig
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
     bbox_to_dimensions, DownloadRequest, DataCollection, WmsRequest

import requests
import matplotlib.animation as animation

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import math
from matplotlib.patches import Rectangle
import glob
import shutil
import json
import argparse
import rasterio

# %% 

CLIENT_ID = '856c8767-5815-46ae-83d8-532d2bd3b4b5'
CLIENT_SECRET = 'wAu1BblYezV%^].?i,j*{/<suZ:@lWYauuXGxNF&' 

config = SHConfig()

config.instance_id = "8a4a0b70-fac5-4f52-be04-3a34230ded19"

link1 = "https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/active_fire_detection/script.js"
f1 = requests.get(link1)
evalscript_active_fire=f1.text

link2='https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/markuse_fire/script.js'
f2=requests.get(link2)
evalscript_wildfire_viz=f2.text

escript_true_color = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ['B04','B03', 'B02'],
                    units: "DN"
                }],
                output: {
                    bands: 3,
                    sampleType: "FLOAT32"
                }
            };
        }
    
        function evaluatePixel(ds) {

            var R = ds.B04
            var G = ds.B03
            var B = ds.B02

            return [R, G, B];
        }
    """

resolution = 23

if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

if config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")


def sentinelhub_request(time_interval, coords, evalscript, data_folder, 
                        save_data=False):
    
    loc_bbox = BBox(bbox=coords, crs=CRS.WGS84)
    loc_size = bbox_to_dimensions(loc_bbox, resolution=resolution)
    
    request_all_bands = SentinelHubRequest(
        data_folder=data_folder,
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_interval
        )],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=loc_bbox,
        size=loc_size,
        config=config
    )
        
    outputs = request_all_bands.get_data()
        
    
    if not np.isfinite(np.nanmean(outputs)):
        print(f'{time_interval} not available')
        return None
    
    else:
        print(f'{time_interval} available')
        if save_data:
            request_all_bands.save_data()
        
    return outputs

# %% Sentinel-1 download HV

if os.getlogin() == 'adrien':
    data_folder = '/home/adrien/Downloads/2021-heatwave/'
    
# make working directory if does not exist
if not  os.path.exists(data_folder):
  os.makedirs(data_folder)


date_range = pd.to_datetime(['2021-08-10', '2021-08-11', '2021-08-12', '2021-08-09', 
                        '2021-08-08', '2021-08-07', '2021-08-06', '2021-08-05'])

latx = 67.009100
lonx = -47.301900
dx = 1
dlat = 0.15
dlon = 0.5

coords = [lonx-dlon,latx-dlat,lonx+dlon,latx+dlat] 

for i, date in enumerate(date_range):
  date_str = date.strftime('%Y-%m-%d')
  im = sentinelhub_request(time_interval=(date_str, date_str), 
                           coords=coords, evalscript=escript_true_color,
                           data_folder=data_folder, save_data=True)
    
# %% rename files with dates

folders = glob.glob(f'{data_folder}/*/')

for folder in folders:
    
    with open(f'{folder}/request.json') as json_file:
        request = json.load(json_file)
        
    date = request['payload']['input']['data'][0]['dataFilter']['timeRange']\
        ['from'][:10]
    
    os.rename(f'{folder}/response.tiff', f'{data_folder}/{date}.tiff')
    
# %% remove temporary folders

folders_to_delete = [f'{data_folder}/{name}' for name in os.listdir(data_folder) 
                     if os.path.isdir(os.path.join(data_folder, name))]

for folder in folders_to_delete:
    
    shutil.rmtree(folder)
    

# %% get coordinates of stations

station_info =  pd.read_csv('/home/adrien/EO-IO/rain_optics_SICE_AWS/AWS_info/'
                            + 'PROMICE_GC-Net_info.csv')
station_info.index = station_info.name

lon_kan_u = station_info.loc['KAN_U'].lon
lat_kan_u = station_info.loc['KAN_U'].lat

lon_kan_m = station_info.loc['KAN_M'].lon
lat_kan_m = station_info.loc['KAN_M'].lat

# %% station positions on raster

ex = rasterio.open(f'{data_folder}2021-08-05.tiff')
    
kan_u_row, kan_u_col = rasterio.transform.rowcol(ex.transform, lon_kan_u, lat_kan_u)
kan_m_row, kan_m_col = rasterio.transform.rowcol(ex.transform, lon_kan_m, lat_kan_m)


# %% plot raster + station position

ex_img = np.array([ex.read(1), ex.read(2), ex.read(3)]).swapaxes(0, 2)

plt.figure()
plt.imshow(ex_img / np.nanmax(ex_img))
plt.scatter(kan_u_row, kan_u_col, marker="^", color='orange', s=80)
plt.scatter(kan_m_row, kan_m_col, marker="^", color='purple', s=80)
    