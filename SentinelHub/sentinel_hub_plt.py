# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

https://apps.sentinel-hub.com/eo-browser/?zoom=2&lat=66.99884&lng=-50.80078&themeId=DEFAULT-THEME&visualizationUrl=https%3A%2F%2Fservices.sentinel-hub.com%2Fogc%2Fwms%2Fbd86bcc0-f318-402b-a145-015f85b9427e&datasetId=S2L2A&fromTime=2021-05-02T00%3A00%3A00.000Z&toTime=2021-05-02T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR

"""

import os
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import SHConfig
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
     bbox_to_dimensions, DownloadRequest, DataCollection
# from pyproj import Proj, transform
# import matplotlib.gridspec as gridspec
import pandas as pd
import requests

font_size=8
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'

CLIENT_ID = '856c8767-5815-46ae-83d8-532d2bd3b4b5'
CLIENT_SECRET = 'wAu1BblYezV%^].?i,j*{/<suZ:@lWYauuXGxNF&' 
data_folder='C:/Users/Pascal/Desktop/UZH_2020/'

JB=1

if JB:
    CLIENT_ID='eb2052e3-23c9-49d5-9da4-05bbe71e333b'
    CLIENT_SECRET=',NpaHPOia%<h-fP{{M/B<kdg2:!%I;t[hs1>|BM1'
    data_folder='/Users/jason/0_dat/S2_hub/'
    

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



config = SHConfig()

resolution=23

if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

if config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")


evalscript = escript_true_color
save_data = False


location='Russian High Arctic'
lonx=90.9
latx=81.127
dx=1
dlat=0.1/dx
dlon=0.75/dx

datex='2020-08-18'
datex='2020-08-16'
datex='2020-08-14'
datex='2020-08-01'
datex='2020-07-06'

df=pd.read_excel('/Users/jason/Dropbox/GAC/GAC2020/GAC_2020v3.xlsx')

location='QAS'
# area including Sermilik and Nunatak to the NE
latx=61.08
lonx=-46.78


# area focused on Qaleraliq
location='Qaleraliq'
latx=61.035
lonx=-46.73

n=len(df)

for i,location in enumerate(df.Glacier):
    # if location=='Humboldt':
    # if location=='Ryder':
    # if location=='79 fjorden':
    # if location=='Academy':
    # if location=='Zachariae':
    # if location=='Qajuuttap':
    if location=='Jakobshavn':

        # if i==0: # sermiliq
        # if location=='Kangia Nunata Sermia':
    # if ((location[0:4]!='UpernavikA')&(location[0:4]!='UpernavikB')&(location[0:4]!='UpernavikD')&(location[0:4]!='UpernavikE av')&(location[0:4]!='Lille')&(i>=0)):
        yos=100
        dlat=0.15
        dlon=0.5
        x0=61 ; x1=81
        y0=0.3 ; y1=1
        m=((y1-y0)/(x1-x0))
        b=y0-m*x0
        xx=np.arange(0,x1,100)
        dlon=df.Latitude[i]*m+b
        if location=='79 fjorden':
            dlon*=1.4
            dlat=0.25
            yos=120
        if location=='Zachariae':
            dlon*=1.3
            dlat=0.2
        if location=='Humboldt':
            dlon*=1.2
            dlat=0.22
            yos=90
        if location=='Academy':
            dlon*=1.2
            dlat=0.22
            yos=90
        if location=='Ryder':
            dlon*=1.2
            dlat=0.22
            yos=90
        print(df.Glacier[i],df.Longitude[i],df.Latitude[i],dlon)
        latx=df.Latitude[i] ; lonx=df.Longitude[i]
        latx= 67.009100 ; lonx=-47.301900
        dx=1.

        dates=['2020-05-03','2020-05-04','2020-05-05','2021-05-03','2021-05-04','2021-05-05']
        # dates=['2021-05-03','2021-05-04','2021-05-05']
        dates=['2021-06-01','2021-06-02','2021-06-04','2021-06-04']
        dates=['2021-06-05','2021-06-06']
        
        dates=['2020-05-03','2020-05-05','2021-05-04','2021-05-05', '2021-06-01','2021-06-02','2021-06-04','2021-06-04','2021-06-05','2021-06-06','2021-06-20','2021-06-21','2021-06-22','2021-06-23','2021-06-24','2021-06-25']
        # dates=['2021-06-23'] # Hunboldt
        dates=['2021-06-26','2021-06-27','2021-06-28']
        dates=['2021-06-29']
        dates=['2021-06-30','2021-07-01']
        dates=['2021-07-07','2021-07-08','2021-07-09']
        dates=['2021-07-12','2021-07-13','2021-07-14']
        dates=['2021-07-01','2021-07-14']
        dates=['2021-07-15','2021-07-16','2021-07-17']
        dates=['2021-07-18','2021-07-19','2021-07-20','2021-07-21','2021-07-22','2021-07-23','2021-07-24','2021-07-25','2021-07-26','2021-07-27','2021-07-28']
        dates=['2021-07-29','2021-07-30','2021-07-31','2021-08-01','2021-08-02']
        dates=['2021-09-06']

        dates=['2021-08-13','2021-08-14','2021-08-15','2021-08-16','2021-08-17','2021-08-18','2021-08-19','2021-08-20']
        dates=['2021-08-21','2021-08-22','2021-08-23','2021-08-24','2021-08-25','2021-08-26','2021-08-27']
        dates=['2021-08-10','2021-08-11','2021-08-12','2021-08-09','2021-08-08','2021-08--7','2021-08-06','2021-08-05']

        # from datetime import date
        # import datetime
                
        # for j in range(4):
        #     today = date.today()
        #     yesterday = today - datetime.timedelta(days=j+4)
        #     print(str(yesterday))
        #     dates=[(str(yesterday))]
            # print(j,dates,yesterday)

# dates=['2020-05-03','2020-05-05','2021-05-05','2021-06-02','2021-06-04','2021-06-05','2021-06-06','2021-06-20','2021-06-22','2021-06-23','2021-06-24','2021-06-25']
# dates=['2021-06-20','2021-06-21','2021-06-22']
        # dates=['2021-03-09','2021-04-23','2021-05-01','2021-05-13','2021-06-02','2021-07-08','2021-09-01','2021-08-01','2021-10-05']
        # dates=['2015-08-16']
        
        # ====================================
        
        ly='p'
        
        # ====================================
        coords = [lonx-dlon,latx-dlat,lonx+dlon,latx+dlat] 
        
        for datex in dates:
            
            time_interval=(datex,datex)
            
            def sentinelhub_request(data_folder=data_folder, time_interval=time_interval, coords=coords,
                                    evalscript=evalscript, data_collection=DataCollection.SENTINEL2_L1C,
                                    save_data=False):
                
                loc_bbox = BBox(bbox=coords, crs=CRS.WGS84)
                loc_size = bbox_to_dimensions(loc_bbox, resolution=resolution)
                
                request_all_bands = SentinelHubRequest(
                    data_folder=data_folder,
                    evalscript=evalscript,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=data_collection,
                            time_interval=time_interval,
                            mosaicking_order='leastCC'
                    )],
                    responses=[
                        SentinelHubRequest.output_response('default', MimeType.TIFF)
                    ],
                    bbox=loc_bbox,
                    size=loc_size,
                    config=config
                )
                
                outputs = request_all_bands.get_data()
                
                if save_data:
                    outputs.save_data()
                    
                return outputs

            
            bd_s2 = sentinelhub_request(time_interval=(datex,datex))
            
            # fig, (ax, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1.],'wspace':0.0})
            plt.close()
            plt.figure()
            from matplotlib.pyplot import figure
            figure(figsize=(8, 6), dpi=200)
            a=bd_s2[0]
            plt.imshow(a / np.nanmax(a))
            plt.axis("off")
            
            x0=a.shape[1]-8 # right
            y0=a.shape[0]-35 # lowerr
            y0=260
            fs=12
            # plt.text(a.shape[1]-30,yos,location+'\n'+datex,ha='right',c='w',fontsize=fs)
            plt.text(a.shape[1]-30,yos,datex,ha='right',c='w',fontsize=fs)
            # x0=a.shape[1]-10
            # y0=a.shape[0]-14
            # x0=150
            # plt.text(x0,y0,'@climate_ice',ha='right',c='k', fontsize=fs*1.01,verticalalignment='center')
            # plt.text(x0,y0,'@climate_ice',ha='right',c='grey', fontsize=fs,verticalalignment='center')
            
            if ly == 'x':plt.show()
            
            if ly == 'p':
                figpath='/Users/jason/Dropbox/sentinelhubpy-tools/Figs/'
                DPIs=[120,200]
                for DPI in DPIs:
                    figpath='/Users/jason/0_dat/S2_hub/'+str(DPI)+'/'
                    os.system('mkdir -p '+figpath)
                    figname=figpath+location+datex+'.png'
                    figname=figpath+'runoff'+datex+'.png'
                    figname=figname.replace(' ','')
                    plt.savefig(figname, bbox_inches='tight', dpi=DPI)

                # print('size',os.stat(figname).st_size)
                # if os.stat(figname).st_size < 3e6:
                    # os.system('/bin/rm '+figname)
                # figname=figpath+site+'_'+varnam2[j]+'_update.eps'
                # plt.savefig(figname)
                # os.system('open '+figname)
                
                    
                    #%%
    for i,location in enumerate(df.Glacier):
        # if location=='Humboldt':
        # if location=='Ryder':
        # if location=='79 fjorden':
        # if location=='Academy':
        # if location=='Zachariae':
        # if location=='Sermilik av':
        if location=='Jakobshavn':

            # nam='2017-09-14_speed_orthographic'
            nam=location
            print("making gif")
            # animpath='/Users/jason/Dropbox/sentinelhubpy-tools/anim/'
            # os.system('mkdir -p '+animpath)
            DPIs=[120]
            for DPI in DPIs:
                figpath='/Users/jason/0_dat/S2_hub/'+str(DPI)+'/'
                inpath=figpath
                animpath=figpath
                msg='convert  -delay 40  -loop 0   '+inpath+location[0:5]+'*.png  '+animpath+location[0:5]+'.gif'
                # msg='ls -l '+inpath+location+'*.png'
                # os.system(msg)
                os.system(msg)