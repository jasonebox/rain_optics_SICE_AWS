# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé and Jason Box, GEUS (Geological Survey of Denmark and Greenland)

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from PIL import Image
import matplotlib

if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_bare_ice_area'
if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS/SICE_bare_ice_area'
    
os.chdir(base_path)

# %% graphics parameters

font_size=24

params = {'legend.fontsize': font_size*0.8,
          # 'figure.figsize': (15, 5),
         'axes.labelsize': font_size,
         'axes.titlesize':font_size,
         'xtick.labelsize':font_size,
         'ytick.labelsize':font_size,
         "ytick.color" : "k",
          "xtick.color" : "k",
          "axes.labelcolor" : "k",
          "axes.edgecolor" : "k",
          'figure.facecolor':'w',
          'axes.grid': True
         }
plt.rcParams.update(params)

years = ['2017', '2018', '2019', '2020', '2021']
colors = ['purple','k', 'r', 'darkorange', 'b']

 # %% two variables are plotted and each has its own parameters

# varnams=['albedo','bare_ice_area']
varnams=['albedo','bare_ice_area_gris']
ytits=['albedo, unitless','bare ice area, km$^{2}$']
legend_locs=[0,2]
end_date=['2001-10-01','2001-09-15']
xoss=[1800,1800]
tit0=['Greenland ice sheet ','Greenland ']
tit1=['albedo','bare ice area']
units=['','km$^{2}$']

for j,varnam in enumerate(varnams):

    if j==1:
    
        plt.figure()
        fig, ax = plt.subplots(figsize=(14,10))
        
        for i, year in enumerate(years):
            
            annual_results = pd.read_csv(f'./bia_csv/SICE_BIA_Greenland_{year}_rawcumul.csv')
        
            datex=pd.to_datetime(annual_results.time)
            
            dt = pd.to_datetime(annual_results.time)
            
            dummy_datetime = pd.to_datetime(['2001' + d[4:] for d in annual_results.time])
        
            datex = pd.to_datetime(['2021' + d[4:] for d in annual_results.time])
            
            ax.plot(dummy_datetime, annual_results[varnam],'-o',
                    linewidth=3,
                    color=colors[i],
                    label=year)

            if year=='2021':
                temp_2021=annual_results[varnam]
                ax.scatter(dummy_datetime[-1:],
                           annual_results[varnam][-1:],s=200,
                           facecolors='none',
                           zorder=20,
                           edgecolors='b')

        # ax.set_title(tit0[j]+tit1[j]+' from Sentinel-3',fontsize=font_size*1.2)
        if j==1:
            ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, 
                                                p: format(int(x), ',')))

        latest_day=str(datex[-1:])[16:26]
        mult=0.9
        xx0=0.99 ; yy0=0.98 ; dy=-0.08 ; cc=0
        
        # 2021 heat and rain wave 
        plt.axvspan(pd.to_datetime('2001-08-09'), pd.to_datetime('2001-08-13-11'),
                    color='b', alpha=0.5,label='cool period\nafter snow')

        
        plt.axvspan(pd.to_datetime('2001-08-13-11'), pd.to_datetime('2001-08-14-19'),
                    color='r', alpha=0.5,label='heatwave')

        plt.axvspan(pd.to_datetime('2001-08-14-19'), pd.to_datetime('2001-08-20-00'), 
                    color='g', alpha=0.5,label='heat and clouds\ndissipating')

        plt.axvspan(pd.to_datetime('2001-08-20-00'), pd.to_datetime('2001-08-27-00'), 
                    color='m', alpha=0.5,label='albedo and latent\nfeedbacks')

        plt.axvspan(pd.to_datetime('2001-08-27'), pd.to_datetime('2001-09-01'), 
                    color='gray', alpha=0.5,label='ablation\nseason end')


        # ax.legend(loc=legend_locs[j])
        mult=0.7
        ax.legend(prop={'size': font_size*mult},loc=legend_locs[j])

        ax.set_ylabel(ytits[j], fontsize=font_size)
        # ax.set_xlabel('Time, day-month', fontsize=font_size)
        
        myFmt = mdates.DateFormatter('%b %d')
        ax.xaxis.set_major_formatter(myFmt)
        
        # ax.set_xlim(pd.to_datetime('2001-06-11'), pd.to_datetime(end_date[j]))
        ax.set_xlim(pd.to_datetime('2001-06-1'), pd.to_datetime('2001-09-20'))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )

        
        if j==0:
            lastval='{:.3f}'.format(annual_results[varnam][-1:].values[0])
        else:
            lastval='{:.0f}'.format(annual_results[varnam][-1:].values[0])

        # plt.text(xx0, yy0+cc*dy,lastval+' '+units[j]+' on\n'+latest_day, 
        #          fontsize=font_size*mult,transform=ax.transAxes, 
        #          color='b',va='top',ha='right') ; cc+=1. 
        
        ly = 'x'
        
        figpath='./figures/'

        if ly == 'p':
            os.system('mkdir -p '+figpath)
                            
            figx,figy=16,12
            # figx,figy=16*2,12*2
            plt.savefig(figpath+'bare_ice_area_rawcumul.png',
                        bbox_inches = 'tight',figsize = (figx,figy),type="png", dpi = 250, facecolor='w')
        
            # im1 = Image.open('/tmp/tmp.png', 'r')
            # width, height = im1.size
            # border=20
            # # Setting the points for cropped image
            # left = border
            # top = border
            # right = width-border
            # bottom = height-border
              
            # # Cropped image of above dimension
            # im1 = im1.crop((left, top, right, bottom))
            # back_im = im1.copy()
        
            # yy0=1290 ; xos=xoss[j]
            # fn='./mapping_daily/ancil/SICE Logo.png'
            # SICE_logo = Image.open(fn, 'r')
            # pixelsx=240 ; size = pixelsx, pixelsx
            # SICE_logo.thumbnail(size, Image.ANTIALIAS)
            # back_im.paste(SICE_logo,(xos, yy0))#), mask=SICE_logo)
            
            # # yy0=1000 ; xos=350
            # fn='./mapping_daily/ancil/PTEP_logo.png'
            # pixelsx=280 ; size = pixelsx, pixelsx
            # PTEP_logo = Image.open(fn, 'r')
            # PTEP_logo.thumbnail(size, Image.ANTIALIAS)
            # back_im.paste(PTEP_logo,(xos+300, yy0+60), mask=PTEP_logo)
        
            # ofile=figpath+'dayplot_latest_'+varnam+'.png'
            # size = 1080,1080
            # back_im.thumbnail(size, Image.ANTIALIAS)
            # back_im.save(ofile,optimize=True,quality=95)
        
            # os.system('mkdir -p /sice-s3/NRT/Greenland/dayplot/')
            # os.system('/bin/cp '+ofile+' /sice-s3/NRT/Greenland/dayplot/dayplot_latest_'+varnam+'.png')
            # # os.system('open '+ofile)
       #%%
# from datetime import datetime


# plt.figure()
# fig, ax = plt.subplots(figsize=(14,10))

# ax.plot(dummy_datetime,temp_2021,'-o',c='b')

# plt.axvspan(pd.to_datetime('2001-08-13-11'), pd.to_datetime('2001-08-14-18'), 
#             color='r', alpha=0.5,label='heatwave with rain')

# plt.axvspan(pd.to_datetime('2001-08-14-18'), pd.to_datetime('2001-08-20-10'), 
#             color='g', alpha=0.5,label='heat & clouds\ndissipating')

# plt.axvspan(pd.to_datetime('2001-08-20-10'), pd.to_datetime('2001-08-27-18'), 
#             color='m', alpha=0.5,label='albedo and latent\nfeedbacks')

# # plt.axvspan(pd.to_datetime('2001-08-19'), pd.to_datetime('2001-08-20'), 
# #             color='k', alpha=0.5,label='day')

# # t0=datetime(2021, 8, 7) ; t1=datetime(2021, 8, 31,12)

# ax.set_xlim(pd.to_datetime('2001-08-07'), pd.to_datetime('2001-08-31-12'))

# plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )

#%%
# df=pd.DataFrame()
# df['21']=pd.Series(temp_2021)
# df["time"]=pd.to_datetime(dummy_datetime)

# df.index = pd.to_datetime(df.time)
# df
# t0=datetime(2001, 8, 20,10) ; t1=datetime(2001, 8, 27,18)
# y=df['21'][t0:t1].values

# import numpy as np
# from numpy.polynomial.polynomial import polyfit
# x=np.arange(len(y))
# b, m = polyfit(x, y, 1)

# print(y[0],y[-1],m,m/((y[0]+y[-1])/2))
# print('change in BIA under feedback phase',(y[-1]-y[0])/(y[0]+y[-1])/2)
# # print(temp_2021[((dummy_datetime>=pd.to_datetime(2021, 8, 20,10))&(dummy_datetime<=pd.to_datetime(2021, 8, 27,18)))])
