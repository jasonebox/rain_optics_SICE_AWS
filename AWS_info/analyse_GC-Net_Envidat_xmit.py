#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:34:22 2021

@author: jeb@geus.dk

https://www.envidat.ch/data-api/gcnet/csv/summit/airtemp1,airtemp2,airtemp_cs500air1, airtemp_cs500air2,swin,swout,sh1,sh2/end/-999/2020-01-01/2021-09-20/

https://www.envidat.ch/data-api/gcnet/csv/humboldt/airtemp1,airtemp2,airtemp_cs500air1, airtemp_cs500air2,swin,swout,sh1,sh2/end/-999/2020-01-01/2021-09-20/

https://www.envidat.ch/data-api/gcnet/csv/petermann/airtemp1,airtemp2,airtemp_cs500air1, airtemp_cs500air2,swin,swout,sh1,sh2/end/-999/2020-01-01/2021-08-20/

https://www.envidat.ch/data-api/gcnet/csv/summit/airtemp1,airtemp2,airtemp_cs500air1, airtemp_cs500air2,swin,swout,sh1,sh2, windspeed1,windspeed2/end/-999/2020-01-01/2021-08-20/

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.dates as mdates

# -------------------------------- chdir
if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS'
elif os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS'

os.chdir(base_path)

font_size=22
th=1
# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = "grey"
plt.rcParams["font.size"] = font_size

sites=["gits","humboldt","petermann","tunu_n","swisscamp_10m_tower","swisscamp","crawfordpoint","nasa_u","summit","dye2","jar1","saddle","southdome","nasa_east","nasa_southeast","neem","east_grip"]

os.system('')
fn='/Users/jason/Dropbox/AWS/GCNet_Envidat/raw/summit_2021-06-01_2021-08-30.csv'; site='summit' ; site2='Summit'
a_or_b='(a)'
# fn='/Users/jason/Dropbox/AWS/GCNet_Envidat/raw/humboldt_2020-01-01_2021-10-20.csv' ; site='humboldt'
# fn='/Users/jason/Dropbox/AWS/GCNet_Envidat/raw/petermann_2020-01-01_2021-10-20.csv' ; site='petermann'
fn='/Users/jason/Dropbox/AWS/GCNet_Envidat/raw/dye2_2020-01-01_2021-10-20.csv' ; site='dye2' ; site2='DYE-2'

os.system('/bin/cp '+fn+' ./AWS_info/AWS_data/'+site+'.txt')

names=['timestamp_iso','TA1','TA2','TA1cs','TA2cs','SWD','SWU','SH1','SH2','U1','U2']
df=pd.read_csv(fn,skiprows=1,names=names)
# print(df.columns)
# df[df<-998]=np.nan

fn='/Users/jason/Dropbox/rain_optics_SICE_AWS/AWS_info/PROMICE_GC-Net_info.csv'
meta=pd.read_csv(fn)
print(meta.columns)
for col in meta.columns[2:5]:
    meta[col] = pd.to_numeric(meta[col])
print(meta)
v=np.where(meta.name==site2) ; v=v[0][0]

for nam in names[1:5]:
    df[nam][df[nam]<-998]=np.nan
    df[nam][df[nam]>10]=np.nan
    df[nam][df[nam]<-60]=np.nan
    
for nam in names[7:9]:
    df[nam][df[nam]<-998]=np.nan
    df[nam][df[nam]>10]=np.nan
    df[nam][df[nam]<-60]=np.nan

for i,nam in enumerate(names[7:9]):
    df[nam][df[nam]<-998]=np.nan
    df['kk']=df[nam].astype(float)+273.15
    df['tr']=(df['kk']/273.15)**0.5
    df['tr']=df['tr']**0.5
    df[nam]*=df['tr']
    # plt.plot(df[nams[i]])
    print(nam)

if site=='petermann':
    df.SH1[df.SH1<3]=np.nan
    df.SH1[df.SH1>4.5]=np.nan
# plt.plot(df.SH1)

N=len(df)
df['ALB']=np.nan
plusminus=13
for i in range(0+plusminus,N-plusminus):
    df['ALB'][i]=np.nansum(df['SWU'][i-plusminus:i+plusminus])/np.nansum(df['SWD'][i-plusminus:i+plusminus])

df['ALB'][df['ALB']>0.99]=np.nan
df['ALB'][df['ALB']<0.5]=np.nan
df['ALB']-=0.034

# plt.plot(df.SWD,'.')

# plt.plot(df.ALB,'.')


df["date"]=pd.to_datetime(df.timestamp_iso)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour

df["time"]=pd.to_datetime(df[['year', 'month', 'day', 'hour']])

df.index = pd.to_datetime(df.time)

# df

#%%

t0=datetime(2021, 8, 11) ; t1=datetime(2021, 8, 18)

if site=='humboldt':
    t0=datetime(2021, 8, 14,0) ; t1=datetime(2021, 8, 20,0)

if site=='dye2':
    t0=datetime(2021, 8, 14,0) ; t1=datetime(2021, 8, 31,0)

x = df.time


##%%
plt.close()
n_rows=4
fig, ax = plt.subplots(n_rows,1,figsize=(10,18))
# ---------------------------------------------------------------------------- rainfall
cc=0

pos="{:.0f}".format(meta.elev[v])+' m, '+"{:.3f}".format(meta.lat[v])+'°N, '+"{:.3f}".format(-meta.lon[v])+'°W'
print(pos)
ax[0].set_title(site2+", "+pos)

mult=1.4
ax[cc].text(-0.13,1.13,a_or_b,transform=ax[0].transAxes, fontsize=font_size*mult,
    verticalalignment='top',rotation=0,color='k', rotation_mode="anchor") 
# ---------------------------------------------------------------------------- SAT
for i,nam in enumerate(names[1:5]):
   print(np.sum(df[nam][t0:t1]>0))
   print(nam)
   if i==3:
       ax[cc].plot(df[nam][t0:t1],
        drawstyle='steps',
        linewidth=th*2,label='air temperature, ° C')   

       ax[cc].fill_between(df['time'][t0:t1],df[nam][t0:t1], where=(df[nam][t0:t1] > 0), color='r', alpha=1)

# ax[cc].set_ylabel('air temperature, ° C', color='k')
# ax[cc].set_ylim(np.nanmin(x),np.nanmax(x))     
ax[cc].get_xaxis().set_visible(True)
ax[cc].legend(prop={'size': font_size})
ax[cc].set_xlim(t0,t1)
ax[cc].axhline(y=0,linestyle='--',linewidth=th*1.5, color='grey')

ax[cc].set_xticklabels([])

# ---------------------------------------------------------------------------- wind
cc+=1

ax[cc].plot(df['U1'][t0:t1],
            drawstyle='steps',
            linewidth=th*2, color='k',label='wind speed, m/s')   

ax[cc].set_ylabel('', color='k')
# ax[cc].set_ylim(np.nanmin(x),np.nanmax(x))     
ax[cc].get_xaxis().set_visible(True)
ax[cc].legend(prop={'size': font_size})
ax[cc].set_xlim(t0,t1)
ax[cc].set_xticklabels([])

# ---------------------------------------------------------------------------- albedo
cc+=1

ax[cc].plot(df['ALB'][t0:t1],
            drawstyle='steps',
            linewidth=th*2, color='k',label='albedo')   

ax[cc].set_ylabel('', color='k')
# ax[cc].set_ylim(np.nanmin(x),np.nanmax(x))     
ax[cc].get_xaxis().set_visible(True)
ax[cc].legend(prop={'size': font_size})
ax[cc].set_xlim(t0,t1)
ax[cc].set_xticklabels([])
# ---------------------------------------------------------------------------- SR50
cc+=1


x1=df["SH1"][t0]-df["SH1"]
x2=df["SH2"][t0]-df["SH2"]

if site=='summit':
    x2[((df.time<datetime(2021,8,14))&(x2>0.05))]=np.nan
    x2[((df.time>datetime(2021,8,16))&(x2>0.14))]=np.nan
    x2[((df.time>datetime(2021,8,15,12))&(x2>0.15))]=np.nan
    x1[((df.time>datetime(2021,8,15))&(x1>0.1))]=np.nan
    
if site=='dye2':
    x2[x2>0.2]=np.nan

SR50=(x1+x2)/2

df.TA1[datetime(2021,8,14):datetime(2021,8,15)]
df.TA2[datetime(2021,8,14):datetime(2021,8,15)]
SR50[datetime(2021,8,14):datetime(2021,8,15)]
ax[cc].plot(SR50[t0:t1],
        drawstyle='steps',
        linewidth=th*2, color='purple',label='surface height, m')
# ax[cc].set_ylabel('m')
ax[cc].get_xaxis().set_visible(True)

mult=0.9
ax[cc].legend(prop={'size': font_size*mult})
ax[cc].set_xlim(t0,t1)
ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%d'))

ax[cc].set_xlim(t0,t1)

plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )

xcolor='darkblue'
ax[cc].xaxis.label.set_color(xcolor)
ax[cc].tick_params(axis='x', colors=xcolor)
mult=0.8
ax[cc].text(-0.15,0.0, "day of\nAug.'21",transform=ax[cc].transAxes, fontsize=font_size*mult,
    verticalalignment='top',rotation=0,color=xcolor, rotation_mode="anchor")  

ly='x'

if ly == 'x':plt.show()

plt_eps=0
fig_path='./AWS_info/Figs/'
if ly == 'p':
    plt.savefig(fig_path+site+'.png', bbox_inches='tight', dpi=250)
    if plt_eps:
        plt.savefig(fig_path+site+'_'+str(i).zfill(2)+nam+'.eps', bbox_inches='tight')

