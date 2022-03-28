#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jeb@geus.dk


"""

sites=['SWC','CP1','SDM','NSE','NUK_U']

site='CP1'
# site='SDM'
# site='NSE'
site='SWC'
# site='NUK_U'

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
from datetime import datetime
import sys

# -------------------------------- chdir
if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS'
elif os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS'

os.chdir(base_path)

sys.path.append(base_path)
import WetBulb as wb

def time_info(t0,t1):
    duration=t1-t0
    s=duration.total_seconds()
    hours, remainder = divmod(s, 3600)
    msg=t0.strftime("%d %b %Hz")+' to '+t1.strftime("%d %b %Hz, ")+str(int(hours))+'h'
    # print(msg)
    return(msg)

def RH2SpecHum(RH, T, pres):
    # Note: RH[T<0] needs to be with regards to ice
    
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    es = 0.622
    
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    
    es_all = Es_Water.copy()
    es_all[T < 0] = Es_Ice[T < 0] 
    
    # specific humidity at saturation
    q_sat = es * es_all/(pres-(1-es)*es_all)

    # specific humidity
    q = RH * q_sat /100
    return q

def RH_ice2water(RH, T):
    # switch ALL timestep to with-regards-to-water
    RH = np.array(RH)
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    ind = T < 0
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    RH_out = RH.copy()
    
    # T_100 = 373.15
    # T_0 = 273.15
    # T = T +T_0
    # # GOFF-GRATCH 1945 equation
    #    # saturation vapour pressure above 0 C (hPa)
    # Es_Water = 10**(  -7.90298*(T_100/T - 1) + 5.02808 * np.log(T_100/T) 
    #     - 1.3816E-7 * (10**(11.344*(1-T/T_100))-1) 
    #     + 8.1328E-3*(10**(-3.49149*(T_100/T-1)) -1.) + np.log(1013.246) )
    # # saturation vapour pressure below 0 C (hPa)
    # Es_Ice = 10**(  -9.09718 * (T_0 / T - 1.) - 3.56654 * np.log(T_0 / T) + 
    #              0.876793 * (1. - T / T_0) + np.log(6.1071)  )   
    
    RH_out[ind] = RH[ind] / Es_Water[ind]*Es_Ice[ind] 

    return RH_out


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
th=1 
font_size=18
# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams["font.size"] = font_size
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"
plt.rcParams["legend.facecolor"] ='w'
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['grid.linewidth'] = th
plt.rcParams['axes.linewidth'] = th #set the value globally
plt.rcParams['figure.figsize'] = 17, 10
plt.rcParams["legend.framealpha"] = 0.8
plt.rcParams['figure.figsize'] = 5, 4

font_size=20


# for site_index,site in enumerate(sites):
#     # if site_index!=10:
#     # if site=='SWC':
#     if site=='CP1':
#     # if site=='SDM':
#     # if site=='NSE':

if site=='SWC':
    a_or_b='(a)'
    site2='Swiss Camp'
    IMEI='300534060524310' ; lat=69+33.21479/60 ;lon=-(49+22.297377/60); elev=1148
    swd_coef=12.00 ; swu_coef=18.00
    t0_pre_rain=datetime(2021, 8, 9) ; t1_pre_rain=datetime(2021, 8, 11)
    t0_post_rain=datetime(2021, 8, 15) ; t1_post_rain=datetime(2021, 8, 19)
    t0=datetime(2021, 8, 10) ; t1=datetime(2021, 8, 24)

if site=='NUK_U':
    a_or_b='(a)'
    site2='NUK_U'
    xtit='UTC time, year 2021'
    t0=datetime(2021, 8, 9,23) ; t1=datetime(2021, 8, 21)
    t0=datetime(2021, 8, 7) ; t1=datetime(2021, 8, 22)
    t0_pre_rain=datetime(2021, 8, 9) ; t1_pre_rain=datetime(2021, 8, 11)
    t0_post_rain=datetime(2021, 8, 15) ; t1_post_rain=datetime(2021, 8, 19)        
    swd_coef=1 ; swu_coef=1
    
if site=='SDM':
    a_or_b='(b)'
    IMEI='300534060529220' ; site2='South Dome' ; lat=63.14889	;lon=-44.81667; elev=2895
    t0_pre_rain=datetime(2021, 8, 11) ; t1_pre_rain=datetime(2021, 8, 12)
    t0_post_rain=datetime(2021, 8, 15) ; t1_post_rain=datetime(2021, 8, 19)
    t0=datetime(2021, 8, 11,6) ; t1=datetime(2021, 8, 20,18)
    # t0=datetime(2021, 8, 6,12) ; t1=datetime(2021, 8, 30,18)
    # t0=datetime(2021, 6, 21,20) ; t1=datetime(2021, 8, 30,18)
    swd_coef=14.75 ; swu_coef=14.44

if site=='NSE':
    a_or_b='(b)'
    IMEI='300534062416350' ; site2='NASA-SE' ; lat=66.2866629	;lon=-42.2961828; elev=2386
    swd_coef=12.66 ; swu_coef=13.9
    t0_pre_rain=datetime(2021, 8, 12,12) ; t1_pre_rain=datetime(2021, 8, 13,12)
    t0_post_rain=datetime(2021, 8, 15,12) ; t1_post_rain=datetime(2021, 8, 16)
    t0=datetime(2021, 8, 13,6) ; t1=datetime(2021, 8, 16,11)
    swd_coef=12.66 ; swu_coef=13.90

if site=='CP1':
    a_or_b='(b)'
    IMEI='300534062418810' ; site2='Crawford Pt. (CP1)'; lat=69.8819; lon=-46.97358; elev=1958
    t0_pre_rain=datetime(2021, 8, 11) ; t1_pre_rain=datetime(2021, 8, 12)
    t0_post_rain=datetime(2021, 8, 15) ; t1_post_rain=datetime(2021, 8, 17)
    t0=datetime(2021, 8, 13) ; t1=datetime(2021, 8, 16)
    t0=datetime(2021, 8, 10) ; t1=datetime(2021, 8, 24)
    swd_coef=15.72 ; swu_coef=12.18

# if site=='CEN':
#     IMEI='300534062419950'; site2='Camp Century' #; lat=69.8819; lon=-46.97358; elev=1958
#     t0=datetime(2021, 8, 12,21) ; t1=datetime(2021, 8, 16,18)
#     t0=datetime(2021, 8, 12,23) ; t1=datetime(2021, 8, 20)
#     t0_pre_rain=datetime(2021, 8, 13) ; t1_pre_rain=datetime(2021, 8, 14)
#     t0_post_rain=datetime(2021, 8, 15) ; t1_post_rain=datetime(2021, 8, 17)

# if site=='NAU':
#     IMEI='300534062412950'; site2='NASA-U' #; lat=69.8819; lon=-46.97358; elev=1958
#     t0=datetime(2021, 8, 12,21) ; t1=datetime(2021, 8, 16,18)
#     t0_pre_rain=datetime(2021, 8, 18) ; t1_pre_rain=datetime(2021, 8, 19)
#     t0_post_rain=datetime(2021, 8, 18) ; t1_post_rain=datetime(2021, 8, 20)
#     t0=datetime(2021, 8, 18) ; t1=datetime(2021, 11, 10)

# if site=='NEM':
#     IMEI='300534062417910' ; site2='NEEM' ; lat=77.5022		;lon=-50.8744; elev=2454
#     t0=datetime(2021, 8, 13,14) ; t1=datetime(2021, 8, 18,12)
#     t0_pre_rain=datetime(2021, 8, 13) ; t1_pre_rain=datetime(2021, 8, 14)
#     t0_post_rain=datetime(2021, 8, 15) ; t1_post_rain=datetime(2021, 8, 17)
#     t0=datetime(2021, 8, 14) ; t1=datetime(2021, 8, 20)

# if site=='QAS_U':
#     site2='QAS_U'
#     xtit='UTC time, year 2021'

# -------------- copy of raw file names to an easier to understand format
# if ((site!='NUK_U')&(site!='QAS_U')):
#     fn='/Users/jason/Dropbox/AWS/GCNv2_xmit/AWS_'+IMEI+'.txt' 
#     fn='/Users/jason/Dropbox/AWS/xmit_latest/AWS_'+IMEI+'.txt' 
# else:
#     fn='/Users/jason/Dropbox/AWS/PROMICE/PROMICE_v04/'+site+'_hour_v04.txt'
# os.system('/bin/cp '+fn+' ./AWS_info/AWS_data/'+site+'.txt')

fn='./AWS_info/AWS_data/'+site+'.txt'
# cols=np.arange(0,49).astype(str)

cols=['time','counter','Pressure_L','Pressure_U','Asp_temp_L','Asp_temp_U','Humidity_L','Humidity_U','WindSpeed_L','WindDirection_L','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_L','SR_U','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','Rain_amount_L','Rain_amount_U','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']
varnamx=['time','counter','Pressure_L','Pressure_U','air temperature','air temperature','Humidity_L','Humidity_U','WindSpeed_L','WindDirection_L','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward\nSky Tempearure_effective','LW Upward','TemperatureRadSensor','SR_L','SR_U','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','rainfall','rainfall','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']
unitsx=['time','counter','Pressure_L','Pressure_U','deg. C','deg. C','Humidity_L','Humidity_U','WindSpeed_L','WindDirection_L','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_L','SR_U','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','mm','mm','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']


cols[0]='time'
skip=1
if site=='NSE':
    skip=3
    df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
if site=='NEM':
    df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
if site=='SDM':
    df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
if site=='NAU':
    skip=4
    df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
if site=='CP1':
    df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
# if site=='CEN':
#     skip=4
# cols=['time','counter','Pressure_L','Pressure_U','Asp_temp_L','Asp_temp_U','Humidity_L','Humidity_U','WindSpeed_L','WindDirection_L','WindSpeed_U','WindDirection_U','SWD','SWU','LW Downward','LW Upward','TemperatureRadSensor','SR_L','SR_U','T_firn_1','T_firn_2','T_firn_3','T_firn_4','T_firn_5','T_firn_6','T_firn_7','T_firn_8','T_firn_9','T_firn_10','T_firn_11','Roll','Pitch','Heading','Rain_amount_L','Rain_amount_U','counterx','Latitude','Longitude','Altitude','ss','Giodal','GeoUnit','Battery','NumberSatellites','HDOP','FanCurrent_L','FanCurrent_U','Quality','LoggerTemp']
#     df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
    
if ((site=='NUK_U')or(site=='QAS_U')):
    # df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
    df=pd.read_csv(fn, delim_whitespace=True)
    df.columns
    df[df == -999.0] = np.nan
    # df[df=="NAN"]=np.nan
    # for col in df.columns:
    #     df[col] = pd.to_numeric(df[col])

    df.rename({'AirPressure(hPa)': 'Pressure_L'}, axis=1, inplace=True)
    df.rename({'RelativeHumidity(%)': 'Humidity_L'}, axis=1, inplace=True)

    df.rename({'HeightStakes(m)': 'SR_L'}, axis=1, inplace=True)
    df.rename({'HeightSensorBoom(m)': 'SR_U'}, axis=1, inplace=True)
    df.rename({'AirTemperature(C)': 'Asp_temp_L'}, axis=1, inplace=True)
    df.rename({'Ac_Rain(mm)': 'Rain_amount_L'}, axis=1, inplace=True)

    df.rename({'LongwaveRadiationDown(W/m2)': 'LWD'}, axis=1, inplace=True)
    df.rename({'LongwaveRadiationUp(W/m2)': 'LWU'}, axis=1, inplace=True)
    df.rename({'ShortwaveRadiationDown_Cor(W/m2)': 'SWD'}, axis=1, inplace=True)
    df.rename({'ShortwaveRadiationUp_Cor(W/m2)': 'SWU'}, axis=1, inplace=True)
    df.rename({'ElevationGPS(m)': 'Altitude'}, axis=1, inplace=True)
    df.rename({'LongitudeGPS(degW)': 'Longitude'}, axis=1, inplace=True)
    df.rename({'LatitudeGPS(degN)': 'Latitude'}, axis=1, inplace=True)
    df.rename({'WindSpeed(m/s)': 'WindSpeed_L'}, axis=1, inplace=True)

    df.rename({'Year': 'year'}, axis=1, inplace=True)
    df.rename({'MonthOfYear': 'month'}, axis=1, inplace=True)
    df.rename({'DayOfMonth': 'day'}, axis=1, inplace=True)
    df.rename({'HourOfDay(UTC)': 'hour'}, axis=1, inplace=True)

    df["time"]=pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.index = pd.to_datetime(df.time)

    # print(df.columns)
    # print(df)
    
if site=='SWC':
    skip=4
    cols=['time','seconds_since_1990','Pressure_L','Asp_temp_L','Humidity_L','WindSpeed_L','winddirection_s_l',
          'SWU','SWD',
         'LW Downward','LW Upward','TemperatureRadSensor','SR_L','SR_U','solar?',
         'thermistorstring_1','thermistorstring_2','thermistorstring_3','thermistorstring_4','thermistorstring_5','thermistorstring_6','thermistorstring_7','thermistorstring_8',
         'roll','pitch','heading','Rain_amount_L','gpstime','Latitude','Longitude','Altitude','giodal','geounit?',
         'battvolt','?1','asp_temp_u','humidity_u','##','##2','##3']
    varnamx=['time','seconds_since_1990','pressure_l','Asp_temp_L','humidity_l','WindSpeed_L','winddirection_s_l','swupper','swlower',
         'lwupper','lwlower','temperatureradsensor','sr_l','sr_u','solar?',
         'thermistorstring_1','thermistorstring_2','thermistorstring_3','thermistorstring_4','thermistorstring_5','thermistorstring_6','thermistorstring_7','thermistorstring_8',
         'roll','pitch','heading','Rain_amount_L','gpstime','Latitude','Longitude','Altitude','giodal','geounit?',
         'battvolt','?1','asp_temp_u','humidity_u','##','##2','##3']
    unitsx=['time','seconds_since_1990','pressure_l','Asp_temp_L','humidity_l','WindSpeed_L','winddirection_s_l','swupper','swlower',
         'lwupper','lwlower','temperatureradsensor','sr_l','sr_u','solar?',
         'thermistorstring_1','thermistorstring_2','thermistorstring_3','thermistorstring_4','thermistorstring_5','thermistorstring_6','thermistorstring_7','thermistorstring_8',
         'roll','pitch','heading','mm','gpstime','Latitude','Longitude','Altitude','giodal','geounit?',
         'battvolt','?1','asp_temp_u','humidity_u','##','##2','##3']

    df=pd.read_csv(fn,header=None,names=cols,skiprows=skip)
    df.Latitude=df.Latitude.astype(float)

df[df=="NAN"]=np.nan

if site!='NUK_U':
    for col in cols[2:]:
        df[col] = pd.to_numeric(df[col])

if ((site=='SWC')or(site=='NUK_U')or(site=='QAS_U')):
    sensor_levels=['L']
else:
    sensor_levels=['L','U']

if ((site!='NUK_U')or(site!='QAS_U')):
    for sensor_level in sensor_levels:
        df['kk']=df['Asp_temp_'+sensor_level].astype(float)+273.15
        df['tr']=(df['kk']/273.15)**0.5
        # tr=tr**0.5
        df["SR_"+sensor_level]*=df['tr']

elev=np.nanmedian(df.Altitude.astype(float)[:-100])

if ((site!='NUK_U')&(site!='QAS_U')):
    df["date"]=pd.to_datetime(df.time)
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
        
    df["time"]=pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.index = pd.to_datetime(df.time)

# df = df.loc[df['time']>'2021-08-01',:] 
df = df.loc[df['time']<'2021-09-01',:] 

##%% albedo
N=len(df)
# ---------------------------------------------- compute albedo
df['ALB']=np.nan
plusminus=11
df['SWD']*=10
df['SWU']*=10
df['SWD']/=swd_coef
df['SWU']/=swu_coef
df['SRnet']=df['SWD']-df['SWU']
df['SWD'][df['SRnet']<0]=np.nan
df['SWU'][df['SRnet']<0]=np.nan
# plt.plot(df['SWD']-df['SWU'])
# plt.plot(df['SWD']-df['SWU'])

for i in range(0+plusminus,N-plusminus):
    df['ALB'][i]=np.nansum(df['SWU'][i-plusminus:i+plusminus])/np.nansum(df['SWD'][i-plusminus:i+plusminus])

# df['ALB'][df['ALB']>0.95]=np.nan
# df['ALB'][df['ALB']<0.3]=np.nan

##%% lat lon slow
df['Lat_decimal']=np.nan
df['Lon_decimal']=np.nan

df.Latitude[df.Latitude=='nan']=np.nan
df.Longitude[df.Longitude=='nan']=np.nan


for i in range(N):
    if np.isfinite(df.Latitude[i]):
        lat_min=(df.Latitude[i]/100-int(df.Latitude[i]/100))*100
        lon_min=(df.Longitude[i]/100-int(df.Longitude[i]/100))*100
        df['Lat_decimal'][i]=int(df.Latitude[i]/100)+lat_min/60
        df['Lon_decimal'][i]=int(df.Longitude[i]/100)+lon_min/60

v=df.Latitude[np.isfinite(df.Latitude)]
lat=np.nanmedian(df.Lat_decimal[:-100])
lon=-np.nanmedian(df.Lon_decimal[:-100])

if site=='NUK_U':
    lat=64.5104
    lon=-49.2710
    elev=1120
print(lat,lon,elev)

if site=='CP1':
    df.Asp_temp_L[0:1]=np.nan
    df.Rain_amount_U-=2.2

##%% rain
# ===================================== undercatch correction
# rain correction after Yang, D., Ishida, S., Goodison, B. E., and Gunther, T.: Bias correction of daily precipitation measurements for Greenland, https://doi.org/10.1029/1998jd200110, 1999.

do_plot=1
if do_plot: fig, ax = plt.subplots(figsize=(10,10))

sensor_levels=['L','U']
if ((site=='SWC')or(site=='NUK_U')):
    sensor_levels=['L']

for sensor_level in sensor_levels:
    df["Rain_amount_uncor"+sensor_level]=df["Rain_amount_"+sensor_level]#.astype(float)
    if do_plot: plt.plot(df["Rain_amount_uncor"+sensor_level][t0:t1],label='uncor '+sensor_level)

    df['k_shield_hellman']=100/(100.00-4.37*df["WindSpeed_"+sensor_level]+0.35*df["WindSpeed_"+sensor_level]*df["WindSpeed_"+sensor_level])
    df['k_shield_hellman'][df['k_shield_hellman']<1.02]=1.02

    # rain_rate=np.zeros(N)
    # rain_acc=np.zeros(N)
    df['rain_rate'+sensor_level]=np.nan
    df['rain_acc'+sensor_level]=np.nan
    temp=0.
    
    for k in range(N-1):
        df['rain_rate'+sensor_level][k]=df["Rain_amount_"+sensor_level][k+1]-df["Rain_amount_"+sensor_level][k]
        df['rain_rate'+sensor_level][k]=df['rain_rate'+sensor_level][k]*df['k_shield_hellman'][k]
        if np.isfinite(df['rain_rate'+sensor_level][k]):
            temp+=df['rain_rate'+sensor_level][k]
        df["Rain_amount_"+sensor_level][k]=temp
        df['rain_acc'+sensor_level][k]=temp
        # print(k,temp)
        # print(k,df["Rain_amount_"+sensor_level][k+1],df["Rain_amount_"+sensor_level][k],temp,df['k_shield_hellman'][k],rain_rate[k])

    # plt.plot(df["Rain_amount_"+sensor_level][t0:t1])
    # plt.plot(df['k_shield_hellman'])
    # plt.plot(df['rain_acc'+sensor_level])

#df.columns
##%%
if site!='NUK_U':
    df['LWD']=df['LW Downward']+5.67e-8*(df['TemperatureRadSensor']+273.15)**4
    df['LWU']=df['LW Upward']+5.67e-8*(df['TemperatureRadSensor']+273.15)**4


df['Ts']=(df.LWU/5.67e-8)**0.25-273.15

#plt.plot(df.LWD-df.LWU)
##%% test of rain parameterization after Charlampidis et al 2015

# df['LWD_BB']=5.67e-8*(df['TemperatureRadSensor']+273.15)**4

# sensor_level='L'
# v=df['rain_rate'+sensor_level]>0
# v=((df['rain_rate'+sensor_level]>0)&(df.LWD>312))
# v=((df['rain_rate'+sensor_level]>0.)&(df.LWD>300)) ; namxx='mm per W/m^2'
# v=((df['rain_rate'+sensor_level]>0.)&(df['Asp_temp_'+sensor_level]>-1)) ; namxx='mm/C'
# v=((df['rain_rate'+sensor_level]>0.)&(df.LWD>df.LWD_BB)) ; namxx='mm/C BB'
# x=df.LWD[v].values
# x=df['Asp_temp_'+sensor_level][v].values
# y=df['rain_rate'+sensor_level][v].values
# b, m = polyfit(x,y, 1)

# fig, ax = plt.subplots(figsize=(8,8))

# plt.scatter(x,y,label='rate = '+"{:.3f}".format(m)+namxx)
# plt.ylabel('rain, mm per h')
# # plt.xlabel('LWD')
# plt.xlabel('air T')
# plt.title(site)
# plt.legend()
# xx=[np.min(x),np.max(x)]
# plt.plot([xx[0],xx[1]], [b + m * xx[0],b + m * xx[1]], '--',linewidth=th*4,c='grey')


##%% Wet Bulb temperature 
sensor_levels=['L','U']
if ((site=='SWC')or(site=='NUK_U')):
    sensor_levels=['L']
for sensor_level in sensor_levels:

    Temperature = np.array(df['Asp_temp_'+sensor_level])
    Pressure = np.array(df['Pressure_'+sensor_level] / 0.01)  # hPa to Pa
    Humidity = np.array(df['Humidity_'+sensor_level])
    Wet_Bulb_Temperature, Equivalent_Temperature, Equivalent_Potential_Temperature \
        = wb.WetBulb(Temperature, Pressure, Humidity, HumidityMode=1)
        
    df['Tw'+sensor_level] = Wet_Bulb_Temperature
    # plt.plot( df['Tw'+sensor_level])
    # plt.plot(df['Tw'+sensor_level][10:]-df['Asp_temp_'+sensor_level][10:])
    # plt.scatter(df['Humidity_'+sensor_level][10:],df['Tw'+sensor_level][10:]-df['Asp_temp_'+sensor_level][10:])
    
#%% surface energy budget after Box and Steffen (2001)
do_SEB=0
if do_SEB:
    ##%% SEB
    
    p0=1000.
    kk=273.15
    k=0.4
    g=9.80665
    Cp=1005.
    Lv=2.501e6
    Rd=287.
    L_fusion=3.34e5
    sec_per_hour=3600
    ro_water=1000
    m2mm=1000
    cp_water=4186

    L_fusion=3.34e5
    cp_ice=2093
    water_tempearture=5 #deg. C
    density_snow=350.

    rain_temperature=1
    mm_of_rain_at_xC = 5 #mm
    
    mm_snow_melted_by_rain = cp_water * mm_of_rain_at_xC * rain_temperature / L_fusion
 
    print('mm_snow_melted_by_rain',mm_snow_melted_by_rain)
    print('L_fusion/cp_ice',L_fusion/cp_ice)
    n_hours=4
    print(1.5e6/(3600*n_hours))
    z2=2.7
    z1=1.5
    dz=z2-z1
    
    df['LRnet']=df['LWD']-df['LWU']
    df['SRnet']=df['SWD']-df['SWU']
    
    df['TA1']=df['Asp_temp_L']
    df['TA2']=df['Asp_temp_U']
    
    df['RH1']=df['Humidity_L']
    df['RH2']=df['Humidity_U']
    
    df['RH1_w'] = RH_ice2water(df['RH1'] ,df['TA1'])
    df['RH2_w'] = RH_ice2water(df['RH2'] ,df['TA2'])
    
    df['SpecHum1'] = RH2SpecHum(df['RH1'], df['TA1'], df['Pressure_L'] )*1000
    df['SpecHum2'] = RH2SpecHum(df['RH2'], df['TA2'], df['Pressure_L'] )*1000
    
    df['Tv1']=(df['TA1']+kk)*(1.+0.61*df['SpecHum1']/1000.)
    df['Tv2']=(df['TA2']+kk)*(1.+0.61*df['SpecHum2']/1000.)
    
    df['potTv1']=df.Tv1*(p0/df['Pressure_L'])**0.286
    df['potTv2']=df.Tv2*(p0/df['Pressure_L'])**0.286
    
    df['dU']=df['WindSpeed_U']-df['WindSpeed_L']
    
    df.dU[df.dU<=0]=np.nan
    df['Tv_bar_21']=(df.Tv1+df.Tv2)/2.
    
    df['Ri']=(g/df['Tv_bar_21'])*(df.potTv2-df.potTv1)/((df.dU/dz)**2)
    df['Ri'][df['Ri']<-0.4]=0.
    df['Ri'][df['Ri']>0.2]=0.2
    
    df['stab']=1.
    
    v=df.Ri>0
    df['stab'][v]=(1-5*df.Ri[v])**2
    v=df.Ri<0
    df['stab'][v]=(1-16*df.Ri[v])**0.75
    
    
    d_lnz_squared=(math.log(z2/z1))**2
    
    df['lv']=Lv-(2400*(df['TA1']))
    
    df['ro_air']=(df['Pressure_L']*100)/(Rd*(df['TA1']+kk))
    
    df['LHF']=df['ro_air']*df['lv']*0.16*((df.SpecHum2-df.SpecHum1)/1000)*df.dU/d_lnz_squared
    df['LHF']=df['LHF']*df['stab']
    # LHF=ro_air*Lv*k^2*dq21*du/d_lnz_squared
    
    # SHF=ro_air*Cp*k^2*dpotTv21*du/d_lnz_squared
    df['SHF']=df['ro_air']*Cp*0.16*(df['potTv2']-df['potTv1'])*df.dU/d_lnz_squared
    df['SHF']=df['SHF']*df['stab']
    df.LHF[df.LHF<-500]=np.nan
    df.SHF[df.SHF<-500]=np.nan
    
    df['Ts']=(df.LWU/5.67e-8)**0.25-kk
    
    df['QRTs'] =0.
    df['QR'] =0.
    v=df['Ts']>0
    df['QRTs'][v]=ro_water*cp_water*df['Ts'][v]*df['rain_rateL'][v]/m2mm/L_fusion
    v=df['TA1']>0
    df['QR'][v]=ro_water*cp_water*df['TwL'][v]*df['rain_rateL'][v]/3600/1000#/L_fusion
    print(np.nanmax(df.QR[t0:t1]))
    print(np.nanmean(df.QR[t0:t1]))
    print(np.nansum(df.QR[t0:t1])/m2mm)
    
    df['SEB']=df['SHF']+df['LHF']+df['SRnet']+df['LRnet']
    
    df["abl_from_EB"]=df["SEB"]/L_fusion*sec_per_hour
    
    ##%%
    fig, ax = plt.subplots(figsize=(10,10))
    
    t0=datetime(2021, 8, 12) ; t1=datetime(2021, 8, 21)
    t0=datetime(2021, 8, 13,9) ; t1=datetime(2021, 8, 16)
    
    # plt.plot(df['QRx'][t0:t1],label='Rain Melt T_rain=T_air')
    # plt.plot(df.SpecHum1)
    # plt.plot(df.SpecHum2-df.SpecHum1)
    # plt.plot(df.dU)
    # plt.plot(df.Tv2-df.Tv1)
    # plt.plot(df.potTv2-df.potTv1)
    plt.title(site)
    plt_SEB=1
    if plt_SEB:
        plt.plot(df.SEB[t0:t1],'k',label='melt energy')
        plt.plot(df.SHF[t0:t1],'r',label='SHF')
        plt.plot(df.LHF[t0:t1],'b',label='LHF')
        plt.plot(df.QR[t0:t1],'m',label='rainHF')
        plt.plot(df.SRnet[t0:t1],'darkorange',label='SRnet')
        plt.plot(df.LRnet[t0:t1],'c',label='LRnet')
        plt.plot(df['rain_rateL'][t0:t1]*100,drawstyle='steps',label='rainfall rate mm/h*100')
        plt.ylabel('W/m**2')
    else:
        # plt.plot(df.SHF[t0:t1],'r',label='SHF')
        # plt.plot(df.LHF[t0:t1],'b',label='LHF')
        plt.plot(df["abl_from_EB"][t0:t1],label='mm melt/h from energy budget')
        # plt.plot(df['QR'][t0:t1],label='Tw mm melt/h from rain')
        # plt.plot(df['QRTs'][t0:t1],label='Ts mm melt/h from rain')
        plt.plot(df['TA1'][t0:t1],'-o',label='air T, C')
        plt.plot(df['TwL'][t0:t1],label='air Tw, C')
        plt.plot(df['Ts'][t0:t1],label='Ts, C')
        plt.plot(df['rain_rateL'][t0:t1],drawstyle='steps',label='rainfall rate mm/h')
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Hh'))
    
    plt.legend()

#%%
    tot_rain=np.sum(df['rain_rateL'][t0:t1])
    tot_melt=np.sum(df['abl_from_EB'][t0:t1])
    tot_melt_from_rain=np.sum(df['QR'][t0:t1])/L_fusion*sec_per_hour
    print('tot_rain',"%.1f"%tot_rain+" mm")
    print('tot_melt',"%.1f"%tot_melt+" mm")
    print('tot_rain÷tot_melt',"%.2f"%(tot_rain/tot_melt)+"")
    print('tot_melt_from_rain',"%.1f"%tot_melt_from_rain+" mm")
    print('tot_melt_from_rain÷tot_melt',"%.3f"%(tot_melt_from_rain/tot_melt)+"")
    
        #%%
       
# if do_plot: plt.plot(df["Rain_amount_"+sensor_level][t0:t1],label='cor '+sensor_level)
        # plt.plot(df["WindSpeed_"+sensor_level][t0:t1])

# if do_plot: 
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )
#     plt.legend()


print('Alb pre rain')
alb0=np.nanmean(df['ALB'][t0_pre_rain:t1_pre_rain]) ; print(alb0)
print('Alb post rain')
alb1=np.nanmean(df['ALB'][t0_post_rain:t1_post_rain]) ; print(alb1)
print('Alb difference')
print(np.nanmean(df['ALB'][t0_post_rain:t1_post_rain])-np.nanmean(df['ALB'][t0_pre_rain:t1_pre_rain]))


if ((site=='SWC')or(site=='NUK_U')or(site=='QAS_U')):
    sensor_levels=['L']
else:
    sensor_levels=['L','U']

# ------------------------------------------------------------------------ rain ammount
rain_amounts=[]
for sensor_level in sensor_levels:
    # plt.plot(df["Rain_amount_"+sensor_level][t0:t1])
    # constrain August rain event
    v=np.where((df['year']==2021)&(df['month']==8)\
               &(df['day']>=13)&(np.isfinite(df["Rain_amount_"+sensor_level])))
    v=np.where((df['time']>t0)&(np.isfinite(df["Rain_amount_"+sensor_level])))
    # print(v)

    v0=v[0][0]
    v=np.where((df['year']==2021)&(df['month']==8)&\
               (df['day']<=20)&(np.isfinite(df["Rain_amount_"+sensor_level])))
    v1=v[0][-1:][0]
    
    # print('rain before after',df["Rain_amount_"+sensor_level][v1],df["Rain_amount_"+sensor_level][v0])
    rain=df["Rain_amount_"+sensor_level][v1]-df["Rain_amount_"+sensor_level][v0]
    
    rain_amounts.append(rain)
    print("Rain_amount_"+sensor_level,"{:.1f}".format(rain))

    df["Rain_amount_"+sensor_level]-=df["Rain_amount_"+sensor_level][v0]


print("average rain","{:.1f}".format(np.mean(rain_amounts)))

sensor_levels=['L','U']
if ((site=='SWC')or(site=='NUK_U')):
    sensor_levels=['L']
for sensor_level in sensor_levels:

    v=((df.time>t0)&(df.time<t1)&(df['Asp_temp_'+sensor_level].astype(float)>0))
    v=[i for i, x in enumerate(v) if x]
    print('T>0 start',sensor_level,df.time[v[0]].strftime("%d %b %H"),'h')
    print('T>0 end',sensor_level,df.time[v[-1]].strftime("%d %b %H"),'h')
    duration=df.time[v[-1]]-df.time[v[0]]
    s=duration.total_seconds()
    hours_melt, remainder = divmod(s, 3600)
    print('-------------',hours_melt/24.,'days',int(hours_melt),"hours")
    # hours, remainder = divmod(s, 3600)
    # minutes, seconds = divmod(remainder, 60)
    # print'{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
    # print(hours)
    v2=((df.time>t0)&(df.time<t1)&(df['Rain_amount_'+sensor_level]>0))
    v2=[i for i, x in enumerate(v2) if x]
    print('rain start',sensor_level,df.time[v2[0]].strftime("%d %b %H"))
    # print('T>0 start',sensor_level,df.time[v[0]].strftime("%d %b %H"),'h')
    # print('T>0 end',sensor_level,df.time[v[-1]].strftime("%d %b %H"),'h')
    duration=df.time[v[-1]]-df.time[v2[0]]
    s=duration.total_seconds()
    hours_rain, remainder = divmod(s, 3600)
    print('-------------',hours_rain/24.,'days',int(hours_rain),"hours")



#%% graphics for publication

x = df.time

plt.close()
n_rows=5
plt_wind=0
plot_RH=1
fig, ax = plt.subplots(n_rows,1,figsize=(10,18))
colors=['b','r','b','r']

# ---------------------------------------------------------------------------- rainfall
cc=0

mult=1.4
ax[cc].text(-0.13,1.13,a_or_b,transform=ax[0].transAxes, fontsize=font_size*mult,
    verticalalignment='top',rotation=0,color='k', rotation_mode="anchor")  

ax[0].set_title(site2+", "+"{:.0f}".format(elev)+' m, '+"{:.3f}".format(lat)+'°N, '+"{:.3f}".format(abs(lon))+'°W')

maxes=[]
nams=['Rain_amount_L','Rain_amount_U','Rain_amount_L_uncorrected','Rain_amount_U_uncorrected']
nams=['Rain_amount_L','Rain_amount_U']
linestyles=["-","-","-.","-."]

if ((site=='SWC')or(site=='NUK_U')or(site=='QAS_U')or(site=='NSE')):
    nams=['Rain_amount_L']

for i,varnam in enumerate(nams):
    # if i>2:
    # if i==5: #t upper
    # if i==14: # LW d
            
    var=df[str(varnam)]
    var-=var[v0]
    maxes.append(np.nanmax(var[v0:v1]))
    
    ax[cc].plot(var[t0:t1], linestyle=linestyles[i],drawstyle='steps',
            color=colors[i],linewidth=th*3,label='rain'+str(i+1),zorder=15)


color_rain='purple'

if site=='SWC':
    t0_rain=datetime(2021,8,13,19) ; t1_rain=datetime(2021,8,15,13)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2,label='rainfall')

if site=='CP1':
    t0_rain=datetime(2021,8,13,20) ; t1_rain=datetime(2021,8,15,16)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2,label='rainfall')
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

if site=='SDM':
    t0_rain=datetime(2021,8,15,6) ; t1_rain=datetime(2021,8,17,8)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2,label='rainfall')
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

if site=='NSE':
    t0_rain=datetime(2021,8,13,12) ; t1_rain=datetime(2021,8,13,22)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2,label='rainfall')
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)
    
    t0_rain=datetime(2021,8,14,16) ; t1_rain=datetime(2021,8,14,18)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2)
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

if site=='NUK_U':
    t0_rain=datetime(2021,8,14,3) ; t1_rain=datetime(2021,8,14,8)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2,label='rainfall')
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

    t0_rain=datetime(2021,8,15,1) ; t1_rain=datetime(2021,8,16,10)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2)
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

    t0_rain=datetime(2021,8,17,11) ; t1_rain=datetime(2021,8,18,8)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2)
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

    t0_rain=datetime(2021,8,18,18) ; t1_rain=datetime(2021,8,18,23)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2)
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

    t0_rain=datetime(2021,8,19,12) ; t1_rain=datetime(2021,8,20,0)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2)
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

ax[cc].get_xaxis().set_visible(True)
mult=1
# if site=='NSE':
#     ax[cc].legend(prop={'size': font_size*mult})
#     # rcParams["legend.framealpha"]
# else:
ax[cc].legend(prop={'size': font_size*mult}, facecolor=(1, 1, 1, 1))
    
# ax.set_ylabel(units)
ax[cc].set_xlim(t0,t1)

# ax2.set_ylim(0.2,0.8)
# if varnam=='Rain_amount_U': 
#     ax.set_ylim(y0,y1)
ax[0].set_ylim(0,np.nanmax(maxes)+0.5)

ax[cc].set_xlim(t0,t1)

ax[cc].set_ylabel('mm')

ax[cc].set_xlim(t0,t1)
# if site=='CP1':
#      ax.set_xlim(t0,t1)
# ax.set_title(varnam[j]+' at '+site+', '+str("%.0f"%elev)+' m elevation'
                   # )
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )
# plt.legend()
ax[cc].set_xticklabels([])

# ---------------------------------------------------------------------------- RH
if plot_RH:
    cc+=1
    sensor_levels=['L','U']
    if ((site=='SWC')or(site=='NUK_U')):
        sensor_levels=['L']
    
    for i,sensor_level in enumerate(sensor_levels):
        y=df['Humidity_'+sensor_level].astype(float)[t0:t1]
        xt=df.time[t0:t1]
        ax[cc].plot(xt,y,
                # drawstyle='steps',
                '-',color=colors[i],
                label='Humidity '+str(i+1),
                linewidth=th*2)#,label='air T, '+sensor_level)   
        ax[cc].fill_between(xt, y, where=(y > 98), color='b', alpha=.1)#,label='RH'+str(i+1)+'>=98%')
    
    ax[cc].axhline(y=0,linestyle='--',linewidth=th*1.5, color='grey')
    
    ax[cc].set_ylabel('Relative Humidity, %', color='k')
    # ax[cc].set_ylim(np.nanmin(y),np.nanmax(y))     
    ax[cc].set_ylim(np.nanmin(y),101)     
    ax[cc].get_xaxis().set_visible(True)
    # ax[cc].legend(prop={'size': font_size})
    ax[cc].legend(prop={'size': font_size*mult}, facecolor=(1, 1, 1, 1))
    ax[cc].set_xlim(t0,t1)
    ax[cc].set_xticklabels([])

# ---------------------------------------------------------------------------- Air and surface T
cc+=1

sensor_levels=['L','U']
if ((site=='SWC')or(site=='NUK_U')):
    sensor_levels=['L']

for i,sensor_level in enumerate(sensor_levels):
    y=df['Asp_temp_'+sensor_level].astype(float)[t0:t1]
    xt=df.time[t0:t1]
    ax[cc].plot(xt,y,
            # drawstyle='steps',
            '-',color=colors[i],
            label='air T '+str(i+1),
            linewidth=th*2)#,label='air T, '+sensor_level)   
    ax[cc].fill_between(xt, y, where=(y > 0), color='orange', alpha=.3)

# ax[cc].plot(df['Ts'][t0:t1],
#         '-',color='k',
#         linewidth=th*2,label='Ts')   

ax[cc].axhline(y=0,linestyle='--',linewidth=th*1.5, color='grey')

ax[cc].set_ylabel('air temperature, ° C', color='k')
ax[cc].legend(prop={'size': font_size*mult}, facecolor=(1, 1, 1, 1))
# ax[cc].set_ylim(np.nanmin(x),np.nanmax(x))     
ax[cc].get_xaxis().set_visible(True)
ax[cc].legend(prop={'size': font_size})
ax[cc].set_xlim(t0,t1)
ax[cc].set_xticklabels([])

# # ---------------------------------------------------------------------------- Rad
plt_rad=0
if plt_rad:
    cc+=1

    xt=df.time[t0:t1]
    
    ax[cc].plot(xt,(df['LWD']-df['LWU']).astype(float)[t0:t1],
            '-o',color='c',label='LWNet',
            linewidth=th*2)

    ax[cc].plot(xt,df['SRnet'].astype(float)[t0:t1]/10,
            '-o',color='darkorange',label='SWNet',
            linewidth=th*2)
    
    ax[cc].axhline(y=0,linestyle='--',linewidth=th*1.5, color='grey')
    
    ax[cc].set_ylabel('W m$^{-2}$', color='k')
    ax[cc].legend(prop={'size': font_size*mult})
    # ax[cc].set_ylim(np.nanmin(x),np.nanmax(x))     
    ax[cc].get_xaxis().set_visible(True)
    ax[cc].legend(loc=1,prop={'size': font_size}, facecolor=(1, 1, 1, 1))
    ax[cc].set_xlim(t0,t1)
    ax[cc].set_xticklabels([])
    
# # ---------------------------------------------------------------------------- wind
if plt_wind:
    cc+=1

    sensor_levels=['L','U']
    if ((site=='SWC')or(site=='NUK_U')):
        sensor_levels=['L']
    
    for i,sensor_level in enumerate(sensor_levels):
        y=df['WindSpeed_'+sensor_level].astype(float)[t0:t1]
        xt=df.time[t0:t1]
        ax[cc].plot(xt,y,
                # drawstyle='steps',
                '-o',color=colors[i],
                label='wind '+sensor_level,
                linewidth=th*2)#,label='air T, '+sensor_level)   
    
    ax[cc].axhline(y=0,linestyle='--',linewidth=th*1.5, color='grey')
    
    ax[cc].set_ylabel('wind speed', color='k')
    ax[cc].legend(prop={'size': font_size*mult})
    # ax[cc].set_ylim(np.nanmin(x),np.nanmax(x))     
    ax[cc].get_xaxis().set_visible(True)
    ax[cc].legend(prop={'size': font_size}, facecolor=(1, 1, 1, 1))
    ax[cc].set_xlim(t0,t1)
    ax[cc].set_xticklabels([])

# ---------------------------------------------------------------------------- albedo
cc+=1

# from matplotlib.patches import Rectangle
# import matplotlib.dates as mdates

ax[cc].plot(df['ALB'][t0:t1],
            drawstyle='steps',
            linewidth=th*2, color='k')#,label='albedo')   

ax[cc].set_ylabel('albedo', color='k')
# ax[cc].set_ylim(np.nanmin(x),np.nanmax(x))   
# print()
y0,y1=ax[cc].get_ylim()


color_snow='k'

if ((site=='SWC')or(site=='CP1')):
    t0_rain=datetime(2021,8,13,20) ; t1_rain=datetime(2021,8,15,16)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2,label='rainfall')

if site=='SDM':
    t0_rain=datetime(2021,8,15,6) ; t1_rain=datetime(2021,8,17,8)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2,label='rainfall')

    # t0_snow=datetime(2021,8,7,16) ; t1_snow=datetime(2021,8,8,4)
    # ax[cc].axvspan(t0_snow, t1_snow,color=color_snow, alpha=0.2,label='snowfall')

    t0_deflation=datetime(2021,8,12,21) ; t1_deflation=datetime(2021,8,17,20)
    ax[4].axvspan(t0_deflation, t1_deflation,color='grey', alpha=0.2,label='snow deflation')

    msg=time_info(t0_snow,t1_snow)
    print(site,'snow',msg)

if site=='NUK_U':


    t0_rain=datetime(2021,8,15,1) ; t1_rain=datetime(2021,8,16,10)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2)
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

    t0_rain=datetime(2021,8,17,11) ; t1_rain=datetime(2021,8,18,8)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2)
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

    t0_rain=datetime(2021,8,18,18) ; t1_rain=datetime(2021,8,18,23)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2)
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

    t0_rain=datetime(2021,8,19,12) ; t1_rain=datetime(2021,8,20,0)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2)
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)

    t0_snow=datetime(2021,8,7,23) ; t1_snow=datetime(2021,8,8,8)
    ax[cc].axvspan(t0_snow, t1_snow,color=color_snow, alpha=0.2,label='snowfall')
    
    msg=time_info(t0_snow,t1_snow)
    print(site,'snow',msg)

    t0_rain=datetime(2021,8,14,3) ; t1_rain=datetime(2021,8,14,8)
    ax[cc].axvspan(t0_rain, t1_rain,color=color_rain, alpha=0.2,label='rainfall')
    msg=time_info(t0_rain,t1_rain)
    print(site,'rain',msg)
    
if site=='CP1':
    t0_snow=datetime(2021,8,11,6) ; t1_snow=datetime(2021,8,13,6)
    ax[cc].axvspan(t0_snow, t1_snow,color='k', alpha=0.2,label='snowfall')


ax[cc].get_xaxis().set_visible(True)


legend = ax[cc].legend(prop={'size': font_size})
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 1))

ax[cc].set_xlim(t0,t1)
ax[cc].set_xticklabels([])
# ---------------------------------------------------------------------------- SR50
cc+=1

height_end=np.nan

# df.columns
x1=df["SR_L"][t0]-df["SR_L"]
SR50=x1
if site=='NUK_U':
    x1[((df.time<datetime(2021,8,9))&(x1<-0.04))]=np.nan
    SR50=x1
if site=='CP1':
    x1[x1>0.04]=np.nan


# if ((site!='SWC')&(site!='NUK_U')):
if ((site!='SWC')&(site!='NUK_U')):
    x2=df["SR_U"][t0]-df["SR_U"]
    if site=='CP1':
        x2[x2>0.04]=np.nan
    # ax[cc].plot(x2[t0:t1],
    #         drawstyle='steps',
    #         linewidth=th*2, color=colors[1],label='surface height 2')
    height_end=x2[t1]

if ((site!='SWC')and(site!='NUK_U')):
    SR50=(x1+x2)/2


# ax[cc].plot(x1[t0:t1],
ax[cc].plot(SR50[t0:t1],
            drawstyle='steps',
            linewidth=th*2, color=colors[0],label='surface height')  

if site=='NUK_U':
    SR50=x1
    xt=df.time[t0:t1]
    t0_snow_accum=datetime(2021,8,8,2) ; t1_snow_accum=datetime(2021,8,8,10)

    xt=df.time[t1_snow_accum:t1]
    ax[cc].fill_between(xt,SR50[t1_snow_accum:t1], where=(SR50[t1_snow_accum:t1] > 0), color='k', alpha=.1)#,label='snowpack')

    ax[cc].axvspan(t0_snow_accum, t1_snow_accum,color=color_rain, alpha=0.2)#,label='snow accumulation')
    
    ax[cc].plot(df['DepthPressureTransducer_Cor(m)'][t0:t1]-df['DepthPressureTransducer_Cor(m)'][t0:t1][0]+0.03,
            drawstyle='steps',
            linewidth=th*2, color=[0.8,0,0],label='ice ablation')

ax[cc].legend(prop={'size': font_size}, facecolor=(1, 1, 1, 1))

ax[cc].set_ylabel('m')
ax[cc].get_xaxis().set_visible(True)


ax[cc].set_xlim(t0,t1)

# ax[cc].set_ylim(0,10)
# ax[cc].set_yticks(np.arange(0, 12, 2))

ax[cc].set_xlim(t0,t1)
ax[cc].xaxis.set_major_locator(mdates.DayLocator(interval=1))   #to get a tick every 15 minutes
ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
xcolor='darkblue'
ax[cc].xaxis.label.set_color(xcolor)
ax[cc].tick_params(axis='x', colors=xcolor)
mult=0.8
ax[cc].text(-0.15,-0.02, "day of\nAug.'21",transform=ax[cc].transAxes, fontsize=font_size*mult,
    verticalalignment='top',rotation=0,color=xcolor, rotation_mode="anchor")  
plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )


if site=='NUK_U':
    props = dict(boxstyle='round', facecolor='w',edgecolor='lightgrey',linewidth=th,alpha=1)

    xx0=0.05    ;   yy0=0.45    ;   dy=0.06
    txt='    snow accumulation\n    snowpack'
    ax[cc].text(xx0, yy0,txt,color='k',
        transform=ax[cc].transAxes,#, fontsize=9
        verticalalignment='top', bbox=props,fontsize=font_size*1,zorder=10,linespacing=1.5)    ;   cc+=1

##%% end stuff

ly='p'
if ly == 'x':plt.show()

plt_logos=0

fig_path='./AWS_info/Figs/'
if ly == 'p':
    plt.savefig(fig_path+site+'.png', bbox_inches='tight', dpi=300)
    # if plt_eps:
    #     plt.savefig(fig_path+site+'_'+str(i).zfill(2)+nam+'.eps', bbox_inches='tight')

out_fn='./stats/'+site+'_event.csv'
out=open(out_fn,'w')
out.write('site,t0_melt,t1_melt,hours melt,t0rain,t1rain,hours rain,alb0,alb1,height change 1,height change 2\n')

out.write(site+\
          ','+str(df.time[v[0]])+','+str(df.time[v[-1]])+\
          ','+'{:.0f}'.format(hours_melt)+\
          ','+str(df.time[v2[0]])+','+str(df.time[v2[-1]])+\
          ','+'{:.0f}'.format(hours_rain)+\
          ','+'{:.2f}'.format(alb0)+',{:.2f}'.format(alb1)+\
          ','+'{:.2f}'.format(x1[t1])+',{:.2f}'.format(height_end)+\
        '\n')

out.close()

out_fn='./stats/'+site+'_event_hourly.csv'
out=open(out_fn,'w')
out.write('site,time,year,month,day,hour,doy,\
          T,rain,alb,height\n')


df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour
df['doy'] = df['time'].dt.dayofyear


v=((df.time>t0)&(df.time<t1))
v=[i for i, x in enumerate(v) if x]

for i in v:
    # print(i)
    out.write(site+\
              ','+str(df.time[i])+\
            ','+str(df['year'][i])+','+str(df['month'][i]).zfill(2)+','+str(df['day'][i]).zfill(2)+\
            ','+str(df['hour'][i]).zfill(2)+','+str(df['doy'][i])+\
            ','+'{:.2f}'.format(df['Asp_temp_L'][i])+\
            ','+'{:.1f}'.format(df['Rain_amount_L'][i])+\
            ','+'{:.3f}'.format(df['ALB'][i])+
            ','+'{:.3f}'.format(x1[i])+\
              
            '\n')

out.close()
