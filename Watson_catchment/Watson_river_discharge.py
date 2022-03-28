# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:59:31 2021

@author: Armin
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from datetime import datetime
# import datetime
import matplotlib.dates as mdates


base_path = 'C:/Users/Armin/Documents/Work/GEUS/Github/Watson_Discharge_analysis/'

if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/Watson_Discharge_analysis/'
    
os.chdir(base_path)

#common settings
fs=14

#read data
fn=base_path+'data/Watson River discharge daily (2006-2021).txt'
data=pd.read_csv(fn, delim_whitespace=True)
data[data==-999]=np.nan #change -999 to nan

#handle leap years
leap_years=[2008,2012,2016,2020]
for i in range(len(data)):
    if data.Year[i] in leap_years:
        data.DayOfYear[i]-=1

#data to plot
df=pd.DataFrame(data['WaterFluxDiversOnly(m3/s)'])
# #%%
# conv=24*3600/10e9 #convert from m3/s into km3 if daily data
# df*=conv
df['Year']=data.Year
df['DOY']=data.DayOfYear
df2=df[df['Year']==2010] ; df2.reset_index(drop=True, inplace=True)
df3=df[df['Year']==2012] ; df3.reset_index(drop=True, inplace=True)
df_2021=df[df['Year']==2021] ; df_2021.reset_index(drop=True, inplace=True)


df_2021['time']=pd.to_datetime(df_2021['DOY'], format='%j').dt.strftime('%Y-%m-%d %H')
df_2021['time'] = pd.to_datetime(df_2021.time) + pd.offsets.DateOffset(years=121)
df_2021['year']=pd.to_datetime(df_2021.time).dt.year
df_2021['month']=pd.to_datetime(df_2021.time).dt.month
df_2021['day']=pd.to_datetime(df_2021.time).dt.day

df_2021["time"]=pd.to_datetime(df_2021[['year', 'month', 'day']])
df_2021.index = pd.to_datetime(df_2021.time)

print(df_2021)


##%%
#make pivot table -> easy to plot
piv_all = pd.pivot_table(df, index=['DOY'],columns=['Year'], values=['WaterFluxDiversOnly(m3/s)'])
piv_2 = pd.pivot_table(df2, index=['DOY'],columns=['Year'], values=['WaterFluxDiversOnly(m3/s)'])
piv_3 = pd.pivot_table(df3, index=['DOY'],columns=['Year'], values=['WaterFluxDiversOnly(m3/s)'])
piv_4 = pd.pivot_table(df_2021, index=['DOY'],columns=['Year'], values=['WaterFluxDiversOnly(m3/s)'])
piv_mean = piv_all['mean']=piv_all.mean(axis=1)


#%% NAO plot


fn='/Users/jason/Dropbox/NAO/daily/nao.reanalysis.t10trunc.1948-present.txt'
dfNAO=pd.read_csv(fn, delim_whitespace=True,names=['year','month','day','NAO'])

dfNAO["time"]=pd.to_datetime(dfNAO[['year', 'month', 'day']])
dfNAO.index = pd.to_datetime(dfNAO.time)

dfNAO = dfNAO.loc[dfNAO['time']>='2021-01-01',:] 


th=1 
font_size=26
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

##%% plot NAO and discharge

t0=datetime(2021, 6, 1) ; t1=datetime(2021, 9, 15)


n_rows=2
fig, ax = plt.subplots(n_rows,1,figsize=(18,10))

cc=0


# plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )
# plt.legend()


ax[cc].plot(-dfNAO.NAO[t0:t1], linestyle='-',
            # drawstyle='steps',
        color='k',linewidth=th*3,label='NAO')


ax[cc].get_xaxis().set_visible(False)
mult=1
ax[cc].legend(prop={'size': font_size*mult})
    
ax[cc].set_xlim(t0,t1)

# ax[0].set_ylim(0,np.nanmax(maxes)+0.5)

ax[cc].set_ylabel('hPa')

ax[cc].set_xlim(t0,t1)

ax[cc].set_xticklabels([])

# ----------------------------------------------------- discharge
cc+=1
# df_2021=df[df['Year']==2021] ; df_2021.reset_index(drop=True, inplace=True)


# ax[cc].plot(x1[t0:t1],
ax[cc].plot(df_2021['WaterFluxDiversOnly(m3/s)'][t0:t1],
            # drawstyle='steps',
            # linewidth=th*2, color=colors[0],label='surface height')  
            linewidth=th*2, color='b',label='discharge')  


ax[cc].set_ylabel("discharge, $m^3 s^{-1}$")
ax[cc].get_xaxis().set_visible(True)

ax[cc].legend(prop={'size': font_size})
ax[cc].set_xlim(t0,t1)

# ax[cc].set_ylim(0,10)
# ax[cc].set_yticks(np.arange(0, 12, 2))

ax[cc].set_xlim(t0,t1)
ax[cc].xaxis.set_major_locator(mdates.DayLocator(interval=10))   #to get a tick every 15 minutes
ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
xcolor='darkblue'
ax[cc].xaxis.label.set_color(xcolor)
ax[cc].tick_params(axis='x', colors=xcolor)
mult=0.8
# ax[cc].text(-0.15,0.0, "day of\nAug.'21",transform=ax[cc].transAxes, fontsize=font_size*mult,
#     verticalalignment='top',rotation=0,color=xcolor, rotation_mode="anchor")  
plt.setp(ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='center' )

ly='x'
fig_path='./Figs/'
if ly == 'p':
    plt.savefig(fig_path+'NAO_vs_discharge_2021.png', bbox_inches='tight', dpi=250)
    # if plt_eps:
    #     plt.savefig(fig_path+site+'_'+str(i).zfill(2)+nam+'.eps', bbox_inches='tight')
#%% plot discharge

import datetime

font_size = 18

params = {
    "legend.fontsize": font_size * 0.8,
    # 'figure.figsize': (15, 5),
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "ytick.color": "k",
    "xtick.color": "k",
    "axes.labelcolor": "k",
    "axes.edgecolor": "k",
    "figure.facecolor": "w",
    "axes.grid": False,
    "legend.framealpha": 1,
}
plt.rcParams.update(params)


fig, ax = plt.subplots(figsize=(9,5))
plt.plot(piv_all, color='lightgrey',linewidth=2) #2006-2021 , label='2006-2021'
plt.plot(piv_2, color='darkorange') #2010 ,label='2010'
plt.plot(piv_3, color='r') #2012 ,label='2012'
plt.plot(piv_4, color='b') #2021  ,label='2021'

watson_2021=pd.DataFrame(columns=['doy','discharge'])
watson_2021.doy=piv_4.index.values
watson_2021.discharge=piv_4.values
watson_2021.to_csv('/Users/jason/Dropbox/rain_optics_SICE_AWS/Watson_catchment/Watson_river_discharge_2021.csv',index=None)

plt.plot(piv_mean, color='k')  #,label='average'
ax.set_ylabel("discharge, $m^3 s^{-1}$")
# ax.set_title('Watson River discharge')

# plot shaded area between certain dates
t0x=pd.to_datetime('2021-08-13')
t1x=pd.to_datetime('2021-08-27')
# ax.axvspan(t0x, t1x,color='grey', alpha=0.3,label='high melt')
# ax.axvspan(222, 232,color='grey', alpha=0.3) #label='high melt'

# len(df_2021['WaterFluxDiversOnly(m3/s)'])
# len(piv_mean)
# ax.fill_between(df.DOY, df_2021['WaterFluxDiversOnly(m3/s)'],piv_mean, color='b', alpha=.1)


# plt.axvline(x=238, color='grey', linestyle='--',linewidth=.7)
#monthlines
# months=[datetime.datetime(2021, 6, 1),datetime.datetime(2021, 7, 1),datetime.datetime(2021, 8, 1), datetime.datetime(2021, 9, 1)]
# month_doy = [months[i].timetuple().tm_yday for i in range(len(months))] # returns 1 for January 1st
# [plt.axvline(x=np.array(month_doy)[i], color='grey', linestyle='--',linewidth=.7) for i in range(len(month_doy))]
# mon=['June', 'July', 'August']
# [plt.text(np.array(month_doy)[i]+11, 2750, mon[i], color='grey') for i in range(len(month_doy))]

#legend
years = ['2010', '2012', '2021','2006-2021','average', 'high melt']
colors = ['darkorange', 'r','b', 'lightgrey', 'k', 'grey']
handles, labels = plt.gca().get_legend_handles_labels()
line1 = Line2D([0], [0], label=years[0], color=colors[0])
line2 = Line2D([0], [0], label=years[1], color=colors[1])
line3 = Line2D([0], [0], label=years[2], color=colors[2]) # 2021
line4 = Line2D([0], [0], label=years[3], color=colors[3])
line5 = Line2D([0], [0], label=years[4], color=colors[4])
# rect = mpatches.Patch(label=years[5], color=colors[5], alpha=0.3)
# handles.extend([line1, line2, line3, line4, line5, rect])
handles.extend([line1, line2, line3, line4, line5])
plt.legend(handles=handles, frameon=True)
# plt.legend()
#x-axis
start_date=datetime.datetime(2021, 5, 5)
end_date=datetime.datetime(2021, 10, 5)
ax.set_xlim(start_date.timetuple().tm_yday, end_date.timetuple().tm_yday)
ticks=[ 134, 151, 165, 181, 195, 212, 226, 243, 257, 273]
ax.set_xticks(ticks)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ly='x'
fig_path='./Figs/'
fig_path='/Users/jason/Dropbox/rain_optics_SICE_AWS/Watson_catchment/Figs/'
if ly == 'p':
    plt.savefig(fig_path+'Warson_discharge_2021.png', bbox_inches='tight', dpi=250)

#%%calculations

#define start/end of [heatwave, whole discharge season]
start=[datetime.datetime(2021, 8, 14), datetime.datetime(2021, 5, 1)]
end=[datetime.datetime(2021, 8, 30), datetime.datetime(2021, 10, 15)]

conv=24*3600/10e9 #convert from m3/s into km3

df_21=df_2021.copy()
df_21.reset_index(inplace=True)
xx=np.zeros((len(df_21)))*np.nan
xx[29:174]=np.array(piv_mean.to_frame()[0])  #two arrays have different length
df_21['mean']=xx #add mean do same dataframe 

for i in range(len(start)):
    #define doy of start/end
    start_doy=start[i].timetuple().tm_yday
    end_doy=end[i].timetuple().tm_yday
    
    #prepare data
    
    #filter time range
    dfnew=df_21.loc[df_21.DOY>=start_doy]
    dfnew=dfnew.loc[dfnew.DOY<=end_doy]
    
    #calc difference
    dis=dfnew.to_numpy()*conv
    dfnew['diff']=dis[:,1]-dis[:,4]
    # delta=(len(dfnew)-dfnew['WaterFluxDiversOnly(m3/s)'].isna().sum()) #days with measures
        
    if i == 0:
        mass_aug=np.nansum(dis[:,1]-dis[:,4])  #Gt
    else:
        mass_21=np.nansum(dis[:,1])#in Gt
        # delta=(len(dfnew)-dfnew['mean'].isna().sum()) #days with measures
        mass_avg=np.nansum(dis[:,4]) #in Gt


print('discharge surplus due to heatwave (Gt): ',mass_aug)
print('discharge 21 (Gt): ',mass_21)
print('discharge avg (Gt): ',mass_avg)
print('ratio HW/21: ',mass_aug/mass_21)
print('ratio HW/avg: ',mass_aug/mass_avg)


#%% cumulative plot
fig, ax = plt.subplots(figsize=(9,5))
plt.plot(np.cumsum(piv_all), color='grey')
plt.plot(np.cumsum(piv_2), color='r')
plt.plot(np.cumsum(piv_3), color='darkorange')
plt.plot(np.cumsum(piv_4), color='b')
plt.plot(np.cumsum(piv_mean), color='k')
plt.legend(handles=handles, frameon=False, fontsize=fs)
ax.axvspan(222, 232,color='grey', alpha=0.3) #label='high melt'

ax.set_ylabel("cumulative discharge, $km^3$", fontsize=fs)
ax.set_xlabel("time, day/month", fontsize=fs)
# ax.set_title('Cumulative discharge Watson River')

# plt.legend()
myFmt = mdates.DateFormatter('%d-%m')
ax.xaxis.set_major_formatter(myFmt)
#x-axis
start_date=datetime.datetime(2021, 5, 5)
end_date=datetime.datetime(2021, 10, 5)
ax.set_xlim(start_date.timetuple().tm_yday, end_date.timetuple().tm_yday)
ticks=[ 134, 151, 165, 181, 195, 212, 226, 243, 257, 273]
ax.set_xticks(ticks)

plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

#%%
# bb=np.zeros((len(piv_mean),1))*np.nan
# bb[20:129,]=np.array(np.cumsum(piv_4))  #two arrays have different length
# kk=np.array(np.cumsum(piv_mean))
# test=bb[:,0]/kk

# fig, ax = plt.subplots(figsize=(9,5))
# # plt.plot(bb, color='b')
# # plt.plot(kk, color='k')
# plt.plot(test)

# plt.axvline(x=227-129, color='grey', linestyle='--',linewidth=.7)
# plt.axvline(x=238-129, color='grey', linestyle='--',linewidth=.7)
# 0.667483 -> 0.808971

#%% compared to 1949-2021 average
fn=base_path+'Watson River discharge yearly (1949-2021).txt'
data=pd.read_csv(fn, delim_whitespace=True)
# data=data[data.Year>2005]
data[data==-999]=np.nan #change -999 to nan
df=pd.DataFrame(data['Discharge_(km3)'])
dis= np.array(df)
mean = np.nanmean(dis)
diff=dis-mean
ratio=dis/mean

