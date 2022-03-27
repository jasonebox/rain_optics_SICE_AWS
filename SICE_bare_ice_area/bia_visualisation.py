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

if os.getlogin() == "adrien":
    base_path = "/home/adrien/EO-IO/gitmoved/rain_optics_SICE_AWS"
if os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/rain_optics_SICE_AWS"


def date_to_int(d: pd._libs.tslibs.timestamps.Timestamp) -> int:
    """Convert datetime to day of year integer"""
    doy_int = int(d.strftime("%j"))
    return doy_int


## %% graphics parameters

font_size = 24

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
    "axes.grid": True,
}
plt.rcParams.update(params)

years = ["2017", "2018", "2019", "2020", "2021"]
colors = ["purple", "k", "r", "darkorange", "b"]

# %% two variables are plotted and each has its own parameters

watson_2021 = pd.read_csv(
    f"{base_path}/Watson_catchment/Watson_river_discharge_2021.csv"
)
watson_2021["date"] = pd.to_datetime(watson_2021["doy"], format="%j").dt.strftime(
    "%Y-%m-%d %H"
)
watson_2021["date"] = pd.to_datetime(watson_2021.date) + pd.offsets.DateOffset(
    years=121
)

# watson_2021['year']=pd.to_datetime(watson_2021.date).dt.year
# watson_2021['month']=pd.to_datetime(watson_2021.date).dt.month
# watson_2021['day']=pd.to_datetime(watson_2021.date).dt.day

# watson_2021["date"]=pd.to_datetime(watson_2021[['year', 'month', 'day']])

watson_2021 = watson_2021.rename(columns={"date": "time"})

# varnams=['albedo','bare_ice_area']
varnams = ["albedo", "bare_ice_area_gris"]
ytits = ["albedo, unitless", "bare ice area, km$^{2}$"]
legend_locs = [0, 2]
end_date = ["2001-10-01", "2001-09-15"]
xoss = [1800, 1800]
tit0 = ["Greenland ice sheet ", "Greenland "]
tit1 = ["albedo", "bare ice area"]
units = ["", "km$^{2}$"]

for j, varnam in enumerate(varnams):

    if j == 1:

        plt.figure()
        fig, ax = plt.subplots(figsize=(14, 10))

        for i, year in enumerate(years):

            annual_results = pd.read_csv(
                f"{base_path}/SICE_bare_ice_area/bia_csv/SICE_BIA_Greenland_{year}.csv"
            )

            datex = pd.to_datetime(list(annual_results.time))

            ax.plot(
                datex.strftime("%j").astype(int),
                annual_results[varnam],
                "-o",
                linewidth=3,
                color=colors[i],
                label=year,
            )

            if year == "2021":
                ax.scatter(
                    datex.strftime("%j").astype(int),
                    annual_results[varnam],
                    s=200,
                    facecolors="none",
                    zorder=20,
                    edgecolors="b",
                )

        # ax.set_title(tit0[j]+tit1[j]+' from Sentinel-3',fontsize=font_size*1.2)
        if j == 1:
            ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
            )

        latest_day = str(datex[-1:])[16:26]
        mult = 0.9
        xx0 = 0.99
        yy0 = 0.98
        dy = -0.08
        cc = 0

        if year == "2021":

            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            doy_watson = (
                pd.to_datetime(list(watson_2021.time)).strftime("%j").astype(int)
            )
            ax2.plot(doy_watson, watson_2021.discharge, "k", zorder=20)

        tcks = pd.date_range("2021-06-01", "2021-09-15", freq="SMS")
        tcks_doy = tcks.strftime("%j").astype(int)
        tcks_str = tcks.strftime("%b-%d")
        ax.set_xticks(tcks_doy)
        ax.set_xticklabels(tcks_str)

        # vt0x,vt1x,m0=return_rate(ALB,'2021-08-7','2021-08-13-11','b',15,axnum,'')
        # vt0x,vt1x,m1=return_rate(ALB,'2021-08-13-11','2021-08-14-19','r',55,axnum,'heatwave\nwith rain')
        # vt0x,vt1x,m1=return_rate(ALB,'2021-08-14-19','2021-08-20-00','g',95,axnum,'warm, cloudy\nno rain')
        # vt0x,vt1x,m2=return_rate(ALB,'2021-08-20-00','2021-08-27-00','m',45,axnum,'melting from\nalbedo feedback')
        # # vt0x,vt1x,m2=return_rate(ALB,'2021-08-27-00','2021-08-31-12','k',25,axnum,'ablation end')
        # ax[cc].axvspan(pd.to_datetime('2021-08-27'), pd.to_datetime('2021-08-31-12'),
        #        color='k', alpha=0.3)#,label='ablation end')
        # ax[cc].text(pd.to_datetime('2021-08-29-10'),3,'end of ablation\nseason',color='k',fontsize=20,alpha=1,ha='center')

        # 2021 heat and rain wave
        # plt.axvline(pd.to_datetime('2001-08-11'), color='gray', linestyle='--',
        #             linewidth=3)
        # plt.axvline(pd.to_datetime('2001-08-22'), color='gray', linestyle='--',
        #             linewidth=3)

        # plt.axvspan(pd.to_datetime('2001-08-07)', pd.to_datetime('2001-08-13-11'),
        #                            color='b', alpha=0.5)

        plt.axvspan(
            date_to_int(pd.to_datetime("2001-08-09")),
            date_to_int(pd.to_datetime("2001-08-13-11", format="%Y-%m-%d-%H")),
            color="b",
            alpha=0.5,
            label="cool period\nafter snow",
        )

        plt.axvspan(
            date_to_int(pd.to_datetime("2001-08-13-11", format="%Y-%m-%d-%H")),
            date_to_int(pd.to_datetime("2001-08-14-19", format="%Y-%m-%d-%H")),
            color="r",
            alpha=0.5,
            label="heatwave",
        )

        plt.axvspan(
            date_to_int(pd.to_datetime("2001-08-14-19", format="%Y-%m-%d-%H")),
            date_to_int(pd.to_datetime("2001-08-20-00", format="%Y-%m-%d-%H")),
            color="g",
            alpha=0.5,
            label="heat and clouds\ndissipating",
        )

        plt.axvspan(
            date_to_int(pd.to_datetime("2001-08-20-00", format="%Y-%m-%d-%H")),
            date_to_int(pd.to_datetime("2001-08-27-00", format="%Y-%m-%d-%H")),
            color="m",
            alpha=0.5,
            label="albedo and latent\nfeedbacks",
        )

        plt.axvspan(
            date_to_int(pd.to_datetime("2001-08-27")),
            date_to_int(pd.to_datetime("2001-09-01")),
            color="gray",
            alpha=0.5,
            label="ablation\nseason end",
        )

        mult = 0.8
        ax.legend(prop={"size": font_size * mult})  # ,loc=2)

        # ax.legend(loc=legend_locs[j])
        ax.set_ylabel(ytits[j], fontsize=font_size)
        # ax.set_xlabel('Time, day-month', fontsize=font_size)

        # ax.set_xlim(pd.to_datetime("2001-06-01"), pd.to_datetime(end_date[j]))
        ax.set_xlim(
            date_to_int(pd.to_datetime("2001-06-01")),
            date_to_int(pd.to_datetime(end_date[j])),
        )

        if j == 0:
            lastval = "{:.3f}".format(annual_results[varnam][-1:].values[0])
        else:
            lastval = "{:.0f}".format(annual_results[varnam][-1:].values[0])

        # plt.text(xx0, yy0+cc*dy,lastval+' '+units[j]+' on\n'+latest_day,
        #          fontsize=font_size*mult,transform=ax.transAxes,
        #          color='b',va='top',ha='right') ; cc+=1.

        ly = "x"

        figpath = "./figures/"

        if ly == "p":
            os.system("mkdir -p " + figpath)

            figx, figy = 16, 12
            # figx,figy=16*2,12*2
            plt.savefig(
                figpath + "bare_ice_area.png",
                bbox_inches="tight",
                figsize=(figx, figy),
                type="png",
                dpi=250,
                facecolor="w",
            )

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
