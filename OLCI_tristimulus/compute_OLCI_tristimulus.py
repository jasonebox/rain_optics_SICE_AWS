#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zurich, Switzerland

compute OLCI Tristimulus based on SentinelHub custom script:

https://custom-scripts.sentinel-hub.com/sentinel-3/tristimulus/#

Applied to August 20 

"""

import rasterio
from rasterio.mask import mask
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

# %% prepare paths

if os.getlogin() == "adrien":
    base_path = "/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_retrievals"
elif os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/rain_optics_SICE_AWS/SICE_retrievals"

os.chdir(base_path)

# %% create dictionnary with all bands for a simple combination with SH script

# get only 1 to 21, no SLSTR bands
toa_files = sorted(glob.glob("./data/2021-08-21/r_TOA_[0-9][0-9].tif"))

bds = {}

for file in toa_files:

    band = file.split("_")[-1].split(".")[0]

    bds[band] = rasterio.open(file).read(1)

# %% apply SH custom script


def tristimulus(bds):

    red = np.log(
        1.0
        + 0.01 * bds["01"]
        + 0.09 * bds["02"]
        + 0.35 * bds["03"]
        + 0.04 * bds["04"]
        + 0.01 * bds["05"]
        + 0.59 * bds["06"]
        + 0.85 * bds["07"]
        + 0.12 * bds["08"]
        + 0.07 * bds["09"]
        + 0.04 * bds["10"]
    )

    green = np.log(
        1.0
        + 0.26 * bds["03"]
        + 0.21 * bds["04"]
        + 0.50 * bds["05"]
        + bds["06"]
        + 0.38 * bds["07"]
        + 0.04 * bds["08"]
        + 0.03 * bds["09"]
        + 0.02 * bds["10"]
    )

    blue = np.log(
        1.0
        + 0.07 * bds["01"]
        + 0.28 * bds["02"]
        + 1.77 * bds["03"]
        + 0.47 * bds["04"]
        + 0.16 * bds["05"]
    )

    return [red, green, blue]


tristi = tristimulus(bds)

tristi_arr = np.array(tristi).swapaxes(1, 2).T

# %% plot tristimulus

plt.figure()
plt.imshow(np.array(tristi).swapaxes(1, 2).T, origin="upper")

# %% write tristimulus out


# get profile example

profile = rasterio.open(toa_files[0], "r").profile
profile.update(count=3)

with rasterio.open(
    "../OLCI_tristimulus/tristimulus_202108021.tif", "w", **profile
) as dst:
    dst.write(tristi_arr[:, :, 0], 1)
    dst.write(tristi_arr[:, :, 1], 2)
    dst.write(tristi_arr[:, :, 2], 3)
