#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, University of Zurich, Switzerland

"""

import rasterio
from pyproj import Transformer

# %% load geodata

base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_retrievals/metatiffs'

elev = rasterio.open(f'{base_path}/elev_1km_1487x2687.tif')

# %% point coordinates

point_lon = -47.301900
point_lat = 67.009100

# %% convert point coordinates to EPSG:3413

inProj = 'epsg:4326'
outProj = 'epsg:3413'

trf = Transformer.from_crs(inProj, outProj, always_xy=True)
point_x, point_y = trf.transform(point_lon, point_lat)

# %% get row and column at point

row, col = rasterio.transform.rowcol(elev.transform, point_x, point_y)

# %% get elevation at row and column

print(f'Elevation at {point_lon}, {point_lat}: {elev.read(1)[row, col]}')
