"""
Double check the 2019 data
"""

import six
import logging
from stompy.grid import unstructured_grid
from stompy import utils
from stompy.spatial import wkb2shp
import pandas as pd
from stompy.plot import plot_wkb
import matplotlib.pyplot as plt
import numpy as np

## 
#grid_fn="../data/untrim/LTO_Restoration_2018_h.grd"
grid_fn="../data/untrim/LTO_Restoration_2019_N.grd"

#grid_version="LTOrestoration2019_N"
##

print("Loading grid")
g=unstructured_grid.UnstructuredGrid.read_untrim(grid_fn)

##

#  -rw-rw-r--  1 rusty rusty 5713590 Oct  4 11:34 marsh_and_veg_v00-LTOrestoration2019_N.dat
df=pd.read_csv("marsh_and_veg_v00-LTOrestoration2019_N.csv")

plt.figure(1).clf()
fig,axs=plt.subplots(1,3,sharex=True,sharey=True,num=1)

g.plot_cells(ax=axs[0],values=df['marsh'])
g.plot_cells(ax=axs[1],values=df['sav'])
g.plot_cells(ax=axs[2],values=df['fav'])

## 
plt.figure(2).clf()
fig,ax=plt.subplots(num=2)

def cmyk_to_rgb(c, m, y, k):
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return r, g, b

rgb=np.c_[ cmyk_to_rgb(df.marsh,df.sav,df.fav,0.15) ]
ccoll=g.plot_cells()
ccoll.set_facecolors(rgb)

ax.axis('off')
ax.axis((605957.2378417695, 639926.0540226211, 4207630.344937345, 4247166.051705137))

##

df=pd.read_csv('marsh_and_veg_v00.csv')

##

# Publication figures:
plt.figure(3).clf()
fig,ax=plt.subplots(num=2)

def cmyk_to_rgb(c, m, y, k):
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return r, g, b

rgb=np.c_[ cmyk_to_rgb(df.marsh,df.sav,df.fav,0.15) ]
ccoll=g.plot_cells()
ccoll.set_facecolors(rgb)

ax.axis('off')
ax.axis((605957.2378417695, 639926.0540226211, 4207630.344937345, 4247166.051705137))

