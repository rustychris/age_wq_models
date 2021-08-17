import pandas as pd
from stompy.grid import unstructured_grid
from stompy import utils,memoize
from stompy.spatial import wkb2shp
from stompy.plot import plot_wkb
import matplotlib.pyplot as plt
import numpy as np

## 
df=pd.read_csv("marsh_and_veg_v00.csv")

##
grid_fn="../data/untrim/LTO_Restoration_2018_h.grd"

@memoize.memoize(lru=1,cache_dir='cache')
def load_grid():
    print("Loading grid")
    return unstructured_grid.UnstructuredGrid.read_untrim(grid_fn)

g=load_grid()

##
marsh_thresh=1.8 # roughly MHHW, but not.

zoom=(573619.117656995, 654316.2358780389, 4172919.034254969, 4281023.084109105)
figsize=[7.16, 8.24]

## 
# Marsh fraction on grid
fig=plt.figure(1)
fig.clf()
fig.set_size_inches(figsize,forward=True)

ccoll=g.plot_cells(values=df['marsh'],cmap='inferno_r')
ccoll.set_edgecolor('face')
ccoll.set_lw(0.25)
plt.colorbar(ccoll,label="%% Marsh (%gm)"%marsh_thresh)
plt.axis('off')
plt.axis('tight')
plt.axis('equal')
fig.tight_layout()

plt.axis(zoom)

##

# Verified that before clipping, these have a max very close
# to 1.0
# Also their sum has a max very near 1.0
fig=plt.figure(2)
fig.clf()
fig.set_size_inches(figsize,forward=True)
ax=fig.add_subplot(1,1,1)

ccoll=g.plot_cells()
ax.axis('off')
ax.axis('equal')
fig.tight_layout()

colors=np.c_[df['sav'],df['fav'],0.4*np.ones_like(df['sav'])]
ccoll.set_facecolors(colors)
ccoll.set_edgecolor('face')
pccoll.set_lw(0.25)

plt.axis(zoom)

sav_art=ax.bar([0],[1],color='#ff0066')
fav_art=ax.bar([0],[1],color='#00ff66')
ax.legend( [sav_art,fav_art],['SAV','FAV'],frameon=0)

## 

zoom=(573258.7324380949, 654904.3449657034, 4174008.8428962724, 4274877.813532808)

# Combine all 3 as RGB
fig=plt.figure(3)
fig.clf()
fig.set_size_inches([6.7,8.25],forward=True)
ax=fig.add_subplot(1,1,1)

ccoll=g.plot_cells()
ax.axis('off')
ax.axis('equal')
fig.tight_layout()

def cfun(sav,fav,marsh):
    sav=np.asarray(sav)
    fav=np.asarray(fav)
    marsh=np.asarray(marsh)

    # CMY
    # colors=np.array( [1.0-sav,1.0-fav,1.0-marsh]).T
    # colors=0.9*colors

    # faded RGB
    # colors=np.array( [sav,fav,marsh]).T
    # colors=0.5*colors + 0.5

    colors=np.array( [sav,fav,marsh]).T
    sat=colors.max(axis=-1)
    colors=sat[...,None]*colors + 0.9*(1-sat)[...,None]

    return colors

ccoll.set_facecolors(cfun(df['sav'],df['fav'],df['marsh']))
ccoll.set_edgecolor('face')
ccoll.set_lw(0.7) # overwide to make things more visible

ax.axis(zoom)

sav_art=ax.bar([0],[1],color=cfun(1,0,0))#'#ff0000')
fav_art=ax.bar([0],[1],color=cfun(0,1,0))#'#00ff00')
marsh_art=ax.bar([0],[1],color=cfun(0,0,1)) # '#0000ff')
ax.legend( [sav_art,fav_art,marsh_art],['SAV','FAV','Marsh'],frameon=0)

fig.savefig('sav_fav_marsh_rgb.png',dpi=200)
