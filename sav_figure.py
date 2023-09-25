"""
SAV figure for nitrate manuscript
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

# 2023-09-06: Crop to match other figures, and add labels for
# AV-related location mentions in the text
# Franks Tract, Temporary Barrier in False River 2015?
# Cache Slough "Complex"
# Old River, Middle River channels if not shown elsewhere
# 

# 
## 
#grid_fn="../data/untrim/LTO_Restoration_2018_h.grd"
grid_fn="../data/untrim/LTO_Restoration_2019_N.grd"

##

print("Loading grid")
g=unstructured_grid.UnstructuredGrid.read_untrim(grid_fn)
poly=g.boundary_polygon()
##
swaths=wkb2shp.shp2geom('swaths-v00.shp')

##

#  -rw-rw-r--  1 rusty rusty 5713590 Oct  4 11:34 marsh_and_veg_v00-LTOrestoration2019_N.dat
df=pd.read_csv("marsh_and_veg_v00-LTOrestoration2019_N.csv")

## 
import cmocean
from stompy.plot import plot_wkb

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
#fig.set_size_inches([5.5,5.5],forward=True)
fig.set_size_inches([4.6,5.5],forward=True)
ax.set_adjustable('datalim')
cax=fig.add_axes([0.78,0.62,0.02,0.25])

cmap=cmocean.cm.algae
import stompy.plot.cmap as scmap

def desaturate(rgb):
    val=np.mean(rgb)
    theta=(val+0.2).clip(0,1)
    # when val==1, return gray
    # when val==0, return rgb
    return [theta*val+(1-theta)*rgb[0],
            theta*val+(1-theta)*rgb[1],
            theta*val+(1-theta)*rgb[2]]

# Desaturate the low end so most of the grid is just gray.
# Keep the same ramp of value though.
# And clip the low end so that we start on a gray that's distinct
# from the background white
# Also ease up on the high end so it's green, not so close to black
# that annotations get lost.
cmap=scmap.transform_color(desaturate,scmap.cmap_clip(cmap,0.05,0.6))

# Bulk up the lines to 1.0 to make some channels more visible
ccoll=g.plot_cells(values=df.sav+df.fav,cmap=cmap,clim=[0.0,1],
                   edgecolor='face',lw=1.,ax=ax)
cbar=plt.colorbar(ccoll,label='Fraction SAV+FAV',cax=cax)
#plot_wkb.plot_wkb(poly,edgecolor='k',fc='none',lw=0.3,zorder=2,ax=ax)

ax.axis('off')
fig.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.98)
#ax.axis((569739., 656663, 4180000, 4271000.))
ax.axis( (595067., 659066., 4177694., 4254381.))

from matplotlib import patches
artists={}
if 1: # show 2018 footprint
    for rec in swaths:
        if rec['year']=='2018':
            color='tab:red'
        else:
            color='tab:blue'
        geo=rec['geom'].difference(rec['geom'].buffer(-800))
        art=plot_wkb.plot_wkb(geo,ax=ax,zorder=2,facecolor=color,alpha=0.3,
                              lw=0.5,edgecolor='none')
        #art=plot_wkb.plot_wkb(rec['geom'],ax=ax,zorder=2,facecolor='none',alpha=0.3,
        #                      lw=1.5,edgecolor=color)
        artists[rec['year']]=art

#import matplotlib.patheffects as pe

ax.legend(artists.values(),artists.keys(),
          frameon=False,loc='upper left',
          bbox_to_anchor=[0.74,1.0])


##

# labels
kw=dict(arrowprops=dict(arrowstyle='-'),
        fontsize=9)

ax.annotate( "Franks Tract",
             [622283., 4211099],
             xytext=[618000, 4197400.],
             ha='right',**kw)

ax.annotate("Cache Slough\nComplex",
            [615617, 4238840.],
            xytext=[608081., 4247990.],
            ha='center',**kw)

ax.plot( [616835.,616968],[ 4212697, 4213162],"r-",lw=3.5)

ax.annotate("False River\nTemp. Barrier",
            [616835, 4212697.],
            xytext=[609000., 4202800],
            ha='center',**kw)

# Try Old / Middle River

ax.annotate("Old R.",
            [626500., 4201000],
            ha='center',
            rotation=287,
            fontsize=kw['fontsize'])
ax.annotate("Middle R.",
            [632000., 4195000.],
            ha='center',
            rotation=298,
            fontsize=kw['fontsize'])


##
fig.savefig('sav-figure.png',dpi=300)


##
# # Alternative approach -- no colorbar, tho
# plt.figure(3).clf()
# fig,ax=plt.subplots(num=3)
# fig.set_size_inches([5.5,5.5],forward=True)
# ax.set_adjustable('datalim')
# 
# ccoll1=g.plot_cells(color='0.8',edgecolor='face',lw=0.3,ax=ax)
# ccoll2=g.plot_cells(color='darkgreen',alpha=df.sav,
#                     edgecolor='face',lw=0.3,ax=ax)
# 
# #plot_wkb.plot_wkb(poly,edgecolor='k',fc='none',lw=0.3,zorder=2,ax=ax)
# 
# # ax.legend([ccoll2],['SAV'])
# 
# ax.axis('off')
# fig.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.98)
# ax.axis((569739.7340534998, 656663.8913317377, 4181680.362288973, 4273026.219875409))

