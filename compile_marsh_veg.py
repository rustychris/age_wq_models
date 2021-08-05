import six
from stompy.grid import unstructured_grid
from stompy import utils
from stompy.spatial import wkb2shp
from stompy.plot import plot_wkb
import matplotlib.pyplot as plt
import numpy as np
## 
grid_fn="../data/untrim/LTO_Restoration_2018_h.grd"
##

g=unstructured_grid.UnstructuredGrid.read_untrim(grid_fn)

##

plt.figure(1).clf()
g.plot_edges(color='k',lw=0.4)

##

# FIX THIS WITH A REAL VALUE
marsh_thresh=1.8 # roughly MHHW, but not.

marsh_frac=np.zeros(g.Ncells(),np.float64)
sav_frac  =np.zeros(g.Ncells(),np.float64)
fav_frac  =np.zeros(g.Ncells(),np.float64)

# 2s.
for c in range(g.Ncells()):
    sA,sz = g.cells['subgrid'][c]
    marsh_frac[c] = sA[sz<-marsh_thresh].sum() / sA.sum()

##

# Looks about right.
fig=plt.figure(1)
fig.clf()
fig.set_size_inches([8.4, 5],forward=True)

ccoll=g.plot_cells(values=marsh_frac)
plt.colorbar(ccoll,label="%% Marsh (%gm)"%marsh_thresh)
plt.axis('off')
plt.axis('tight')
plt.axis('equal')
fig.tight_layout()

plt.axis((602608.1236986985, 624097.5402707062, 4231791.272191033, 4246013.668029016))
plt.savefig('marsh_frac_csc.png')

##
sav_shp="../data/veg/2018/Delta_201810_SAV_shp.shp"
why_shp="../data/veg/2018/Delta_201810_WHY_shp.shp"
wpr_shp="../data/veg/2018/Delta_201810_WPR_shp.shp"

sav=wkb2shp.shp2geom(sav_shp) # 57907 features, ~1.7m resolution
##

# Options:
#  A: iterate over SAV features.
#     for each feature can select overlapping cells
#       for each overlapping cell calculate area of the intersection, increment sav for cell.
#  B: Convert to raster
#     Iterate over cells
#       get overlapping pixels, calculate fraction at the resolution of whole pixels.
#  C: Like B, but calculate exact areas.

def loop():
    dest_frac=sav_frac

    # Deconstruct cell_clip_mask to avoid recomputation of the cell bounds
    nodes=g.cells['nodes']
    x=np.where( nodes>=0, g.nodes['x'][nodes,0], np.nan )
    y=np.where( nodes>=0, g.nodes['x'][nodes,1], np.nan )
    xmin=np.nanmin(x,axis=1)
    ymin=np.nanmin(y,axis=1)
    xmax=np.nanmax(x,axis=1)
    ymax=np.nanmax(y,axis=1)

    for i,feat in utils.progress(enumerate(sav)):
        geom=feat['geom']
        if not geom.is_valid:
            geoms=geom.buffer(0) # for some cases this is enough to break up a bowtie
        else:
            geoms=[geom]
        for geom in geoms:
            xxyy=[geom.bounds[j] for j in [0,2,1,3]]
            #clip_mask=g.cell_clip_mask(xxyy,by_center=False)
            clip_mask=(xmin<xxyy[1])&(xmax>xxyy[0])&(ymin<xxyy[3])&(ymax>xxyy[2])
            cells=np.nonzero(clip_mask)[0]
            for c in utils.progress(cells,msg="  cell %s"):
                cp=g.cell_polygon(c)
                overlap=geom.intersection(cp)
                dest_frac[c] += overlap.area
        if i>5000:
            break # DBG
# slow, but not unusably slow.
            
## topology issue:
# Input has polygons where the exterior is pinched down to a point, i.e.
# should be two polygons.

fig=plt.figure(1)
fig.clf()

ccoll=g.plot_cells(mask=[c])

plot_wkb.plot_wkb(geom,alpha=0.3)
plot_wkb.plot_wkb(cp,alpha=0.2)
    
plt.axis('off')
plt.axis('tight')
plt.axis('equal')
fig.tight_layout()

for p_i,p in enumerate(np.array(geom.exterior)):
    plt.text(p[0],p[1],str(p_i),ha='center',va='center')
    
