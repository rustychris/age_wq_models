import six
import logging
from stompy.grid import unstructured_grid
from stompy import utils
from stompy.spatial import wkb2shp
from stompy.plot import plot_wkb
import matplotlib.pyplot as plt
import numpy as np
from shapely import prepared, ops
import time
## 
grid_fn="../data/untrim/LTO_Restoration_2018_h.grd"
##

print("Loading grid")
g=unstructured_grid.UnstructuredGrid.read_untrim(grid_fn)

##

# FIX THIS WITH A REAL VALUE
marsh_thresh=1.8 # roughly MHHW, but not.

marsh_frac=np.zeros(g.Ncells(),np.float64)
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
missing_2018_shp="../data/veg/2018/missing_swath.shp"

sav2018_shp="../data/veg/2018/Delta_201810_SAV_shp.shp"
sav2020_shp="../data/veg/2020/202007_rfclass_v7_SAV.shp"

why2018_shp="../data/veg/2018/Delta_201810_WHY_shp.shp"
why2020_shp="../data/veg/2020/202007_rfclass_v7_WH.shp"

wpr2018_shp="../data/veg/2018/Delta_201810_WPR_shp.shp"
wpr2020a_shp="../data/veg/2020/202007_rfclass_v7_WPinEMR.shp"
wpr2020b_shp="../data/veg/2020/202007_rfclass_v7_WP.shp"

## 

def sanitize(geom):
    """
    Return a list of valid geometries from a potentially invalid geometry
    """
    old_level=logging.root.level
    try:
        logging.root.setLevel(logging.WARNING)
        if not geom.is_valid:
            geoms=geom.buffer(0) # for some cases this is enough to break up a bowtie
        else:
            geoms=[geom]
    finally:
        logging.root.setLevel(old_level)
    return geoms

def int_area(args):
    a,b,idx=args
    return idx,a.intersection(b).area

# Options:
#  A: iterate over SAV features.
#     for each feature can select overlapping cells
#       for each overlapping cell calculate area of the intersection, increment sav for cell.
#  B: Convert to raster
#     Iterate over cells
#       get overlapping pixels, calculate fraction at the resolution of whole pixels.
#  C: Like B, but calculate exact areas.

class Overlay(object):
    def __init__(self,grid,src_shp,exclude_shp=None,include_shp=None,pool=None):
        print("Top of overlay")
        self.pool=pool
        self.delayed=[]
        self.grid=grid
        dest_frac=self.dest_frac=np.zeros(self.grid.Ncells(),np.float64)
        Ac=self.grid.cells_area()

        # Masking
        print("Load masks")
        exclude=include=None
        prep_mask=lambda shp: ops.cascaded_union( wkb2shp.shp2geom(shp)['geom'] )
        if exclude_shp is not None:
            exclude=prep_mask(exclude_shp)
        elif include_shp is not None:
            include=prep_mask(include_shp)
            
        # Prep src_shp:
        print("Load source geometries")
        src=wkb2shp.shp2geom(src_shp)
        print("Subset source geometries")
        src_subset=[]
        for geomA in src['geom']:
            if include:
                # bounds tests
                inc_xyxy=include.bounds
                geo_xyxy=geomA.bounds
                if inc_xyxy[0]>geo_xyxy[2]: continue
                if inc_xyxy[1]>geo_xyxy[3]: continue
                if inc_xyxy[2]<geo_xyxy[0]: continue
                if inc_xyxy[3]<geo_xyxy[1]: continue
            src_subset.append(geomA)
            
        print("Sanitize geometries")
        if self.pool:
            sane_geoms=self.pool.imap_unordered(sanitize,src_subset,chunksize=40)
        else:
            sane_geoms=(sanitize(geom) for geom in src_src_subset)
            
        sane_count=0
        src_geoms=[]
        for multigeom in sane_geoms: # multipart, or singleton list
            sane_count+=1
            if sane_count%1000==0:
                print("%d sanitized of %d original"%(len(src_geoms),len(src)))
            for geomB in multigeom:
                if include:
                    geom_clip=geomB.intersection(include)
                elif exclude:
                    geom_clip=geomB.difference(exclude)
                else:
                    geom_clip=geomB
                if geom_clip.area==0.0: continue
                if geom_clip.type=="MultiPolygon":
                    for subgeom in geom_clip:
                        src_geoms.append(subgeom)
                else:
                    src_geoms.append(geom_clip)
        print("Src geoms: %d after preparation: %d"%(len(src),len(src_geoms)))
        
        # Deconstruct cell_clip_mask to avoid recomputation of the cell bounds
        nodes=g.cells['nodes']
        x=np.where( nodes>=0, g.nodes['x'][nodes,0], np.nan )
        y=np.where( nodes>=0, g.nodes['x'][nodes,1], np.nan )
        xmin=np.nanmin(x,axis=1)
        ymin=np.nanmin(y,axis=1)
        xmax=np.nanmax(x,axis=1)
        ymax=np.nanmax(y,axis=1)

        # Overlay
        for geom in utils.progress(src_geoms):
            xxyy=[geom.bounds[j] for j in [0,2,1,3]]
            clip_mask=(xmin<xxyy[1])&(xmax>xxyy[0])&(ymin<xxyy[3])&(ymax>xxyy[2])
            cells=np.nonzero(clip_mask)[0]
            cps=[g.cell_polygon(c) for c in cells]
            if len(cells)>20:
                geom_p=prepared.prep(geom)
            else:
                geom_p=geom
            for c,cp in utils.progress(zip(cells,cps),msg="  cell %s"):
                if not geom_p.intersects(cp): continue
                elif geom_p.contains(cp):
                    dest_frac[c] += 1
                else:
                    if self.delayed is not None:
                        self.delayed.append([geom,cp,c])
                    else:
                        # This line is where all the expense is 
                        overlap=geom.intersection(cp)
                        dest_frac[c] += overlap.area / Ac[c]
        # Come back to handle all true intersections in one go via multiprocessing
        if self.pool:
            print("Multiprocessing over %d intersection calls"%len(self.delayed))
            results=pool.imap_unordered(int_area,self.delayed,chunksize=40)
        else:
            print("Single processing over %d intersection calls"%len(self.delayed))
            results=(int_area(a) for a in self.delayed)
        for i,(c,area) in enumerate(results):
            if i%1000==0:
                print("Tabulated %d/%d intersection results"%(i,len(self.delayed)))
            dest_frac[c]+=area/Ac[c]

        if np.any(np.isnan(dest_frac)):
            print("ERROR: some dest_frac not finite")
        else:
            if dest_frac.min() < 0.0:
                print("ERROR: min dest_frac <0")
            if dest_frac.max() > 1.01:
                print("ERROR: max dest_frac >1.01")
        dest_frac[dest_frac>1.0]=1.0

# considerably faster than SAV for 2018
# Single thread, full 2020 runs in maybe 5 minutes

##

with mp.Pool(4) as pool:
    sav2020_ovl=Overlay(g,sav2020_shp,include_shp=missing_2018_shp,pool=pool)
    np.save('sav_2020.np',sav2020_ovl.dest_frac)
    

# when pushing the hard work to delayed ?
# seems to be quite stuck not moving through geometries or cells?
# Where? last message was INFO:utils:26774/57907
# Ah - it's during sanitize.
# A few features are *very* slow to sanitize
## 
import multiprocessing as mp
t=time.time()
with mp.Pool(4) as pool:
    sav2018_ovl=Overlay(g,sav2018_shp,exclude_shp=missing_2018_shp,pool=pool)
elapsed=time.time()-t

# 2350s
print("2018 SAV processed in %.3fs"%elapsed)
np.save('sav_2018.np',sav2018_ovl.dest_frac)

##

with mp.Pool(4) as pool:
    why2018_ovl=Overlay(g,why2018_shp,exclude_shp=missing_2018_shp,pool=pool)
np.save('why_2018.np',why2018_ovl.dest_frac)

# 65s or so
t=time.time()
with mp.Pool(4) as pool:
    why2020_ovl=Overlay(g,why2020_shp,include_shp=missing_2018_shp,pool=pool)
elapsed=time.time()-t
dest_frac=why2020_ovl.dest_frac.clip(0,1)
np.save('why_2020.np',dest_frac)

##

wpr2018_shp="../data/veg/2018/Delta_201810_WPR_shp.shp"
wpr2020a_shp="../data/veg/2020/202007_rfclass_v7_WPinEMR.shp"
wpr2020b_shp="../data/veg/2020/202007_rfclass_v7_WP.shp"

t=time.time()
with mp.Pool(4) as pool:
    wpr2020a_ovl=Overlay(g,wpr2020a_shp,include_shp=missing_2018_shp,pool=pool)
    np.save('wpr_2020a.np',wpr2020a_ovl.dest_frac)

    wpr2020b_ovl=Overlay(g,wpr2020b_shp,include_shp=missing_2018_shp,pool=pool)
    np.save('wpr_2020b.np',wpr2020b_ovl.dest_frac)

    wpr2018_ovl=Overlay(g,wpr2018_shp,exclude_shp=missing_2018_shp,pool=pool)
    np.save('wpr_2018.np',wpr2018_ovl.dest_frac)
elapsed=time.time()-t
print("Elapsed for water primrose: %.3fs"%elapsed)

    
##
                    
# slow, but not unusably slow.

# For 5000, while profiling, 378s
# With the prep step and a threshold of 100, this is down to 117s
# threshold of 20 that's down to 107s. Diminishing returns.
# With contains check: down to 76s. 64s in intersection, so no substantial
# overhead for the contains call.
    
##
fig=plt.figure(1)
fig.clf()
fig.set_size_inches([8.4, 5],forward=True)

ccoll=g.plot_cells(values=dest_frac)
plt.colorbar(ccoll,label="% 2018 SAV")
plt.axis('off')
plt.axis('tight')
plt.axis('equal')
fig.tight_layout()

plt.axis((602608.1236986985, 624097.5402707062, 4231791.272191033, 4246013.668029016))
#plt.savefig('sav2020_frac_csc.png')

##

srcs=dict(
    sav_2018= np.load('sav_2018.np.npy'),
    sav_2020= np.load('sav_2020.np.npy'),
    why_2018= np.load('why_2018.np.npy'),
    why_2020= np.load('why_2020.np.npy'),
    wpr_2018= np.load('wpr_2018.np.npy'),
    wpr_2020a=np.load('wpr_2020a.np.npy'),
    wpr_2020b=np.load('wpr_2020b.np.npy')
)

##

# Verified that before clipping, these have a max very close
# to 1.0
# Also their sum has a max very near 1.0
sav_combined=(srcs['sav_2018'] + srcs['sav_2020']).clip(0,1.0)
fav_combined=(srcs['why_2018'] + srcs['why_2020'] + srcs['wpr_2018']
              + srcs['wpr_2020a'] + srcs['wpr_2020b']).clip(0,1.0)

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)

ccoll=g.plot_cells()
ax.axis('off')
ax.axis('equal')
fig.tight_layout()

colors=np.c_[sav_combined,fav_combined,0.4*np.ones_like(sav_combined)]
ccoll.set_facecolors(colors)

##

# What's a good datum for marsh?
# At Sacramento, NOAA puts MHHW at 2.43m NAVD88
# Port Chicago MHHW is at 1.83m NAVD88
# Just punt with 1.8
import pandas as pd

df=pd.DataFrame()

df['cell']=1+np.arange(g.Ncells())
df['marsh']=marsh_frac
df['sav']=sav_combined
df['fav']=fav_combined

df.to_csv("marsh_and_veg_v00.csv",index=False,float_format="%.3f")

## 
fmt = '%8d %6.3f %6.3f %6.3f'
with open("marsh_and_veg_v00.dat",'wt') as fp:
    fp.write("cell      marsh  sav    fav  \n")
    np.savetxt(fp, df.values, fmt=fmt)

