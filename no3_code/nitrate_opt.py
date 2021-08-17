# proof of concept NO3 prediction along mainstem sacramento
# strategy:
# - make really bad predictions of NO3 based only on nitrification
#   - assume a constant nitrification rate for now
# - also estimate NH4 at stations
#   - if NH4 much less than NH4 at Freeport, assumption of constant nitrification is questionable
# - examine residuals to understand what might be missing
#   - temperature experienced by tracer
# - estimate optimal rate
#   
from __future__ import print_function
import six
from scipy.optimize import fmin 

import os.path, sys
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from netCDF4 import Dataset, num2date
from shapely import geometry

from pylab import date2num as d2n
from pylab import num2date as n2d
from stompy.utils import fill_invalid, to_dt64, to_datetime, to_dnum, interp_near
from stompy.filters import lowpass_godin, lowpass
from stompy.grid import unstructured_grid 
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import math
import copy
import pandas as pd
import pyproj
import functools
from stompy.memoize import memoize 
from obsdata_utils import save_current_file, get_var, get_var_dss
from obsdata_utils import get_var_dss_light
#from scipy.integrate import solve_ivp
from scipy.special import lambertw
from scipy.stats import linregress
from scipy import stats
from DSS_IO import (DSS_IO, bad_data_val, get_dss_interval_for_times,
                    split_DSS_record_name, create_DSS_record_name)
import stompy.plot.cmap as scmap
from stompy.plot import plot_wkb
jet=scmap.load_gradient('turbo.cpt') # jet-ish, but with smooth gradients

import pdb

plt.style.use('meps.mplstyle')

N_g = 14.0067 # weight of a mole of N

import nitrate_common as common
six.moves.reload_module(common)

### Load underway data, hydro grid, and thin data by unique time step/cell
uw_df_thin,grd=common.load_underway()
grd_poly=grd.boundary_polygon()
nrows = len(uw_df_thin)

##

fig_dir="fig000"
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
    
##  Read tracer outputs, including age, depth, temperature

age_data=common.read_tracer_output()

# age_data[variable][station] = time series array
# variable: 
# RH: dn_age[sta] => age_data['dn'][sta]
# and bare 'age' and 'conc' deleted

## 

# read flow data from UnTRIM input
# low-pass filter flow
flow_dss_fname = 'UnTRIM_flow_BCs_2011-2019.dss'
flow_dss_rec = {'sr':'//SACWWTP/FLOW//1HOUR/CIWQS-MERGED/',
                'fp':'/FILLED+SHIFT-7HRS/FREEPORT/FLOW-GODIN//15MIN/USGS-MERGED/'}
flow_stations = ['sr', 'fp']
dn_flow, flow = get_var_dss(stations=flow_stations, dss_records=flow_dss_rec,
                                                    dss_fname=flow_dss_fname)
flow['sr_lp'] = lowpass_godin(flow['sr'], dn_flow['sr'], ends='nan')
# dilute NH4 at Freeport
# NH4 in "ambient" water based on underway mapping in July 2020
# NH4 - typical Sac Regional NH4 in July/Aug from Kraus spreadsheet for 2020
NH4_sr = 30 # mg/L as N
NH4_am_uM = 1.0 # uM
NH4_am = (NH4_am_uM/1000.)*N_g

# put the flow on same time steps as age
flow_age = {}
for sta in flow_stations:
    f_interp = interp1d(dn_flow[sta],flow[sta])
    flow_age[sta] = f_interp(age_data['dn']['di'])

# assume fp and sr are same length (true as of now)
NH4_start_dt = dt.datetime(2018,7,1)
NH4_end_dt = dt.datetime(2018,11,1)
NH4_dt = []
NH4_dt.append(NH4_start_dt)
while NH4_dt[-1] < NH4_end_dt:
    NH4_dt.append(NH4_dt[-1]+dt.timedelta(hours=0.25))
NH4_dt = np.asarray(NH4_dt)
NH4_dn = d2n(NH4_dt)

NH4_fp = np.zeros_like(NH4_dn)
for n, dn in enumerate(NH4_dn):
    nflow_fp = np.where(dn_flow['fp']>=dn)[0][0]
    nflow_sr = np.where(dn_flow['sr']>=dn)[0][0]
    flow_am = flow['fp'][nflow_fp] - flow['sr_lp'][nflow_sr] # ambient flow
    NH4_fp[n] = (NH4_sr*flow['sr_lp'][nflow_sr] + NH4_am*flow_am)/flow['fp'][nflow_fp]

    
### 
# read USGS continuous monitoring NO3 data
usgs_dss_fname = 'chla_data.dss'
NO3_dss_rec = {'fp':'/SAC R FREEPORT/11447650/INORGANIC_NITRO//15MIN/USGS/',
               'dc':'/SAC R ABV DCC/11447890/INORGANIC_NITRO//15MIN/USGS/',
               'cl':'/CACHE SL S LIB ISL NR RV/11455315/INORGANIC_NITRO//15MIN/USGS/',
               'cr':'/CACHE SL AB RYER ISL NR RV/11455385/INORGANIC_NITRO//15MIN/USGS/',
               'di':'/SAC R A DECKER ISL RV/11455478/INORGANIC_NITRO//15MIN/USGS/',
               'vs':'/SUISUN BAY VAN SICKLE/11455508/INORGANIC_NITRO//15MIN/USGS/'}
NO3_stations = ['fp','dc','cr','cl','di','vs']
# These come in with uniform timestamps and some nan values
dn_NO3, NO3 = get_var_dss(stations=NO3_stations, dss_records=NO3_dss_rec,
                                                 dss_fname=usgs_dss_fname)
# put the NO3 on same time steps as age
NO3_age = {}
for sta in NO3_stations:
    f_interp = interp1d(dn_NO3[sta],NO3[sta])
    NO3_age[sta] = f_interp(age_data['dn']['di'])

# fill NO3 data at Freeport
NO3['fp_fill'] = fill_invalid(NO3['fp'])
NO3['fp_lp'] = lowpass_godin(NO3['fp_fill'], dn_NO3['fp'], ends='nan')

##
# RH: check on gap in NO3 at freeport
# Put USGS station data into a dataframe
usgs_df_long=pd.concat([ pd.DataFrame(dict(time=to_dt64( dn_NO3[sta.replace('_fill','').replace('_lp','')]),
                                           no3=NO3[sta],sta=sta))
                         for sta in NO3.keys()])
# Combine to wide format
usgs_df_wide=usgs_df_long.set_index(['time','sta']).unstack('sta').droplevel(0,1)

# basic linear regression:
from statsmodels.formula.api import ols
#model = ols(formula="fp~dc", data=usgs_df_wide).fit()
#usgs_df_wide['fp_lin']=model.predict(usgs_df_wide)

#linear regression against lagged value
usgs_dn=to_dnum(usgs_df_wide.index.values)
# adjust lag to maximize R-squared: 0.7 days => R2=0.933
usgs_df_wide['dc_lag']=np.interp( usgs_dn+0.7,
                                  usgs_dn,usgs_df_wide.dc)
model_lag = ols(formula="fp~dc_lag", data=usgs_df_wide).fit()
usgs_df_wide['fp_lag']=model_lag.predict(usgs_df_wide)
##
if 1: # 1 to enable filling FP gaps with lag/regressed DCC
    # Fill FP gaps with this model. First, interpolate onto age
    # times, but not over large gaps (max_dx=1 day)
    # fill NO3 data at Freeport
    valid=np.isfinite(NO3['fp'])
    max_gap_days=1.0
    print("fp:       %d of %d in NO3[fp] are valid"%(valid.sum(),len(valid)))
    fp_fill=interp_near( dn_NO3['fp'], dn_NO3['fp'][valid], NO3['fp'][valid], max_dx=max_gap_days)
    fp_valid=np.isfinite(fp_fill)
    print("fp:       %d of %d valid after fill small gaps"%(fp_valid.sum(),len(fp_valid)))
    
    # Use the linear model and lagged DCC data to fill in fp where the lagged
    # DCC data is valid.
    fp_lag_valid=np.isfinite(usgs_df_wide['fp_lag'].values)
    fp_lag_fill=interp_near( dn_NO3['fp'], usgs_dn[fp_lag_valid], usgs_df_wide['fp_lag'].values[fp_lag_valid],
                             max_dx=max_gap_days)
    lag_valid=np.isfinite(fp_lag_fill)
    print("fp_lag:   %d of %d valid after fill small gaps"%(lag_valid.sum(),len(lag_valid)))

    combined=np.where(fp_valid,fp_fill,fp_lag_fill)
    print("combined: %d of %d valid"%(np.isfinite(combined).sum(),len(combined)))
    
    NO3['fp_fill'] = fill_invalid(combined)
    print("fp_fill:  %d of %d valid"%(np.isfinite(NO3['fp_fill']).sum(),len(NO3['fp_fill'])))
    # Seems like the signals, especially with the splices and questionable data,
    # have some noise that Godin is missing. Go with long-ish Butterworth.
    # NO3['fp_lp'] = lowpass_godin(NO3['fp_fill'], dn_NO3['fp'], ends='nan')
    NO3['fp_lp'] = lowpass(NO3['fp_fill'], dn_NO3['fp'], cutoff=2.5)

    # Don't update NO3_age -- it's used for plotting observed NO3 and calculating residuals,
    # so don't pollute it with this 
    
##

# Make sure it's kosher to use searchsorted below
assert np.all(np.diff(dn_NO3['fp'])>=0) # just to be sure
assert np.all(np.diff(NH4_dn)>=0) # just to be sure

    
# How slow is the forward model with no preprocessing?
def nitri_model(dnums_in,age_in,
                k_ni,kmm,daily_NO3_loss,Csat,method='mm'):
    # These could be preprocessed
    dn_lagged = dnums_in - age_in
    n_NO3=np.searchsorted(dn_NO3['fp'],dn_lagged)
    n_NH4 = np.searchsorted(NH4_dn,dn_lagged)
    NH4_lag = NH4_fp[n_NH4] # lagged boundary NH4
    NO3_lag = NO3['fp_lp'][n_NO3] # lagged boundary NO3

    # Steps dependent on model parameters
    if method=='first': # First order model:
        NH4_atten=1.-np.exp(-k_ni*age_in) # fraction of NH4 nitrified
        d_nit=NH4_lag*NH4_atten # mg-N/l moved from NH4 to NO3
    elif method=='mm': # Michaelis Menten closed form
        # unit check:
        # (mg/l / mg/l) * exp( mg/l / mg/l & mg/l/day / mg/l * days )
        #       1         exp(     1       &  1 )
        # 1
        F=NH4_lag/Csat*np.exp(NH4_lag/Csat-kmm/Csat*age_in)
        # For F>=0, lambertw on principal branch is real. safe to cast.
        # Write this as flux to make mass conservation clear (minor overhead)
        d_nit=NH4_lag-Csat*np.real(lambertw(F))
        # unit check:
        #   mg/l - mg/l * 1
    else:
        raise Exception("bad method %s"%method)
        
    # add in loss term
    NO3_pred = NO3_lag + d_nit - daily_NO3_loss*age_in
    NH4_pred = NH4_lag - d_nit
    return NH4_pred,NO3_pred

# now that necessary data is loaded in, plot maps

##

# Run model
dnums = uw_df_thin['dnums'].values
uw_age = uw_df_thin['age'].values # RH: used to be just 'age'

# 3.2ms
# Now updated to use optimized values 
NH4_mm,NO3_mm = nitri_model(dnums,uw_age,k_ni=np.nan,kmm=0.053,Csat=0.35,daily_NO3_loss=0.0021)

# plot maps
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
fig.set_size_inches([12,12],forward=True)
x = uw_df_thin['x'].values
y = uw_df_thin['y'].values
NO3_obs = uw_df_thin['NO3-N'].values
NH4_obs = uw_df_thin['x'].values
cmax = 0.4
sc_obs = ax.scatter(x, y, c=NO3_obs, cmap=jet, s=60, vmin=0, vmax=cmax)
sc_mm = ax.scatter(x, y, c=NO3_mm, cmap=jet, s=5, vmin=0, vmax=cmax)
plt.ion()
plt.colorbar(sc_obs,label="NO$_3$ (mg/l)")
ax.text(0.98,0.95,"Outer=observed\nInner=MM model",transform=ax.transAxes,ha='right')
ax.axis('off')
ax.axis('equal')
plot_wkb.plot_wkb(grd_poly,ax=ax,fc='0.8',ec='0.8',zorder=-2)
fig.tight_layout()
plt.show()
fig.savefig(os.path.join(fig_dir,'obs_vs_model_map.png'),dpi=150)

##

# Scatter of the same thing
x = uw_df_thin['x'].values
y = uw_df_thin['y'].values

from stompy.spatial import wkb2shp
regions=wkb2shp.shp2geom('regions_v00.shp')

#in_corridor=np.array([common.sac_poly.contains( geometry.Point(pnt) )
#                      for pnt in np.c_[x,y] ])

by_region={} # bitmask for each region
for feat in regions:
    by_region[feat['name']]=np.array([feat['geom'].contains( geometry.Point(pnt) )
                                      for pnt in np.c_[x,y] ])
    

NO3_obs = uw_df_thin['NO3-N'].values
NH4_obs = uw_df_thin['x'].values

region_labels={'sac':'Mainstem Sac.',
               'cache':'Cache Slough Complex',
               'suisun':'Suisun Bay',
               'south_interior':'Interior and South Delta'}
colors=['tab:Blue','tab:Orange','tab:Red','tab:Green']


## Scatter, colored by regions
plt.figure(2).clf()
fig,(ax_map,ax)=plt.subplots(1,2,num=2)
fig.set_size_inches([12,7],forward=True)

ax_map.axis('equal')
ax_map.axis('off')
plot_wkb.plot_wkb(grd_poly,ax=ax_map,color='0.8',zorder=-2)
scats=[]
# Choose a nicer order for layering
region_names=['south_interior','cache','sac','suisun']
for region,col in zip(region_names,colors):
    # Coloring by dnum wasn't that helpful.
    mask=by_region[region]
    scat=ax.scatter(NO3_obs[mask], NO3_mm[mask], s=4, color=col,label=region_labels[region])
    scat_map=ax_map.scatter(x[mask], y[mask], s=4, color=col,label=region_labels[region])
    
ax.set_xlabel('Observed NO$_3$')
ax.set_ylabel('Predicted NO$_3$')
ax.plot([0,0.6],[0,0.6],color='k',lw=0.5)
ax.set_aspect(1.0)
ax.set_adjustable('box')
ax.axis(xmax=0.45,xmin=0,ymin=0.0,ymax=0.45)
ax_map.legend(loc='lower left',frameon=0)

ax_map.axis( (579000., 656131., 4175818., 4283670.) )
fig.tight_layout()

fig.savefig(os.path.join(fig_dir,'obs_vs_model_scatter_regions.png'),dpi=150)

## Scatter, colored by age, conc
for i,scal in enumerate(['age','conc']):
    plt.figure(3+i).clf()
    fig,(ax_map,ax)=plt.subplots(1,2,num=3+i)
    fig.set_size_inches([12,7],forward=True)

    ax_map.axis('equal')
    ax_map.axis('off')
    plot_wkb.plot_wkb(grd_poly,ax=ax_map,color='0.8',zorder=-2)

    val=uw_df_thin[scal]
    scat=ax.scatter(NO3_obs, NO3_mm, 4, val)
    scat_map=ax_map.scatter(x, y, 4, val)
        
    ax.set_xlabel('Observed NO$_3$')
    ax.set_ylabel('Predicted NO$_3$')
    ax.plot([0,0.6],[0,0.6],color='k',lw=0.5)
    ax.set_aspect(1.0)
    ax.set_adjustable('box')
    ax.axis(xmax=0.45,xmin=0,ymin=0.0,ymax=0.45)

    if scal=='age':
        plt.setp([scat,scat_map], clim=[0,40])
        plt.colorbar(scat_map,label="Age (d)")
    elif scal=='conc':
        plt.setp([scat,scat_map], clim=[0,1])
        plt.colorbar(scat_map,label="Concentration (-)")

    ax_map.axis( (579000., 656131., 4175818., 4283670.) )
    fig.tight_layout()

    fig.savefig(os.path.join(fig_dir,'obs_vs_model_scatter_%s.png'%scal),dpi=150)


## 
# Optimization:

noffset = 0

def cost_params(method='mm',
                opt_stations=['dc','cr','cl','di','vs'],
                **model_params):
    rmses=[]
    for sta in opt_stations:
        dnum_in=age_data['dn'][sta][noffset:]
        age_in =age_data['age'][sta][noffset:]
        # Vectorized:
        NH4_model,NO3_model = nitri_model(dnum_in,age_in,method=method,**model_params)

        NO3_res = NO3_age[sta] - NO3_model
        valid = np.isfinite(NO3_res)
        res_val = NO3_res[valid]

        rmses.append(np.sqrt(np.mean(res_val**2)))
    return np.mean(rmses)


##

# First, just kmm



# Ranges for parameter scan
kmms=np.linspace(0.01,0.20,50)
losses=np.linspace(0.000, 0.0030, 25)
Csats=np.linspace(0.1, 1.9, 30)
knis=np.linspace(0.01,0.20,30)

opt_label='no_cachelib'
opt_stations=['dc','cr',
              'di','vs']

#opt_label='with_cachelib'
#opt_stations=['dc','cr','cl',
#              'di','vs']

# wrapper to hand to fmin 
def cost_kmm(params):
    return cost_params(daily_NO3_loss=0.0015,
                       Csat=1.0,k_ni=0.0,kmm=params[0],
                       opt_stations=opt_stations)

res=fmin(cost_kmm,[0.010],full_output=True)
best,best_cost,n_steps, n_funcs,status = res

rmses=[ cost_kmm([kmm]) for kmm in kmms]

plt.figure(20).clf()
fig,ax=plt.subplots(1,1,num=20)
ax.plot(kmms,rmses)
ax.set_ylabel('RMSE (mg/l)')
ax.set_xlabel('k$_{mm}$ (mg-N/l / day)')

ax.plot(best[0],best_cost,'ko')
ax.text(best[0],best_cost,"\nk$_{mm}$=%.4f\nRMSE=%.4f"%(best[0],best_cost),va='top')
ax.axis(ymin=0.01)

# What are the units of kmm?
# Based on original nitrate.py,
# kmm = 1.5*Csat*k_ni ~ mg/l/day, interpreted as the maximum 0th order rate
# k_ni ~ 1/day, i.e. the rate constant for a 1st order reaction
# Csat ~ mg/l, the half-saturation concentration

fig.savefig(os.path.join(fig_dir,'opt_kmm_%s.png'%opt_label),dpi=150)

# Kmm and NO3 loss
def cost_kmm_loss(params):
    return cost_params(daily_NO3_loss=params[1],
                       Csat=1.0,k_ni=0.0,kmm=params[0],
                       opt_stations=opt_stations)

rmse_kmm_loss=np.zeros( (len(kmms),len(losses)), np.float64)
for row,kmm in enumerate(kmms):
    for col,loss in enumerate(losses):
        rmse_kmm_loss[row,col]=cost_kmm_loss([kmm,loss])
res=fmin(cost_kmm_loss,[0.010,0.0015],full_output=True)
best,best_cost,n_steps, n_funcs,status = res

plt.figure(21).clf()
fig,ax=plt.subplots(1,1,num=21)
fig.set_size_inches([6.4,4.8],forward=True)
cset=ax.contourf(losses,kmms,rmse_kmm_loss,
                 np.linspace(0.03,0.10,36),cmap=turbo,extend='both')
plt.colorbar(cset,label='RMSE (mg/l)')
ax.set_xlabel('Loss (day$^{-1}$)')
ax.set_ylabel('k$_{mm}$ (mg/l / day)' )
fig.subplots_adjust(left=0.17,bottom=0.15)

ax.plot(best[1],best[0],'ko')
ax.text(best[1],best[0],"k$_{mm}$=%.4f\nNO3 loss=%.5f\nRMSE=%.4f"%(best[0],best[1],best_cost),va='top')
ax.axis(ymin=0.01)
fig.savefig(os.path.join(fig_dir,'opt_kmm_loss_%s.png'%opt_label),dpi=150)


# First order
def cost_kni_loss(params):
    return cost_params(daily_NO3_loss=params[1],
                       Csat=0,k_ni=params[0],kmm=0.0,method='first',
                       opt_stations=opt_stations)
rmse_kni_loss=np.zeros( (len(knis),len(losses)), np.float64)
for row,kni in enumerate(knis):
    for col,loss in enumerate(losses):
        rmse_kni_loss[row,col]=cost_kni_loss([kni,loss])
res=fmin(cost_kni_loss,[0.010,0.0015],full_output=True)
best,best_cost,n_steps, n_funcs,status = res

plt.figure(22).clf()
fig,ax=plt.subplots(1,1,num=22)
fig.set_size_inches([6.4,4.8],forward=True)
cset=ax.contourf(losses,knis,rmse_kni_loss,
                 np.linspace(0.03,0.10,36),cmap=turbo,extend='both')
plt.colorbar(cset,label='RMSE (mg/l)')
ax.set_xlabel('Loss (day$^{-1}$)')
ax.set_ylabel('k$_{nit}$ (1/day)' )
fig.subplots_adjust(left=0.17,bottom=0.15)
ax.plot(best[1],best[0],'ko')
ax.text(best[1],best[0],"k$_{ni}$=%.4f\nNO3 loss=%.5f\nRMSE=%.4f"%(best[0],best[1],best_cost),va='top')
ax.axis(ymin=0.01)
fig.savefig(os.path.join(fig_dir,'opt_kni_loss_%s.png'%opt_label),dpi=150)


# Show parameter sweep for kmm and Csat
def cost_kmm_Csat(params):
    return cost_params(daily_NO3_loss=0.0015,
                       Csat=params[1],k_ni=0.0,kmm=params[0],
                       opt_stations=opt_stations)

rmse_kmm_Csat=np.zeros( (len(kmms),len(Csats)), np.float64)
for row,kmm in enumerate(kmms):
    for col,Csat in enumerate(Csats):
        rmse_kmm_Csat[row,col]=cost_kmm_Csat([kmm,Csat])
res=fmin(cost_kmm_Csat,[0.010,1.0],full_output=True)
best,best_cost,n_steps, n_funcs,status = res

plt.figure(23).clf()
fig,ax=plt.subplots(1,1,num=23)
fig.set_size_inches([6.4,4.8],forward=True)
cset=ax.contourf(Csats,kmms,rmse_kmm_Csat,
                 np.linspace(0.03,0.10,36),
                 cmap=turbo,extend='both')
plt.colorbar(cset,label='RMSE (mg/l)')
ax.set_xlabel('C$_{sat}$ (mg/l)')
ax.set_ylabel('k$_{mm}$ (mg/l / day)' )
fig.subplots_adjust(left=0.17,bottom=0.15)

ax.plot(best[1],best[0],'ko')
ax.text(best[1],best[0],"k$_{mm}$=%.4f\nCsat=%.3f\nRMSE=%.4f"%(best[0],best[1],best_cost),va='top')
#ax.axis(ymin=0.01)

# Omitting Cache at Liberty

fig.savefig(os.path.join(fig_dir,'opt_kmm_Csat_%s.png'%opt_label),dpi=150)

##

# And optimization on all 3 parameters
def cost_kmm_Csat_loss(params):
    return cost_params(daily_NO3_loss=params[2],
                       Csat=params[1],k_ni=0.0,kmm=params[0],
                       opt_stations=opt_stations)
res=fmin(cost_kmm_Csat_loss,[0.010,1.0,0.0015],full_output=True)
best,best_cost,n_steps, n_funcs,status = res

# kmm=0.053 mg/l/day
# Csat=0.35 mg/l
# NO3 loss=0.0021 1/day
# RMSE, without Cache: 0.0346

##
# evolution equation based on enrichment of NO3 by nitrification

# interpolate all variables to same time steps, corresponding to July-Aug

NO3_1st = {}
NO3_mm  = {}
NH4_1st = {}
NH4_mm  = {}

pred_stations = ['dc','cr','cl','di','vs']

npred = len(pred_stations)
noffset = 0
dt_days = np.diff(age_data['dn']['di'])[0]
model_params=dict(daily_NO3_loss = 0.0015,
                  Csat=1.0,k_ni=0.08,kmm=0.12)

for sta in pred_stations:
    dnum_in=age_data['dn'][sta][noffset:]
    age_in =age_data['age'][sta][noffset:]
    # Vectorized:
    NH4_mm[sta],NO3_mm[sta] = nitri_model(dnum_in,age_in,**model_params)
    NH4_1st[sta],NO3_1st[sta] = nitri_model(dnum_in,age_in,method='first',**model_params)



##         
plot_stations = pred_stations # could change later
methods = ['mm','first']
nplot = len(plot_stations)

for method in methods:

    if method == 'first':
        NO3_plot = NO3_1st
        NH4_plot = NH4_1st
    elif method == 'mm':
        NO3_plot = NO3_mm
        NH4_plot = NH4_mm
    else:
        print( "invalid method")
        sys.exit(0)
    # make time series plot
    fig, ax = plt.subplots(npred+3, 1, sharex="all",figsize=[16,16])
    for ns, sta in enumerate(plot_stations):
        ax[ns].plot_date(age_data['dn'][sta],NO3_age[sta],'-',label='observed')
        ax[ns].plot_date(age_data['dn'][sta],NO3_plot[sta],'-',label='predicted')
        ax[ns].plot_date(age_data['dn'][sta],NO3_age['fp'],'-',label='Freeport')
        ax[ns].legend()
        ax[ns].set_xlim(age_data['dn'][sta][0],age_data['dn'][sta][-1])
        ax[ns].set_ylim(0,0.6)
        ax[ns].set_ylabel('NO3 N')
        ax[ns].set_title(common.label_dict[sta])

    ax[npred].plot(NH4_dn,NH4_fp,label='Freeport')
    for sta in plot_stations:
        ax[npred].plot(age_data['dn'][sta],NH4_plot[sta],label=common.label_dict[sta])
    ax[npred].set_xlim(age_data['dn']['di'][0],age_data['dn']['di'][-1])
    ax[npred].legend()
    ax[npred].set_ylabel('NH4 N')

    for sta in plot_stations:
        ax[npred+1].plot(age_data['dn'][sta],age_data['age'][sta],label=common.label_dict[sta])
    ax[npred+1].set_xlim(age_data['dn']['di'][0],age_data['dn']['di'][-1])
    ax[npred+1].legend()
    ax[npred+1].set_ylabel('Age')

    for sta in plot_stations:
        ax[npred+2].plot(age_data['dn'][sta],age_data['conc'][sta],label=common.label_dict[sta])
    ax[npred+2].set_xlim(age_data['dn']['di'][0],age_data['dn']['di'][-1])
    ax[npred+2].legend()
    ax[npred+2].set_ylabel('Conc')

    fig.autofmt_xdate()
    plt.savefig(os.path.join(fig_dir,'%s.png'%method),bbox_inches='tight')
    plt.close()

    # make residual plot
    res_vars = common.age_vars # do everything for now
    nres_vars = len(res_vars)
    # ymax_dict = {'Age':30,
    fig, axes = plt.subplots(nres_vars, npred+1,
                             sharey='row',figsize=[16,16])
    NO3_res = {}
    xlim_dict = {'age':[0,40],
                 'conc':[0,1],
                 'depth':[0,10],
                 'temperature':[15,22]}
    metrics = {}
    metrics['sta'] = []
    metrics['se'] = np.zeros(nplot+1, np.float64)
    metrics['rmse'] = np.zeros(nplot+1, np.float64)
    metrics['r'] = np.zeros(nplot+1, np.float64)
    metrics['wm'] = np.zeros(nplot+1, np.float64)
    for ns, sta in enumerate(plot_stations):
        metrics['sta'].append(sta)
        NO3_res[sta] = NO3_age[sta] - NO3_plot[sta]
        valid = np.where(np.isfinite(NO3_res[sta]))[0]
        res_val = NO3_res[sta][valid]
        if ns == 0:
            all_res = copy.deepcopy(res_val)
            all_NO3_obs  = copy.deepcopy(NO3_age[sta][valid])
            all_NO3_pred = copy.deepcopy(NO3_plot[sta][valid])
        else:
            all_res = np.concatenate((all_res,res_val))
            all_NO3_obs  = np.concatenate((all_NO3_obs, NO3_age[sta][valid]))
            all_NO3_pred = np.concatenate((all_NO3_pred,NO3_plot[sta][valid]))
    # calculate and print metrics
        metrics['se'][ns] = np.std(NO3_res[sta][valid])
        metrics['rmse'][ns] = np.sqrt(mean_squared_error(NO3_age[sta][valid],NO3_plot[sta][valid]))
        mm, bb, rr, pp, se = stats.linregress(NO3_age[sta][valid], NO3_plot[sta][valid])
        metrics['r'][ns] = rr
        obs_mean = np.mean(NO3_age[sta][valid])
        abs_NO3_pred_diff = np.abs(NO3_plot[sta][valid]-obs_mean)
        abs_NO3_obs_diff  = np.abs(NO3_age[sta][valid]-obs_mean)
        denom = np.sum(np.power(abs_NO3_pred_diff+abs_NO3_obs_diff,2))
        metrics['wm'][ns] = 1.-np.sum(np.power(NO3_res[sta][valid],2))/denom
#        metrics['wm'][ns] = np.power(all_res,2)/denom
    metrics['sta'].append('all')
    metrics['se'][nplot] = np.std(all_res)
    metrics['rmse'][nplot] = np.sqrt(mean_squared_error(all_NO3_obs, all_NO3_pred))
    mm, bb, rr, pp, se = stats.linregress(all_NO3_obs, all_NO3_pred)
    metrics['r'][nplot] = rr
    obs_mean = np.mean(all_NO3_obs)
    abs_NO3_pred_diff = np.abs(all_NO3_pred-obs_mean)
    abs_NO3_obs_diff  = np.abs(all_NO3_obs-obs_mean)
    denom = np.sum(np.power(abs_NO3_pred_diff+abs_NO3_obs_diff,2))
    metrics['wm'][nplot] = 1.-np.sum(np.power(all_res,2))/denom
    for nv, var in enumerate(res_vars):
#       all_data = np.empty(1, dtype=np.float64)
        for ns, sta in enumerate(plot_stations):
            #axes[nv,ns].plot(NO3_res[sta],age_data[var][sta],'.')
            axes[nv,ns].plot(age_data[var][sta],NO3_res[sta],'.')
            if nv == 0:
                axes[nv,ns].set_title(common.label_dict[sta])
            if nv == nres_vars-1:
                axes[nv,ns].set_xlabel('NO3 as N Residual')
            if ns == 0:
                axes[nv,ns].set_ylabel('NO3 as N Residual')
            axes[nv,ns].set_xlabel(common.var_label_dict[var])
            axes[nv,ns].set_xlim(xlim_dict[var])
            valid = np.where(np.isfinite(NO3_res[sta]))[0]
            res_val = NO3_res[sta][valid]
            var_val = age_data[var][sta][valid]
            mm, bb, rr, pp, se = stats.linregress(var_val, res_val)
            x_min = min(var_val)
            x_max = max(var_val)
            xx = np.asarray([x_min,x_max])
            yy = mm*xx + bb
            axes[nv,ns].plot(xx,yy,'-')
            if ns == 0:
                all_data = copy.deepcopy(var_val)
            else:
                all_data = np.concatenate((all_data,var_val))
        axes[nv,npred].plot(all_data,all_res,'.')
        if nv == 0:
            axes[nv,npred].set_title('all stations')
        axes[nv,npred].set_xlabel(common.var_label_dict[var])
        axes[nv,npred].set_xlim(xlim_dict[var])
        mm, bb, rr, pp, se = stats.linregress(all_data, all_res)
        x_min = min(all_data)
        x_max = max(all_data)
        xx = np.asarray([x_min,x_max])
        yy = mm*xx + bb
        axes[nv,npred].plot(xx,yy,'-')

    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir,'%s_residual.png'%method),bbox_inches='tight')
    plt.close()
    sequence=['sta','rmse','se','r','wm']
    pd.DataFrame.from_dict(metrics).to_csv('%s_metrics.csv'%method,columns=sequence,index=False)

