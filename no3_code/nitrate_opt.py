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
                k_ni,kmm,daily_NO3_loss,Csat,method='MM'):
    # These could be preprocessed
    dn_lagged = dnums_in - age_in
    n_NO3=np.searchsorted(dn_NO3['fp'],dn_lagged)
    n_NH4 = np.searchsorted(NH4_dn,dn_lagged)
    NH4_lag = NH4_fp[n_NH4] # lagged boundary NH4
    NO3_lag = NO3['fp_lp'][n_NO3] # lagged boundary NO3

    # Steps dependent on model parameters
    if method=='first': # First order model:
        NH4_atten=1.-np.exp(-k_ni*age_in) # fraction of NH4 nitrified
        nit_flux=NH4_lag*NH4_atten
    elif method=='MM': # Michaelis Menten closed form 
        F=NH4_lag/Csat*np.exp(NH4_lag/Csat-kmm/Csat*uw_age)
        # For F>=0, lambertw on principal branch is real. safe to cast.
        # Write this as flux to make mass conservation clear (minor overhead)
        nit_flux=NH4_lag-Csat*np.real(lambertw(F))
    else:
        raise Exception("bad method %s"%method)
        
    # add in loss term
    NO3_pred = NO3_lag + nit_flux - daily_NO3_loss*age_in
    NH4_pred = NH4_lag - nit_flux
    return NH4_pred,NO3_pred

# now that necessary data is loaded in, plot maps


##

# Run model
dnums = uw_df_thin['dnums'].values
uw_age = uw_df_thin['age'].values # RH: used to be just 'age'

# 3.2ms
NH4_mm,NO3_mm = nitri_model(dnums,uw_age,k_ni=0.08,kmm=0.12,Csat=1.0,daily_NO3_loss=0.0015)

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
plt.figure(2).clf()
fig,axs=plt.subplots(1,2,num=2)
fig.set_size_inches([12,7],forward=True)
x = uw_df_thin['x'].values
y = uw_df_thin['y'].values

in_corridor=np.array([common.sac_poly.contains( geometry.Point(pnt) )
                      for pnt in np.c_[x,y] ])

NO3_obs = uw_df_thin['NO3-N'].values
NH4_obs = uw_df_thin['x'].values

for ax in axs:
    # Coloring by dnum wasn't that helpful.
    scat1=ax.scatter(NO3_obs[~in_corridor], NO3_mm[~in_corridor], s=5, color='0.85',label="Off axis")
    scat2=ax.scatter(NO3_obs[in_corridor], NO3_mm[in_corridor], s=20, color='tab:blue',label="Sac corridor")
    ax.set_xlabel('Observed NO$_3$')
    ax.set_ylabel('Predicted NO$_3$')

axs[1].plot([0,0.6],[0,0.6],color='tab:red',lw=0.5)

axs[1].axis(xmax=0.45,xmin=0,ymin=0.13,ymax=0.48)
axs[0].axis(xmax=2.00,xmin=0,ymin=0.13,ymax=0.48)
axs[1].legend(loc='lower right')
plt.ion()
plt.show()

fig.savefig(os.path.join(fig_dir,'obs_vs_model_scatter.png'),dpi=150)


## 

# evolution equation based on enrichment of NO3 by nitrification

# interpolate all variables to same time steps, corresponding to July-Aug

NO3_pred = {}
NO3_mm = {}
NH4_pred = {}
NH4_mm = {}
NH4_lag = {}
NO3_lag = {}
NH4_atten = {}
NH4_atten_mm = {}
pred_stations = ['dc','cr','cl','di','vs']
#pred_stations = ['vs']
npred = len(pred_stations)
noffset = 0
dt_days = np.diff(age_data['dn']['di'])[0]
daily_NO3_loss = 0.0015
for sta in pred_stations:
    NO3_pred[sta] = np.zeros_like(age_data['dn'][sta])
    NH4_pred[sta] = np.zeros_like(age_data['dn'][sta])
    NH4_lag[sta] = np.zeros_like(age_data['dn'][sta]) # lagged boundary NH4
    NO3_lag[sta] = np.zeros_like(age_data['dn'][sta]) # lagged boundary NO3
    NH4_atten[sta] = np.zeros_like(age_data['dn'][sta])
    NO3_mm[sta] = np.zeros_like(age_data['dn'][sta])
    NH4_mm[sta] = np.zeros_like(age_data['dn'][sta])
    for nloop, dn in enumerate(age_data['dn'][sta][noffset:]):
        n = nloop + noffset
        dn_lagged = age_data['dn'][sta][n] - age_data['age'][sta][n] # RH: age =>age_data['age']
        n_NO3 = np.where(dn_NO3['fp']>=dn_lagged)[0][0]
        #pdb.set_trace()
        n_NH4 = np.where(NH4_dn>=dn_lagged)[0][0]
        NH4_lag[sta][n] = NH4_fp[n_NH4]
        NH4_atten[sta][n] = 1.-math.exp(-k_ni*age_data['age'][sta][n])
        NO3_lag[sta][n] = NO3['fp_lp'][n_NO3]
        NO3_pred[sta][n] = NO3_lag[sta][n] + NH4_lag[sta][n]*NH4_atten[sta][n]
        NH4_pred[sta][n] = NH4_fp[n_NH4]*math.exp(-k_ni*age_data['age'][sta][n])
        # Michaelis Menten
        # Analytical MM:
        F=NH4_lag[sta][n]/Csat*np.exp(NH4_lag[sta][n]/Csat-kmm/Csat*age_data['age'][sta][n])
        NH4_mm[sta][n] = Csat*np.real(lambertw(F))
        NO3_mm[sta][n] = NO3_lag[sta][n] + NH4_lag[sta][n] - NH4_mm[sta][n]
        # add in loss term
        NO3_mm[sta][n] -= daily_NO3_loss*age_data['age'][sta][n]


##         
plot_stations = pred_stations # could change later
methods = ['mm','first']
nplot = len(plot_stations)

for method in methods:

    if method == 'first':
        NO3_plot = NO3_pred
        NH4_plot = NH4_pred
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

