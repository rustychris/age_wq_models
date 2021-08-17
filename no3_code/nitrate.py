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

grd_fn=r"LTO_Restoration_2018_h.grd"

def load_underway(extract_from_nc=0,thin='mean'):
    if extract_from_nc: # get untrim age etc. for each x,y,t and save in dataframe
        var_list = {'conc':['Mesh2_scalar2_3d', 'kgm-3', 1.],
                    'age-conc':['Mesh2_scalar3_3d', 'd*kgm-3', 86400.],
                    'depth-age-conc':['Mesh2_scalar4_3d', 'd*mkgm-3', 86400.],
                    'temperature-age-conc':['Mesh2_scalar5_3d', 'deg*mkgm-3', 86400.],
                    'depth':['compute', 'm', 'depth-age-conc', 'age-conc'],
                    'temperature':['compute', 'degrees Celsius', 'temperature-age-conc', 'age-conc'],
                    'age': ['compute', 'days','age-conc', 'conc']}
        #ncfile=r"\\compute16\E\temp\UnTRIM\temperature_2018_16_Febstart_SacReg_age_v8\untrim_hydro_Oct2018_time_chunk_new.nc"
        ncfile=r"\\compute16\E\temp\UnTRIM\temperature_2018_16_Febstart_SacReg_age_v8\untrim_hydro_Oct2018.nc"
        # read continuous underway data
        underway_dir = r'R:\UCD\Projects\CDFW_Holleman\Observations\Underway_measurements\2018'
        underway_csv = 'Delta Water Quality Mapping October 2018 High Resolution_edited.csv'
        underway_csv = os.path.join(underway_dir,underway_csv)
        underway_df = pd.read_csv(underway_csv)
        crs_wgs = pyproj.Proj(proj='latlon',datum='WGS84',ellps='WGS84')
        crs_utm = pyproj.Proj(init='epsg:26910',preserve_units=True)
        valid = np.where(np.isfinite(underway_df['Latitude (Decimal Degrees, NAD83)']))[0]
        uw_df = copy.deepcopy(underway_df.iloc[valid])
        nrows = len(uw_df)
        #uw_df['x'] = np.zeros(nrows, np.float64)
        #uw_df['y'] = np.zeros(nrows, np.float64)
        lat = uw_df['Latitude (Decimal Degrees, NAD83)'].values
        lon = uw_df['Longitude (Decimal Degrees, NAD83)'].values
        uw_df['x'], uw_df['y'] = pyproj.transform(crs_wgs,crs_utm,lon,lat)
        x = uw_df['x'].values
        y = uw_df['y'].values
        xy = np.asarray(zip(x,y))
        dstring = uw_df['Timestamp (PST)'].values
        #uw_df['NO3-N'] = np.zeros(nrows, np.float64)
        dtimes = []
        NO3_um = uw_df['NO3 (uM) (SUNA) HR'].values
        uw_df['NO3-N'] = NO3_um*N_g/1000.
        NH4_um = uw_df['NH4 (uM) (Tline) HR'].values
        uw_df['NH4-N'] = NH4_um*N_g/1000.
        print("reading grid")
        grd = unstructured_grid.UnTRIM08Grid(grd_fn)
        print("read grid")
        cells_i = np.zeros(nrows, np.int32)
        indices = []
        for nr in range(nrows):
            dtime_row = dt.datetime.strptime(dstring[nr], '%m/%d/%Y %H:%M')
            dtimes.append(dtime_row)
            cells_i[nr] = grd.select_cells_nearest(xy[nr,:],inside=False)
        dtimes = np.asarray(dtimes)
        uw_df['dtimes'] = dtimes
        uw_df['dnums']=d2n(uw_df['dtimes'].values)
        uw_df['i']=cells_i
        uw_df['selected']=np.zeros(nrows, np.int32)
        # read nc
        nc = Dataset(ncfile, 'r')
        t = nc.variables['Mesh2_data_time']
        time = num2date(t[:], t.units)
        row_n = np.zeros(nrows, np.int32)
        for nr in range(nrows):
            nearest_dtime = min(time,key=lambda x: abs(x - dtimes[nr]))
            row_n[nr] = np.argwhere(time==nearest_dtime)[0][0]
        uw_df['n']=row_n
        extract_vars = ['conc','age','temperature','depth']
        #extract_vars = ['age']
        var_arrays = {}
        pdb.set_trace()
        @memoize(lru=10)
        def kbi_for_time(n):
            return nc.variables['Mesh2_face_bottom_layer'][:, n]
        @memoize(lru=10)
        def kti_for_time(n):
            return nc.variables['Mesh2_face_top_layer'][:, n]
        @memoize(lru=10)
        def vol_for_time(n):
            return nc.variables['Mesh2_face_water_volume'][:, :, n]
        @memoize(lru=10)
        def var_for_time(var, n):
            return nc.variables[var][:, :, n]
        for var in extract_vars:
            var_arrays[var]=np.zeros(nrows, np.float64)
            last_i = 0
            last_n = 0
            for nr in range(nrows):
                i = cells_i[nr]
                n = row_n[nr]
                if (last_n == n) and (last_i ==i):
                    var_arrays[var][nr] = data
                    #uw_df[var].iloc[nr] = data
                    print(var,nr,i,n,data )
                    continue
                else:
                    indices.append(nr)
                #kbi = nc.variables['Mesh2_face_bottom_layer'][i, n] - 1
                kbi = kbi_for_time(n)[i] - 1
                #kti = nc.variables['Mesh2_face_top_layer'][i, n] - 1
                kti = kti_for_time(n)[i] - 1
                #vol = nc.variables['Mesh2_face_water_volume'][i, :, n]
                vol = vol_for_time(n)[i, :]
                vol_sum = np.sum(vol[kbi:kti+1])
                if var_list[var][0] != 'compute':
                    scale = var_list[var][2]
                    data_k = var_for_time(var_list[var][0], n)[i, :]
                    #data_k = nc.variables[var_list[var][0]][i, :, n]
                    data = np.sum(data_k[kbi:kti+1] * vol[kbi:kti+1])/vol_sum
                    data = data / scale
                    var_arrays[var][nr] = data
                    #uw_df[var].iloc[nr] = data
                else:
                    new_var = var_list[var_list[var][2]][0]
                    new_var_scale = var_list[var_list[var][2]][2] # numerator variable
                    #data_k = nc.variables[new_var][i, :, n]  
                    data_k = var_for_time(new_var, n)[i, :]
                    #convert units of numerator
                    data = np.sum(data_k[kbi:kti+1] * vol[kbi:kti+1])/vol_sum
                    numerator = data/new_var_scale
    #               data_main = nc.variables[new_var][i, :, n] / new_var_scale #convert units of numerator
                    new_var_divider = var_list[var_list[var][3]][0] # denominator variable
                    divider_scale = var_list[var_list[var][3]][2] # scaling factor for denominator
                    data_divider_k = var_for_time(new_var_divider, n)[i, :]
                    #data_divider_k = nc.variables[new_var_divider][i, :, n] # denominator value
                    data_divider = np.sum(data_divider_k[kbi:kti+1] * vol[kbi:kti+1])/vol_sum
                    denominator = data_divider / divider_scale # convert units of denominator
                    if denominator < 1.e-20:
                        data = 0.0
                    else:
                        data = numerator / denominator 
                    var_arrays[var][nr] = data
                    #uw_df[var].iloc[nr] = data
                last_i = i
                last_n = n
                print(var,nr,i,n,data)
            uw_df[var] = var_arrays[var]
            iarray = np.asarray(indices)
            uw_df['selected'].values[iarray]=1

        pdb.set_trace()
        uw_df.to_csv('uw_df.csv',index=False)
    else:
        uw_df = pd.read_csv('uw_df.csv')
        # Grab the grid, too
        # What's slow in reading this grid? 25s. No single hot spot, just
        # lots of slow file reading/parsing.
        grd = unstructured_grid.UnTRIM08Grid(grd_fn)

    # Include the thinning in here, too

    if thin=='first': # thin out the datafrom for unique i,n combinations, taking first
        indices = np.where(uw_df['selected']==1)[0]
        uw_df_thin = uw_df.iloc[indices]
    elif thin=='mean':
        # Alternative: thin dataframe by grouping/mean
        # creates slightly fewer rows, due to the ship passing
        # back and forth over a boundary in a short time span.
        # For our purposes I think it's reasonable to combine them.
        grped=uw_df.groupby(['i','n'])
        uw_df_grped=grped.mean() # this drops the object fields dtimes and Timestamp (PST)
        uw_df_grped['dtimes']=grped['dtimes'].first()
        uw_df_grped['Timestamp (PST)']=grped['Timestamp (PST)'].first()
        uw_df_thin=uw_df_grped.reset_index()
    else:
        uw_df_thin=uw_df # no thinningn
        
    return uw_df_thin,grd

# Load underway data, hydro grid, and thin data by unique time step/cell
uw_df_thin,grd=load_underway()
grd_poly=grd.boundary_polygon()
nrows = len(uw_df_thin)

## ###############

# Read tracer outputs, including age, depth, temperature
# read in information needed for NO3 predictions
# read age first because will use selected time for interpolation etc.
age_dss_fname = 'AgeScalars.dss'
age_vars = ['age','conc','depth','temperature']
var_label_dict = {'age':'Age [days]',
                  'conc':'Concentration',
                  'depth':'Depth [m]',
                  'temperature':'Temperature [$^\circ$C]'}
age_stations=['dc','cr','cl','di','vs']
# File also has DWSC, but I think Ed was omitting for a reason
label_dict = {'dc':'Sac ab DCC',
              'cr':'Cache ab Ryer',
              'cl':'Cache Lib',
              'di':'Sac Decker',
              'vs':'Van Sickle'}
rec_template = '/UT/$STA$/$VAR$//30MIN/TEMPERATURE_2018_16_FEBSTART_SAC/'
#rec_template = '/UT/%(STA)s/$VAR$//30MIN/TEMPERATURE_2018_16_FEBSTART_SAC/'
#rec_template%dss_sta_dict

age_data = {}
for var in age_vars:
    dss_rec = {}
    dss_sta_dict = {} # overwrite each time
    for sta in age_stations:
        dss_sta = label_dict[sta].upper().replace(" ","_")
        dss_sta_dict[sta] = dss_sta
        dss_rec[sta] = rec_template.replace("$STA$",dss_sta).replace("$VAR$",var)
    dn_age, age_data[var] = get_var_dss(stations=age_stations, 
                                        dss_records=dss_rec,
                                        dss_fname=age_dss_fname)
    for sta in age_stations:
        valid = np.where(np.isfinite(age_data[var][sta]))
        dn_age[sta] = dn_age[sta][valid]
        age_data[var][sta] = age_data[var][sta][valid]
# put in loop
# save in short names also 
age = age_data['age']
conc = age_data['conc']


#dn_age_vs, age_vs = get_var_dss_light(age_dss_fname, age_dss_rec['vs'],
#                                      tstart=date_start_dt, tend=date_end_dt)
#pdb.set_trace() # RH: what is age here? dict of stations, arrays
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

# Changed to the optimized values ( fn(kmm,Csat,loss), ignoring Cache-Lib)
Csat=0.35
k_ni = 0.08
#k_ni = 0.09
#k_ni = 0.05
#kmm = 2*Csat*k_ni
kmm = 0.053 # 1.5*Csat*k_ni
daily_NO3_loss=0.0021

#def dCdtMM(t,C):
#    return -kmm*C/(Csat+C)

# put the flow on same time steps as age
flow_age = {}
for sta in flow_stations:
    f_interp = interp1d(dn_flow[sta],flow[sta])
    flow_age[sta] = f_interp(dn_age['di'])

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
    NO3_age[sta] = f_interp(dn_age['di'])

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
if 0:
    fig=plt.figure(22)
    fig.clf()
    ax=fig.add_subplot()
    for sta in ['fp','dc']:
        ax.plot(dn_NO3[sta],NO3[sta],label=sta)
    ax.plot(dn_NO3['fp'],NO3['fp_fill'],label='fp fill')
    ax.plot(dn_NO3['fp'],NO3['fp_lp'],label='fp lp')
    ax.plot(usgs_df_wide.index.values, usgs_df_wide['fp_lag'],label='fp lag')
    plt.setp(ax.lines,lw=1.0)
    ax.legend()
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.axis(xmin=np.datetime64('2018-10-01'),
            xmax=np.datetime64('2018-11-05'))

    ax.legend()
    ax.axis((736920.6060114561, 737038.9861075104, -0.006265952088709725, 0.5495188518367599))
    fig.savefig('freeport-lag-filled.png')
    
##
# now that necessary data is loaded in, plot maps
# daily_NO3_loss = 0.0015 # defined above
NO3_pred = np.zeros(nrows, np.float64)
NH4_pred = np.zeros_like(NO3_pred)
NH4_lag = np.zeros_like(NO3_pred) # lagged boundary NH4
NO3_lag = np.zeros_like(NO3_pred) # lagged boundary NO3
NH4_atten = np.zeros_like(NO3_pred)
NO3_mm = np.zeros_like(NO3_pred)
NH4_mm = np.zeros_like(NO3_pred)
dnums = uw_df_thin['dnums'].values
uw_age = uw_df_thin['age'].values # RH: This overwrites age from above. Renaming to uw_age
for nr in range(nrows):
    dn_lagged = dnums[nr] - uw_age[nr]
    n_NO3 = np.where(dn_NO3['fp']>=dn_lagged)[0][0]
    n_NH4 = np.where(NH4_dn>=dn_lagged)[0][0]
    NH4_lag[nr] = NH4_fp[n_NH4]
    NH4_atten[nr] = 1.-math.exp(-k_ni*uw_age[nr])
    NO3_lag[nr] = NO3['fp_lp'][n_NO3]
    NO3_pred[nr] = NO3_lag[nr] + NH4_lag[nr]*NH4_atten[nr]
    NH4_pred[nr] = NH4_fp[n_NH4]*math.exp(-k_ni*uw_age[nr])
    # Michaelis Menten
    # Analytical MM:
    F=NH4_lag[nr]/Csat*np.exp(NH4_lag[nr]/Csat-kmm/Csat*uw_age[nr])
    NH4_mm[nr] = Csat*lambertw(F)
    NO3_mm[nr] = NO3_lag[nr] + NH4_lag[nr] - NH4_mm[nr]
    # add in loss term
    NO3_mm[nr] -= daily_NO3_loss*uw_age[nr]


## RH
if 0:
    # Did the change in FP do anything for the underway predictions?
    # Have to run the code above once with the
    # dcc fill in enabled, save the result,
    # NO3_mm_with_dcc=NO3_mm.copy()
    # and run again without.
    plt.figure(23).clf()
    fig,axs=plt.subplots(1,2,num=23)
    axs[0].plot( NO3_mm, NO3_mm_with_dcc, 'g.')
    axs[0].plot([0.14,0.45],[0.14,0.45],'k-',lw=0.9)

    axs[0].set_xlabel('NO3_mm, Filled FP data')
    axs[0].set_ylabel('NO3_mm, Lagged DCC data')
    
    scat = axs[1].scatter(uw_df_thin['x'].values,uw_df_thin['y'].values, c=NO3_mm_with_dcc-NO3_mm,
                          cmap='coolwarm', s=60, vmin=-0.01, vmax=0.01)

    plot_wkb.plot_wkb(grd_poly,fc='0.8',ec='0.8',zorder=-2,ax=axs[1])
    [axs[1].axis(z) for z in ['tight','equal','off',(570257.0603246342, 656121.6409803774, 4196446.208894607, 4257788.628702029)]]
    cax=fig.add_axes([0.55,0.9,0.20,0.02])
    plt.colorbar(scat,cax=cax,label='NO3 lagged DC - fill FP\n mg-N/l',orientation='horizontal')
    fig.tight_layout()
    fig.savefig('compare_fill_vs_lag.png',dpi=150)

## 
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
fig.savefig('obs_vs_model_map.png',dpi=150)

##

from stompy.plot import plot_utils
if 0:
    sac_corridor=plot_utils.draw_polyline()
else:
    # Follows Georgiana, tho.
    sac_corridor=np.array([[ 628277, 4270210],[ 630351, 4270136],[ 631092, 4267690],[ 631907, 4255316],
                           [ 628795, 4238126],[ 631611, 4235088],[ 629240, 4230790],[ 628128, 4228493],
                           [ 624720, 4222417],[ 626128, 4220120],[ 627091, 4216563],[ 623460, 4216415],
                           [ 614495, 4215526],[ 598786, 4211376],[ 595451, 4209820],[ 576557, 4209524],
                           [ 575001, 4212858],[ 583522, 4215378],[ 589598, 4213451],[ 592265, 4213525],
                           [ 596933, 4212710],[ 601231, 4215155],[ 604936, 4214488],[ 613383, 4221972],
                           [ 615754, 4221083],[ 626202, 4234939],[ 621534, 4242645],[ 625016, 4269691]])
sac_poly=geometry.Polygon(sac_corridor)

# Scatter of the same thing
plt.figure(2).clf()
fig,axs=plt.subplots(1,2,num=2)
fig.set_size_inches([12,7],forward=True)
x = uw_df_thin['x'].values
y = uw_df_thin['y'].values

in_corridor=np.array([sac_poly.contains( geometry.Point(pnt) )
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

fig.savefig('obs_vs_model_scatter.png',dpi=150)


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
dt_days = np.diff(dn_age['di'])[0]
daily_NO3_loss = 0.0015
for sta in pred_stations:
    NO3_pred[sta] = np.zeros_like(dn_age[sta])
    NH4_pred[sta] = np.zeros_like(dn_age[sta])
    NH4_lag[sta] = np.zeros_like(dn_age[sta]) # lagged boundary NH4
    NO3_lag[sta] = np.zeros_like(dn_age[sta]) # lagged boundary NO3
    NH4_atten[sta] = np.zeros_like(dn_age[sta])
    NO3_mm[sta] = np.zeros_like(dn_age[sta])
    NH4_mm[sta] = np.zeros_like(dn_age[sta])
    for nloop, dn in enumerate(dn_age[sta][noffset:]):
        n = nloop + noffset
        dn_lagged = dn_age[sta][n] - age[sta][n] # RH: age is just an array though
        n_NO3 = np.where(dn_NO3['fp']>=dn_lagged)[0][0]
        #pdb.set_trace()
        n_NH4 = np.where(NH4_dn>=dn_lagged)[0][0]
        NH4_lag[sta][n] = NH4_fp[n_NH4]
        NH4_atten[sta][n] = 1.-math.exp(-k_ni*age[sta][n])
        NO3_lag[sta][n] = NO3['fp_lp'][n_NO3]
        NO3_pred[sta][n] = NO3_lag[sta][n] + NH4_lag[sta][n]*NH4_atten[sta][n]
        NH4_pred[sta][n] = NH4_fp[n_NH4]*math.exp(-k_ni*age[sta][n])
        # Michaelis Menten
        # Analytical MM:
        F=NH4_lag[sta][n]/Csat*np.exp(NH4_lag[sta][n]/Csat-kmm/Csat*age[sta][n])
        NH4_mm[sta][n] = Csat*lambertw(F)
        NO3_mm[sta][n] = NO3_lag[sta][n] + NH4_lag[sta][n] - NH4_mm[sta][n]
        # add in loss term
        NO3_mm[sta][n] -= daily_NO3_loss*age[sta][n]

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
        ax[ns].plot_date(dn_age[sta],NO3_age[sta],'-',label='observed')
        ax[ns].plot_date(dn_age[sta],NO3_plot[sta],'-',label='predicted')
        ax[ns].plot_date(dn_age[sta],NO3_age['fp'],'-',label='Freeport')
        ax[ns].legend()
        ax[ns].set_xlim(dn_age[sta][0],dn_age[sta][-1])
        ax[ns].set_ylim(0,0.6)
        ax[ns].set_ylabel('NO3 N')
        ax[ns].set_title(label_dict[sta])

    ax[npred].plot(NH4_dn,NH4_fp,label='Freeport')
    for sta in plot_stations:
        ax[npred].plot(dn_age[sta],NH4_plot[sta],label=label_dict[sta])
    ax[npred].set_xlim(dn_age['di'][0],dn_age['di'][-1])
    ax[npred].legend()
    ax[npred].set_ylabel('NH4 N')

    for sta in plot_stations:
        ax[npred+1].plot(dn_age[sta],age[sta],label=label_dict[sta])
    ax[npred+1].set_xlim(dn_age['di'][0],dn_age['di'][-1])
    ax[npred+1].legend()
    ax[npred+1].set_ylabel('Age')

    for sta in plot_stations:
        ax[npred+2].plot(dn_age[sta],conc[sta],label=label_dict[sta])
    ax[npred+2].set_xlim(dn_age['di'][0],dn_age['di'][-1])
    ax[npred+2].legend()
    ax[npred+2].set_ylabel('Conc')

    fig.autofmt_xdate()
    plt.savefig('%s.png'%method,bbox_inches='tight')
    plt.close()

    # make residual plot
    res_vars = age_vars # do everything for now
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
                axes[nv,ns].set_title(label_dict[sta])
            if nv == nres_vars-1:
                axes[nv,ns].set_xlabel('NO3 as N Residual')
            if ns == 0:
                axes[nv,ns].set_ylabel('NO3 as N Residual')
            axes[nv,ns].set_xlabel(var_label_dict[var])
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
        axes[nv,npred].set_xlabel(var_label_dict[var])
        axes[nv,npred].set_xlim(xlim_dict[var])
        mm, bb, rr, pp, se = stats.linregress(all_data, all_res)
        x_min = min(all_data)
        x_max = max(all_data)
        xx = np.asarray([x_min,x_max])
        yy = mm*xx + bb
        axes[nv,npred].plot(xx,yy,'-')

    fig.tight_layout()
    # pdb.set_trace()
    plt.savefig('%s_residual.png'%method,bbox_inches='tight')
    plt.close()
    sequence=['sta','rmse','se','r','wm']
    pd.DataFrame.from_dict(metrics).to_csv('%s_metrics.csv'%method,columns=sequence,index=False)

