# proof of concept NO3 prediction along mainstem sacramento
# strategy:
# - make really bad predictions of NO3 based only on nitrification
#   - assume a constant nitrification rate for now
# - also estimate NH3 at stations
#   - if NH3 much less than NH3 at Freeport, assumption of constant nitrification is questionable
# - examine residuals to understand what might be missing
#   - temperature experienced by tracer
# - estimate optimal rate
#   

import os.path, sys
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from pylab import date2num as d2n
from pylab import num2date as n2d
from stompy.utils import fill_invalid
from stompy.filters import lowpass_godin
from scipy.interpolate import interp1d
import math
import pandas as pd
from obsdata_utils import save_current_file, get_var, get_var_dss
from obsdata_utils import get_var_dss_light
from DSS_IO import (DSS_IO, bad_data_val, get_dss_interval_for_times,
                    split_DSS_record_name, create_DSS_record_name)
import pdb

plt.style.use('meps.mplstyle')

# read age first because will use selected time for interpolation etc.
age_dss_fname = 'AgeScalars.dss'
age_dss_rec = {'di':'/UT/SAC_DECKER/AGE/01SEP2018/30MIN/TEMPERATURE_2018_16_FEBSTART_SAC/',
               'vs':'/UT/MALLARD_UPPER/AGE/01SEP2018/30MIN/TEMPERATURE_2018_16_FEBSTART_SAC/'}
age_stations=['di','vs']
dn_age, age = get_var_dss(stations=age_stations, dss_records=age_dss_rec,
                                                 dss_fname=age_dss_fname)
conc_dss_rec = {'di':'/UT/SAC_DECKER/CONC/01SEP2018/30MIN/TEMPERATURE_2018_16_FEBSTART_SAC/',
                'vs':'/UT/MALLARD_UPPER/CONC/01SEP2018/30MIN/TEMPERATURE_2018_16_FEBSTART_SAC/'}
dn_conc, conc = get_var_dss(stations=age_stations, dss_records=conc_dss_rec,
                                                   dss_fname=age_dss_fname)
#date_start_dt=dt.datetime(2018,9,1)
#date_end_dt=dt.datetime(2018,10,1)
#dn_age_vs, age_vs = get_var_dss_light(age_dss_fname, age_dss_rec['vs'],
#                                      tstart=date_start_dt, tend=date_end_dt)
for sta in age_stations:
    valid = np.where(np.isfinite(age[sta]))
    dn_age[sta] = dn_age[sta][valid]
    age[sta] = age[sta][valid]

# read flow data from UnTRIM input
# low-pass filter flow
flow_dss_fname = 'UnTRIM_flow_BCs_2011-2019.dss'
flow_dss_rec = {'sr':'//SACWWTP/FLOW//1HOUR/CIWQS-MERGED/',
                'fp':'/FILLED+SHIFT-7HRS/FREEPORT/FLOW-GODIN//15MIN/USGS-MERGED/'}
flow_stations = ['sr', 'fp']
dn_flow, flow = get_var_dss(stations=flow_stations, dss_records=flow_dss_rec,
                                                    dss_fname=flow_dss_fname)
flow['sr_lp'] = lowpass_godin(flow['sr'], dn_flow['sr'], ends='nan')
# dilute NH3 at Freeport
# NH3 in "ambient" water based on underway mapping in July 2020
# NH3_sr - typical Sac Regional NH3 in July/Aug from Kraus spreadsheet for 2020
NH3_sr = 30 # mg/L as N
NH3_am_uM = 1.0 # uM
N_g = 14.0067
NH3_am = (NH3_am_uM/1000.)*N_g 

# put the flow on same time steps as age
flow_age = {}
for sta in flow_stations:
    f_interp = interp1d(dn_flow[sta],flow[sta])
    flow_age[sta] = f_interp(dn_age['di'])

# assume fp and sr are same length (true as of now)
NH3_start_dt = dt.datetime(2018,7,1)
NH3_end_dt = dt.datetime(2018,10,1)
NH3_dt = []
NH3_dt.append(NH3_start_dt)
while NH3_dt[-1] < NH3_end_dt:
    NH3_dt.append(NH3_dt[-1]+dt.timedelta(hours=0.25))
NH3_dt = np.asarray(NH3_dt)
NH3_dn = d2n(NH3_dt)

NH3_fp = np.zeros_like(NH3_dn)
for n, dn in enumerate(NH3_dn):
    nflow_fp = np.where(dn_flow['fp']>=dn)[0][0]
    nflow_sr = np.where(dn_flow['sr']>=dn)[0][0]
    flow_am = flow['fp'][nflow_fp] - flow['sr_lp'][nflow_sr] # ambient flow
    NH3_fp[n] = (NH3_sr*flow['sr_lp'][nflow_sr] + NH3_am*flow_am)/flow['fp'][nflow_fp]
#NH3_fp = np.zeros_like(flow_age['fp'])
#for n, dn in enumerate(dn_age['di']):
#    flow_am = flow_age['fp'][n] - flow_age['sr'][n] # ambient flow
#    NH3_fp[n] = (NH3_sr*flow_age['sr'][n] + NH3_am*flow_am)/flow_age['fp'][n]

# read USGS continuous monitoring NO3 data
usgs_dss_fname = 'chla_data.dss'
NO3_dss_rec = {'fp':'/SAC R FREEPORT/11447650/INORGANIC_NITRO//15MIN/USGS/',
               'di':'/SAC R A DECKER ISL RV/11455478/INORGANIC_NITRO//15MIN/USGS/',
               'vs':'/SUISUN BAY VAN SICKLE/11455508/INORGANIC_NITRO//15MIN/USGS/'}
NO3_stations = ['fp','di','vs']
dn_NO3, NO3 = get_var_dss(stations=NO3_stations, dss_records=NO3_dss_rec,
                                                       dss_fname=usgs_dss_fname)
# put the NO3 on same time steps as age
NO3_age = {}
for sta in NO3_stations:
    f_interp = interp1d(dn_NO3[sta],NO3[sta])
    NO3_age[sta] = f_interp(dn_age['di'])

# fill NO3 data at Freeport
NO3['fp_fill'] = fill_invalid(NO3['fp'])

# evolution equation based on enrichment of NO3 by nitrification

# interpolate all variables to same time steps, corresponding to July-Aug

NO3_pred = {}
NH3_pred = {}
pred_stations = ['di','vs']
noffset = 0
dt_days = np.diff(dn_age['di'])[0]
k_ni = 0.07
for sta in pred_stations:
    NO3_pred[sta] = np.zeros_like(dn_age[sta])
    NH3_pred[sta] = np.zeros_like(dn_age[sta])
    for nloop, dn in enumerate(dn_age[sta][noffset:]):
        n = nloop + noffset
        n_age = int(age[sta][n]/dt_days)
        age_lagged = dn_age[sta][n] - age[sta][n]
        n_NO3 = np.where(dn_NO3['fp']>=age_lagged)[0][0]
        n_NH3 = np.where(NH3_dn>=age_lagged)[0][0]
        NO3_pred[sta][n] = NO3['fp'][n_NO3] + NH3_fp[n_NH3]*(1.-math.exp(-k_ni*age[sta][n]))
        NH3_pred[sta][n] = NH3_fp[n_NH3]*math.exp(-k_ni*age[sta][n])

fig, ax = plt.subplots(5, 1, sharex="all")
fig.set_size_inches(10, 8)

plt.ion()
ax[0].plot_date(dn_age['di'],NO3_age['di'],'-',label='observed Decker Island')
ax[0].plot_date(dn_age['di'],NO3_pred['di'],'-',label='predicted Decker Island')
ax[0].plot_date(dn_age['di'],NO3_age['fp'],'-',label='Freeport')
ax[0].legend()
ax[0].set_xlim(dn_age['di'][0],dn_age['di'][-1])
ax[0].set_ylabel('NO3 mg l$^{-1}$ N')

ax[1].plot(dn_age['vs'],NO3_age['vs'],label='observed Van Sickle')
ax[1].plot(dn_age['vs'],NO3_pred['vs'],label='predicted Van Sickle')
ax[1].plot(dn_age['vs'],NO3_age['fp'],label='Freeport')
ax[1].legend()
ax[1].set_xlim(dn_age['di'][0],dn_age['di'][-1])
ax[1].set_ylabel('NO3 mg l$^{-1}$ N')

ax[2].plot(NH3_dn,NH3_fp,label='Freeport')
ax[2].plot(dn_age['di'],NH3_pred['di'],label='Decker Island')
ax[2].plot(dn_age['di'],NH3_pred['vs'],label='Van Sickle')
ax[2].set_xlim(dn_age['di'][0],dn_age['di'][-1])
ax[2].legend()
ax[2].set_ylabel('NH3 mg l$^{-1}$ N')

ax[3].plot(dn_age['di'],age['di'],label='Decker Island')
ax[3].plot(dn_age['vs'],age['vs'],label='Van Sickle')
ax[3].set_xlim(dn_age['di'][0],dn_age['di'][-1])
ax[3].legend()
ax[3].set_ylabel('Age day$^{-1}$')

ax[4].plot(dn_conc['di'],conc['di'],label='Decker Island')
ax[4].plot(dn_conc['vs'],conc['vs'],label='Van Sickle')
ax[4].set_xlim(dn_age['di'][0],dn_age['di'][-1])
ax[4].legend()
ax[4].set_ylabel('Concentration')

fig.autofmt_xdate()
plt.show()
pdb.set_trace()
