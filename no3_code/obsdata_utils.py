
import os.path, sys
#import rmapy
import copy
import numpy as np
from matplotlib.dates import num2date, date2num, DayLocator
import datetime as dt
from pylab import date2num as d2n
from stompy.utils import hour_tide
from scipy.interpolate import interp1d
import pdb
import pandas as pd
# from DSS_IO import (DSS_IO, bad_data_val, get_dss_interval_for_times,
#                     split_DSS_record_name, create_DSS_record_name)
from DSS_IO_vue import DSS_IO

import pytz
import shelve
import shutil
from stompy.utils import fill_invalid


def create_dir(a_path):
    if not os.path.isdir(a_path):
        os.mkdir(a_path)


def save_current_file(destination,
                      other_files=None):
    files_to_copy = []
    curr_file = os.path.realpath(__file__)
    files_to_copy.append(curr_file)
    if other_files is not None:
        files_to_copy += other_files
    for afile in files_to_copy:
        shutil.copy(afile, destination)


def save_fit_results(fit_results, destination):
    df = pd.DataFrame(fit_results, index=[0])
    df.to_csv(os.path.join(destination, "fit_results.csv"))


def get_hydro_var(nchydro, var, nrec=slice(None), cell=slice(None)):
    return nchydro.rg.variables[nc_hydro.make_var_name(var)][cell, ..., nrec]


def get_hydro_var_full_depth_mean(nchydro, var, nrec, cell_idx=slice(None)):
    nc_hydro.cache_wse(nrec)  # maybe re-do cache as individual vars?
    vol = get_hydro_var(nchydro, 'face_water_volume', nrec, cell=cell_idx)
    # nPoly = len(self.cache.kb)
    data = get_hydro_var(nchydro, var, nrec, cell=cell_idx)
    data_mean = np.zeros(len(cell_idx))
    for i in range(len(cell_idx)):  # range(nPoly):
        kbi = nc_hydro.cache.kb[i]
        kti = nc_hydro.cache.kt[i]
        # data_mean[i] = np.sum(data[i,kbi:kti]*vol[i,kbi:kti])/np.sum(vol[i,kbi:kti])
        data_mean[i] = np.sum(data[i, kbi:kti + 1] * vol[i, kbi:kti + 1]) / np.sum(vol[i, kbi:kti + 1])
    return data_mean


def get_var(ts_file, plot_stations, varname=None):
    ts_dframe = pd.read_csv(ts_file, index_col=0)
    ts_rec = ts_dframe.to_records()
    dt_var = {}
    dn_var = {}
    var = {}
    for ps in plot_stations:
        try:
            indices = np.where(ts_rec["Site"] == ps)[0]
        except:
            indices = np.where(ts_rec["SiteName"] == ps)[0]
        try:
            dstrings = ts_rec.Datetime_UTC[indices]
        except AttributeError:
            dstrings = ts_rec.datetime_DLS[indices]

        try:
            try:
                dtimes = [dt.datetime.strptime(ds, "%m/%d/%Y %H:%M") for ds in dstrings]
            except:
                dtimes = [dt.datetime.strptime(ds,"%Y-%m-%d %H:%M:%S") for ds in dstrings]
        except:
            dtimes = [dt.datetime.strptime(ds, "%Y-%m-%dT%H:%M:%SZ") for ds in dstrings]
        dt_var[ps] = np.asarray(dtimes)
        dn_var[ps] = d2n(dt_var[ps])
        var[ps] = ts_rec[varname][indices]
    return dn_var, var


def get_var_dss(dss_fname, stations, dss_records,
                tstart = dt.datetime(2018, 1, 1, ),
                tend = dt.datetime(2019, 12, 31)):
    """
    Parameters:
        stations: List(stations)
        dss_records: dss_recs(default),
            DSS records to read the variable.
    Returns:
        t: Dict([station]: [dateteime objs])
        var: Dict([station]: [data])
    """
    dss_io = DSS_IO(dss_fname)
    t = {}
    var = {}
    for station in stations:
        tt, vv = dss_io.read_DSS_record(dss_records[station], tstart, tend)
        # tt = np.array([d2n(time_pst.astimezone(pytz.timezone("UTC"))) for time_pst in tt])
        tt = d2n(np.array(tt))
        vv = np.array(vv)
        vv[vv < -999.] = np.nan   # Temporary fix for bad data (06May2020)
        t[station], var[station] = tt, vv
    return t, var


def get_var_dss_light(dss_filename, dss_record,
                      tstart=dt.datetime(2018, 1, 1),
                      tend=dt.datetime(2019, 12, 31)):
    """
    Parameters:
        dss_filename: str,
            DSS filename
        dss_record: str,
            DSS record to read the variable.
    Returns:
        t: np.array of datetime objects
        var: fill_invalid applied numpy array of variable
    """
    dss_io_inst = DSS_IO(dss_filename)
    t, var = dss_io_inst.read_DSS_record(dss_record, tstart, tend)
    var = np.asarray(var)
    var[var < 0.] = np.nan
    return np.asarray(t), fill_invalid(var)


def get_datums(dn, eta):
    days = np.unique(dn.astype(np.int32))
    noons = days.astype(np.float64) + 0.5
    llw = np.nan*np.ones_like(noons)
    hhw = np.nan*np.ones_like(noons)
    half_tidal_day = 24.8/(2.*24.)
    tr = np.zeros_like(eta)
    for n, noon in enumerate(noons):
        dn_min = noon - half_tidal_day
        dn_max = noon + half_tidal_day
        indices = np.where(np.logical_and(dn >= dn_min, dn <= dn_max))[0]
        llw[n] = np.min(eta[indices])
        hhw[n] = np.max(eta[indices])
        tr[indices] = hhw[n] - llw[n]  # Added 17Jul2020 by SN
    # overwrite ends because don't have full day of data at ends
    llw[0] = llw[1]
    llw[-1] = llw[-2]
    hhw[0] = hhw[1]
    hhw[-1] = hhw[-2]
    return noons, llw, hhw, tr


def get_filled_eta(ts_file, year=2017):
    ts_dframe = pd.read_csv(ts_file, index_col=0)
    ts_rec = ts_dframe.to_records() 
    # form dictionaries for eta and time for each station
    eta = {}
    dt_eta = {}
    dn_eta = {}
    dn_min = 999999999.

    dn_max = 0.
    fifteen_min = 15.*60./86400.
    for ps in plot_stations:
        indices = np.where(ts_rec["Site"] == ps)[0]
        eta[ps] = ts_rec.Depth[indices]
        dstrings = ts_rec.datetime[indices]
        dtimes = [dt.datetime.strptime(ds,"%Y-%m-%dT%H:%M:%SZ") for ds in dstrings]
        dt_eta_tmp = np.asarray(dtimes)
        dt_year = np.asarray([dty.year for dty in dt_eta_tmp])
        i_year = np.where(dt_year == year)[0]
        # subset_current year of eta and dtimes
        dt_eta[ps] = np.asarray(dtimes)[i_year]
        eta[ps] = eta[ps][i_year]
        dn_eta[ps] = d2n(dt_eta[ps])
        dn_min = min(np.minimum(dn_eta[ps],dn_min))
        dn_max = max(np.maximum(dn_eta[ps],dn_max))
    # overwrite eta with eta interpolated onto same time stamps
    # fill in order of plot stations
    dn_fill = np.arange(dn_min,dn_max,fifteen_min)
    eta_fill = np.nan*np.ones_like(dn_fill)
    for n, ps in enumerate(plot_stations):
        f_interp = interp1d(dn_eta[ps], eta[ps], bounds_error=False)
        eta_tmp = f_interp(dn_fill)
        if n == 0:
            FM_mean = np.nanmean(eta_tmp)
        d_eta = np.nanmean(eta_tmp) - FM_mean
        fillme = np.where(np.isnan(eta_fill))[0]
        eta_fill[fillme] = eta_tmp[fillme] - d_eta
        
    return dn_fill, eta_fill


def get_filled_stage(dn_stg, dat_stg, plt_stats):
    for ps in plt_stats:
        f_interp = interp1d(dn_stg[ps], dat_stg[ps], bounds_error=False)
        stg_tmp = f_interp(dn_stg[ps])
        fillme = np.where(np.isnan(dat_stg[ps]))[0]
        dat_stg[ps][fillme] = stg_tmp[fillme]
    return dn_stg, dat_stg


def medfilt_weighted(data, N, weights=None):
    quantile = 0.5
    if weights is None:
        weights = np.ones(len(data))

    data = np.asarray(data)
    weights = np.asarray(weights)

    result = np.zeros_like(data)
    neg = N // 2
    pos = N - neg
    for i in range(len(result)):
        # say N is 5
        # when i=0, we get [0,3]
        # when i=10, we get [8,13
        slc = slice(max(0, i - neg),
                    min(i + pos, len(data)))
        subdata = data[slc]
        # Sort the data
        ind_sorted = np.argsort(subdata)
        sorted_data = subdata[ind_sorted]
        sorted_weights = weights[slc][ind_sorted]
        # Compute the auxiliary arrays
        Sn = np.cumsum(sorted_weights)
        # center those and normalize
        Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]
        result[i] = np.interp(quantile, Pn, sorted_data)
    return result

