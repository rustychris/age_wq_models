import pandas as pd
import os
import copy
import numpy as np
import pyproj
from netCDF4 import Dataset, num2date
from stompy.grid import unstructured_grid

from obsdata_utils import save_current_file, get_var, get_var_dss
from obsdata_utils import get_var_dss_light


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

        uw_df.to_csv('uw_df.csv',index=False)
    else:
        uw_df = pd.read_csv('inputs-20210827/uw_df.csv')
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

## Reading tracer output from DSS 
age_vars = ['age','conc','depth',#'temperature'
            'fav-age','sav-age','marsh-age']
var_label_dict = {'age':'Age [days]',
                  'conc':'Concentration',
                  'depth':'Depth [m]',
                  'fav-age':'FAV age [days]',
                  'sav-age':'SAV age [days]',
                  'marsh-age':'Marsh age [days]',
                  #'temperature':'Temperature [$^\circ$C]'
}
age_stations=['dc','cr','cl','di','vs']
# File also has DWSC, but I think Ed was omitting for a reason
label_dict = {'dc':'Sac ab DCC',
              'cr':'Cache ab Ryer',
              'cl':'Cache Lib',
              'di':'Sac Decker',
              'vs':'Van Sickle'}
def read_tracer_output():
    # read in information needed for NO3 predictions
    # read age first because will use selected time for interpolation etc.
    age_dss_fname = 'inputs-20210827/AgeScalars.dss'
    rec_template = '/UT/$STA$/$VAR$//30MIN/TEMPERATURE_2018_16_FEBSTART_SAC/'
    #rec_template = '/UT/%(STA)s/$VAR$//30MIN/TEMPERATURE_2018_16_FEBSTART_SAC/'
    #rec_template%dss_sta_dict

    age_data = {}
    age_data['dn']={}
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
        # Trim to valid data only, per station.
        for sta in age_stations:
            valid = np.where(np.isfinite(age_data[var][sta]))
            sta_dn= dn_age[sta][valid]
            # Note that we fetch multiple variables, each time getting dn,
            # but store it only once. Verify that dn is the same across variables.
            if sta in age_data['dn']:
                assert np.all( age_data['dn'][sta]==sta_dn)
            else:
                age_data['dn'][sta] = sta_dn
            age_data[var][sta] = age_data[var][sta][valid]
    return age_data

if 0:
    from stompy.plot import plot_utils
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
from shapely import geometry
sac_poly=geometry.Polygon(sac_corridor)
