# Emulate reading record via DSSUTL2, but using
# command line interface to DSSVUE.
import datetime as dt
import subprocess
import datetime
import os
import numpy as np
import pandas as pd
from stompy import memoize

vue_script=r"""
# HEC-DSSVUE script to emulate dssutl2.exe
import java
import datetime
from hec.heclib.dss import HecDss

hecfile=HecDss.open("%(dss_file_name)s",1) # must exist
data=hecfile.get("%(dss_path)s",True) # True to read full time series

# def fmt_digits(
with open("%(csv_out)s",'w') as fp:
    fp.write("time,value\n")
    times = data.getTimes()
    for i,V in enumerate(data.values):
        t = times.element(i)
        # Note that hour is sometimes 24!
        tstr="%%d-%%02d-%%02d %%02d:%%02d"%%( t.year(),t.month(),t.day(),
                                        t.hour(),t.minute() )
        fp.write(tstr+",%%.6f\n"%%V)
    
hecfile.done()
"""

hec_dssvue="/home/rusty/software/HEC-DSSVue-323-Beta-Linux/hec-dssvue.sh"

# Cache extracted data
@memoize.memoize(50)
def worker(dss_file_name,record_name,tstart,tend,script_fn,csv_out='output.csv'):
    args=dict(dss_file_name=dss_file_name,
              dss_path=record_name,
              csv_out=csv_out)

    with open(script_fn,'wt') as fp:
        fp.write(vue_script%args)

    if os.path.exists(args['csv_out']):
        os.unlink(args['csv_out'])

    subprocess.call([hec_dssvue,script_fn])
    df=pd.read_csv(args['csv_out'])
    
    # Come back to fix up dates:
    def parse_date(s):
        Y=int(s[:4])
        m=int(s[5:7])
        d=int(s[8:10])
        H=int(s[11:13])
        M=int(s[14:16])

        t=datetime.datetime(Y,m,d,0,M) + datetime.timedelta(hours=H)
        return t

    df['time']=df['time'].apply(parse_date)
    
    return df
    
class DSS_IO(object):
    '''
    Class to handle reading and writing data to DSS files
    '''
    bad_data_value = -999999.

    script_fn="hec_script.py"
    bad_data=[-901,-902]
    csv_out="output.csv"
    
    def __init__(self, dss_file_name, **kwargs):
        self.__dict__.update(kwargs)
        self.dss_file_name = dss_file_name

    def read_DSS_record(self, record_name, tstart, tend):
        df=worker(self.dss_file_name,record_name,tstart,tend,self.script_fn)
        
        date_sel=(df['time'].values>=np.datetime64(tstart) ) & ( df['time'].values<=np.datetime64(tend))
        df=df[date_sel]

        datesOut=df['time'].dt.to_pydatetime()
        valsOut=df['value'].values
        
        # Fix up bad values:
        for bad_datum in self.bad_data:
            valsOut[ valsOut==bad_datum ] = self.bad_data_value
        
        return (list(datesOut), list(valsOut))

# So it's supposed to return a tuple, a sequence of dates and sequence of values.
# dates are python datetime
# values of -902.0 are replaced with bad_data_val, and the array converted to a list
# 

##

# The info passed to DSS_IO
if __name__=='__main__':
    dss_file_name="AgeScalars.dss"
    record_name='/UT/SAC_AB_DCC/age//30MIN/TEMPERATURE_2018_16_FEBSTART_SAC/'
    tstart=dt.datetime(2018, 1, 1, 0, 0)
    tend=dt.datetime(2019, 12, 31, 0, 0)
    dss_io=DSS_IO(dss_file_name)
    result=dss_io.read_DSS_record(record_name,tstart,tend)
