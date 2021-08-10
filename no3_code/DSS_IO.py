'''
Created on Sep 21, 2016

@author: steve
@organization: Resource Management Associates
@contact: steve@rmanet.com
@note: 
'''
__updated__ = '05-30-2017 09:18'

import glob
import os
import subprocess
import pdb

import datetime as dt
import numpy as np


bad_data_val = -999999.

class DSS_IO(object):
    '''
    Class to handle reading and writing data to DSS files
    '''

    def __init__(self, dss_file_name, **kwargs):

        self.__dict__.update(kwargs)

        self.dss_file_name = dss_file_name

        self.input_file_name = 'tmpDSS_Input.txt'
        self.data_file_name = 'tmpDSS_Output.txt'
        self.junk_file_name = 'tmpDSS_Junk.txt'

    def read_DSS_record(self, record_name, tstart, tend):

        f = open(self.input_file_name, 'w')
        f.write(self.dss_file_name + '\n')
        f.write(record_name + '\n')
        fmt = '%Y/%m/%d %H:%M'
        f.write(tstart.strftime(fmt) + '\n')
        f.write(tend.strftime(fmt) + '\n')
        f.close()

        jnk_stdout = open(self.junk_file_name, 'w')
        subprocess.call('DSSUTL2.exe', stdout=jnk_stdout)
        jnk_stdout.close()

        datesOut = []
        try:
            tmp_matrix = np.loadtxt(self.data_file_name)
        except ValueError:
            return ([], [])

        for j in range(len(tmp_matrix)):
            a = list(map(int, tmp_matrix[j][:-1]))
            datesOut.append(dt.datetime(a[0], a[1], a[2], a[3], a[4], a[5]))
        vOut = tmp_matrix[:, -1]
        idx = vOut == -902.  # DSS missing data flag... could conflict with a real flow value
        vOut[idx] = bad_data_val
        idx = vOut == -901.  # Another DSS missing data flag (a different flag)
        vOut[idx] = bad_data_val
        valsOut = vOut.tolist()
        return (datesOut[:], valsOut[:])

    def write_DSS_record(self, record_name, t, v, units, recType):
        """
        Write a record to a DSS file
        """
        f = open(self.data_file_name, 'w')
        fmt = '%d%b%Y  %H%M'
        for j in range(len(t)):
            if v[j] != bad_data_val:
                vstr = '{0:0.6f}'.format(v[j])
            else:
                vstr = '---'
            line = t[j].strftime(fmt) + '  ' + vstr + '\n'
            f.write(line)
        f.close()

        f = open(self.input_file_name, 'w')
        f.write(self.dss_file_name + '\n')
        outStr = 'EV data=' + record_name + ' UNITS=' + units + ' TYPE=' + recType + '\n'
        f.write(outStr)
        f.write('EF [DATE]  [TIME]  [data]\n')
        f.write('EF.M --- \n')
        outStr = 'IMP ' + self.data_file_name + '\n'
        f.write(outStr)
        f.close()

        cmdLine = "DSSUTL input=" + self.input_file_name + " >" + self.junk_file_name
        jnk_stdout = open(self.junk_file_name, 'w')
        #pdb.set_trace()
        subprocess.call(cmdLine, stdout=jnk_stdout)
        jnk_stdout.close()

    def writeDSSrecord_from_file(self, dataFileName, recordName, units, recType):
        """
        Write a record to a DSS file using a formatted text file as input
        """
        f = open(self.input_file_name, 'w')
        f.write(self.dss_file_name + '\n')
        outStr = 'EV data=' + recordName + ' UNITS=' + units + ' TYPE=' + recType + '\n'
        f.write(outStr)
        f.write('EF [DATE]  [TIME]  [data]\n')
        f.write('EF.M --- \n')

        outStr = 'IMP ' + dataFileName + '\n'
        f.write(outStr)
        f.close()
        cmdLine = "DSSUTL input=" + self.input_file_name + " >" + self.junk_file_name
        jnk_stdout = open(self.junk_file_name, 'w')
        pdb.set_trace()
        subprocess.call(cmdLine, stdout=jnk_stdout)
        jnk_stdout.close()

    def write_multi_DSS_records_from_file(self, dataFileName, recordName_list, units_list, recType):
        """
        Write a record to a DSS file using multiple formatted text files as input
        """
        nrecs = len(recordName_list)
        f = open(self.input_file_name, 'w')
        f.write(self.dss_file_name + '\n')
        for j in range(nrecs):
            outStr = 'EV data' + str(j + 1) + '=' + recordName_list[j] + ' UNITS=' + units_list[j] + ' TYPE=' + recType + '\n'
            f.write(outStr)
        efline = 'EF [DATE]  [TIME]'
        for j in range(nrecs):
            efline = efline + '  [data' + str(j + 1) + ']'
        f.write(efline + '\n')
        f.write('EF.M --- \n')

        outStr = 'IMP ' + dataFileName + '\n'
        f.write(outStr)
        f.close()
        cmdLine = "DSSUTL input=" + self.input_file_name + " >" + self.junk_file_name
        jnk_stdout = open(self.junk_file_name, 'w')
        subprocess.call(cmdLine, stdout=jnk_stdout)
        jnk_stdout.close()

    def cleanup(self):
        print('Cleaning up DSS_IO tmp files')
        file_form = 'tmpDSS*'
        files = glob.glob(file_form)
        for f in files:
            os.remove(f)

def get_dss_interval_for_times(tin):

    tdelt = tin[1] - tin[0]
    if (tdelt == dt.timedelta(hours=3)):
        return '3HOUR'
    elif (tdelt == dt.timedelta(hours=2)):
        return '2HOUR'
    elif (tdelt == dt.timedelta(hours=1)):
        return '1HOUR'
    elif (tdelt == dt.timedelta(minutes=30)):
        return '30MIN'
    elif (tdelt == dt.timedelta(minutes=15)):
        return '15MIN'
    elif (tdelt == dt.timedelta(minutes=5)):
        return '5MIN'
    elif (tdelt == dt.timedelta(minutes=3)):
        return '3MIN'
    elif (tdelt == dt.timedelta(minutes=2)):
        return '2MIN'
    elif (tdelt == dt.timedelta(minutes=1)):
        return '1MIN'
    else:
        return '-9999'

def split_DSS_record_name(record_name):
    sname = record_name.split('/')
    return sname[1:-1]

def create_DSS_record_name(record_parts):
    rec_parts = record_parts[:]
    rec_parts.insert(0, '')
    rec_parts.append('')
    return '/'.join(rec_parts)
