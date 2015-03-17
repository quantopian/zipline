#cython: cdivision=True

cimport numpy as np
from cpython cimport bool
import numpy as np
from pandas.tslib import Timestamp

from numpy cimport *

def update_sid(dict bardata, object[:] columns, int64_t[:] sids,
               float64_t[:, :] vals, object dt):
    cdef:
        float64_t[:] row
        dict siddata
        int sid, i

    for i in range(len(sids)):
        sid = sids[i]
        row = vals[i]
        siddata = bardata[sid].__dict__

        siddata['open'] = row[0]
        siddata['high'] = row[1]
        siddata['low'] = row[2]
        siddata['close'] = row[3]
        siddata['price'] = row[3]
        siddata['volume'] = row[4]
        siddata['dt'] = dt
