#cython: cdivision=True

cimport numpy as np
from cpython cimport bool
import numpy as np
from pandas.tslib import Timestamp

from numpy cimport *
from cpython.dict cimport PyDict_Clear, PyDict_DelItem, PyDict_SetItem

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

def update_last_sales(object positions, int64_t[:] sids, float64_t[:, :] vals,
                     object dt, object position_last_sales):

    cdef:
        object pos
        int sid
        float64_t price

    for i in range(len(sids)):
        sid = sids[i]
        price = vals[i][3] # 3 is close
        if price == price:
            # note that positions is a defaultdict, bleh
            pos = positions[sid]
            pos.last_sale_date = dt
            pos.last_sale_price = price
            position_last_sales[sid] = price
