#cython: cdivision=True

cimport numpy as np
from cpython cimport bool
import numpy as np
from pandas.tslib import Timestamp

from numpy cimport *
from cpython.dict cimport PyDict_Clear, PyDict_DelItem, PyDict_SetItem


def update_sid(dict bardata, object[:] columns, int64_t[:] sids,
               float64_t[:, :] values, object dt):
    cdef:
        float64_t[:] row
        dict siddata
        int sid, i, num_cols
        str col

    num_cols = len(columns)
    for i in range(len(sids)):
        sid = sids[i]

        # -1 is a sentinel for sid removal
        if sid == -1:
            continue

        row = values[i]
        siddata = bardata[sid].__dict__

        for k in range(num_cols):
            col = columns[k]
            siddata[col] = row[k]
        siddata['dt'] = dt


def update_last_sales(object positions, object[:] columns, int64_t[:] sids,
                      float64_t[:, :] values, object dt, object position_last_sales,
                      int price_loc):

    cdef:
        object pos
        int sid
        float64_t price

    for i in range(len(sids)):
        sid = sids[i]

        # -1 is a sentinel for sid removal
        if sid == -1:
            continue

        if sid not in positions:
            continue

        price = values[i][price_loc] # 3 is close
        if price == price:
            # note that positions is a defaultdict, bleh
            pos = positions[sid]
            pos.last_sale_date = dt
            pos.last_sale_price = price
            position_last_sales[sid] = price

def unset_sids(int64_t[:] sids, set missing):
    """
    Utility function to replace missing sids with a missing sid 
    sentinel
    """
    cdef:
        int sid

    for i in range(len(sids)):
        sid = sids[i]
        # unset with sentinel
        if sid in missing:
            sids[i] = -1
