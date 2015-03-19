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
        int sid, i, num_cols
        str col

    num_cols = len(columns)
    for i in range(len(sids)):
        sid = sids[i]
        row = vals[i]
        siddata = bardata[sid].__dict__

        for k in range(num_cols):
            col = columns[k]
            siddata[col] = row[k]
        siddata['dt'] = dt


def update_last_sales(object positions, object[:] columns, int64_t[:] sids,
                      float64_t[:, :] vals, object dt, object position_last_sales):

    cdef:
        object pos
        int sid, price_idx = -1
        float64_t price

    for i in range(len(columns)):
        if columns[i] == 'price':
            price_idx = i
            break

    if price_idx == -1:
        raise Exception("WideTradeEvent must have a price column")

    for i in range(len(sids)):
        sid = sids[i]
        if sid not in positions:
            continue

        price = vals[i][price_idx] # 3 is close
        if price == price:
            # note that positions is a defaultdict, bleh
            pos = positions[sid]
            pos.last_sale_date = dt
            pos.last_sale_price = price
            position_last_sales[sid] = price
