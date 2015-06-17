cimport numpy as np
import numpy as np
import pandas as pd
cimport cython

from contextlib2 import ExitStack
from contextlib import contextmanager

from zipline.utils.api_support import ZiplineAPI

cdef np.int64_t minute_in_nano = 60000000000

cdef class DayEngine:

    cdef np.int64_t[:] market_opens, market_closes

    def __init__(self, market_opens, market_closes):
        self.market_opens = market_opens
        self.market_closes = market_closes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef market_minutes(self, np.intp_t i):
        cdef np.int64_t[:] market_opens, market_closes
        market_opens = self.market_opens
        market_closes = self.market_closes

        return np.arange(market_opens[i], market_closes[i], minute_in_nano)
