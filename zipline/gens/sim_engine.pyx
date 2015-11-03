cimport numpy as np
import numpy as np
import pandas as pd
cimport cython

cdef np.int64_t _nanos_in_minute = 60000000000
NANOS_IN_MINUTE = _nanos_in_minute

cpdef enum:
    DATA_AVAILABLE = 0
    ONCE_A_DAY = 1
    UPDATE_BENCHMARK = 2
    CALC_DAILY_PERFORMANCE = 3
    CALC_MINUTE_PERFORMANCE = 4


cdef class MinuteSimulationClock:

    cdef object trading_days
    cdef object data_portal
    cdef np.int64_t[:] market_opens, market_closes

    def __init__(self,
                 trading_days,
                 market_opens,
                 market_closes,
                 data_portal):
        self.market_opens = market_opens
        self.market_closes = market_closes
        self.trading_days = trading_days
        self.data_portal = data_portal

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef market_minutes(self, np.intp_t i):
        cdef np.int64_t[:] market_opens, market_closes
        market_opens = self.market_opens
        market_closes = self.market_closes

        return np.arange(market_opens[i],
                         market_closes[i] + _nanos_in_minute,
                         _nanos_in_minute)

    def __iter__(self):
        first_trading_day_idx = 0
        data_portal = self.data_portal
        for day_idx, day in enumerate(self.trading_days):
            day_offset = (day_idx + first_trading_day_idx) * 390
            yield day, ONCE_A_DAY

            minutes = pd.DatetimeIndex(self.market_minutes(day_idx), tz='UTC')
            for minute_idx, minute in enumerate(minutes):
                data_portal.cur_data_offset = day_offset + minute_idx
                yield minute, DATA_AVAILABLE

            #yield day, UPDATE_BENCHMARK
            yield day, CALC_DAILY_PERFORMANCE


cdef class DailySimulationClock:

    cdef object trading_days

    def __init__(self, trading_days):
        self.trading_days = trading_days

    def __iter__(self):
        for i, day in enumerate(self.trading_days):
            yield day, ONCE_A_DAY
            yield day, DATA_AVAILABLE
            #yield day, UPDATE_BENCHMARK
            yield day, CALC_DAILY_PERFORMANCE
