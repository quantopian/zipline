# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from cython cimport boundscheck, wraparound
from numpy import finfo, float64, nan, isnan
from numpy cimport intp_t, float64_t, uint32_t

@boundscheck(False)
@wraparound(False)
cpdef void _minute_to_session_open(intp_t[:] close_locs,
                                   float64_t[:] data,
                                   float64_t[:] out):
    cdef intp_t i, close_loc, loc = 0
    cdef float64_t val
    for i, close_loc in enumerate(close_locs):
        val = data[loc]
        while isnan(val) and loc <= close_loc:
            loc += 1
            val = data[loc]
        out[i] = val
        loc = close_loc + 1


@boundscheck(False)
@wraparound(False)
cpdef void _minute_to_session_high(intp_t[:] close_locs,
                                   float64_t[:] data,
                                   float64_t[:] out):
    cdef intp_t i, close_loc, loc = 0
    cdef float64_t val
    for i, close_loc in enumerate(close_locs):
        val = -1
        while loc <= close_loc:
            val = max(val, data[loc])
            loc += 1
        if val == -1:
            val = nan
        out[i] = val
        loc = close_loc + 1
        

@boundscheck(False)
@wraparound(False)
cpdef void _minute_to_session_low(intp_t[:] close_locs,
                                   float64_t[:] data,
                                   float64_t[:] out):
    cdef intp_t i, close_loc, loc = 0
    cdef float64_t val
    cdef float64_t max_float = finfo(float64).max
    for i, close_loc in enumerate(close_locs):
        val = max_float
        while loc <= close_loc:
            val = min(val, data[loc])
            loc += 1
        if val == max_float:
            val = nan
        out[i] = val
        loc = close_loc + 1


@boundscheck(False)
@wraparound(False)
cpdef void _minute_to_session_close(intp_t[:] close_locs,
                                    float64_t[:] data,
                                    float64_t[:] out):
    cdef intp_t i, close_loc, loc = 0
    cdef float64_t val
    loc = len(data) - 1
    num_out = len(close_locs)
    for j in range(num_out, 0, -1):
        i = j - 1
        if i > 0:
            close_loc = close_locs[i - 1]
        else:
            close_loc = -1
        val = data[loc]
        while isnan(val) and loc > close_loc:
            loc -= 1
            val = data[loc]
        out[i] = val
        loc = close_loc


@boundscheck(False)
@wraparound(False)
cpdef void _minute_to_session_volume(intp_t[:] close_locs,
                                     uint32_t[:] data,
                                     uint32_t[:] out):
    cdef intp_t i, close_loc, loc = 0
    cdef uint32_t val
    loc = 0
    for i, close_loc in enumerate(close_locs):
        val = 0
        while loc <= close_loc:
            val += data[loc]
            loc += 1
        out[i] = val
        loc = close_loc + 1    
