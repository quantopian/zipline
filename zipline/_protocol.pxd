#
# Copyright 2015 Quantopian, Inc.
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

cdef class BarData:
    cdef object data_portal
    cdef object simulation_dt_func
    cdef object data_frequency
    cdef dict _views
    cdef object _get_equity_price_view(BarData, object)
    cdef object _create_sid_view(BarData, object)
    cdef object _is_stale_for_asset(BarData, object, object, object)
    cdef object _can_trade_for_asset(BarData, object, object, object)

cdef class SidView:
    cdef object asset
    cdef object data_portal
    cdef object simulation_dt_func
    cdef object data_frequency
