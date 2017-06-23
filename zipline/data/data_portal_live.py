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

import pandas as pd

from zipline.data.data_portal import DataPortal

from logbook import Logger

log = Logger('DataPortalLive')


class DataPortalLive(DataPortal):
    def __init__(self, broker, *args, **kwargs):
        self.broker = broker
        super(DataPortalLive, self).__init__(*args, **kwargs)

    def get_history_window(self,
                           assets,
                           end_dt,
                           bar_count,
                           frequency,
                           field,
                           data_frequency,
                           ffill=True):
        history_window = super(self.__class__, self).get_history_window(
            assets,
            end_dt,
            bar_count,
            frequency,
            field,
            data_frequency,
            ffill)

        # The returned dataframe contains today's value as a NaN because
        # end_dt points to the current wall clock. We drop today's
        # value to be in sync with the simulation's behavior.
        today = pd.to_datetime('now').date()
        return history_window[history_window.index.date != today]

    def get_spot_value(self, assets, field, dt, data_frequency):
        return self.broker.get_spot_value(assets, field, dt, data_frequency)

    def get_adjusted_value(self, asset, field, dt,
                           perspective_dt,
                           data_frequency,
                           spot_value=None):
        raise NotImplementedError("get_adjusted_value is not implemented yet!")
