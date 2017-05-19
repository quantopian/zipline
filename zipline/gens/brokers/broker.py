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

from abc import ABCMeta, abstractmethod, abstractproperty


class Broker(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def subscribe_to_market_data(self, symbol):
        pass

    @abstractproperty
    def positions(self):
        pass

    @abstractproperty
    def portfolio(self):
        pass

    @abstractproperty
    def account(self):
        pass

    @abstractproperty
    def time_skew(self):
        pass

    @abstractmethod
    def order(self, asset, amount, limit_price, stop_price, style):
        pass

    @abstractmethod
    def get_open_orders(self, asset):
        pass

    @abstractmethod
    def get_order(self, order_id):
        pass

    @abstractmethod
    def cancel_order(self, order_param):
        pass

    @abstractmethod
    def get_spot_value(self, assets, field, dt, data_frequency):
        pass
