#
# Copyright 2012 Quantopian, Inc.
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


class Order(object):

    def __init__(self, initial_values=None):
        if initial_values:
            self.__dict__ = initial_values

    def __getitem__(self, name):
        return self.__dict__[name]

class MarketOrder(Order):

    def __init__(self, initial_values=None):
        if initial_values:
            super.__dict__ = initial_values

    def __getitem__(self, name):
        return super.__dict__[name]

class LimitOrder(Order):

    def __init__(self, initial_values=None):
        if initial_values:
            super.__dict__ = initial_values

    def __getitem__(self, name):
        return super.__dict__[name]

class StopOrder(Order):

    def __init__(self, initial_values=None):
        if initial_values:
            super.__dict__ = initial_values

    def __getitem__(self, name):
        return super.__dict__[name]

class StopLimitOrder(Order):

    def __init__(self, initial_values=None):
        if initial_values:
            super.__dict__ = initial_values

    def __getitem__(self, name):
        return super.__dict__[name]
