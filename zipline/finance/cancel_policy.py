#
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
import abc

from abc import abstractmethod
from six import with_metaclass, iteritems

from zipline.gens.sim_engine import DAY_END
from zipline.utils.serialization_utils import (
    VERSION_LABEL
)


class CancelPolicy(with_metaclass(abc.ABCMeta)):

    @abstractmethod
    def should_cancel(self, event):
        pass

    def __getstate__(self):
        state_dict = {k: v for k, v in iteritems(self.__dict__)}

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):
        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("%s saved state is too old." %
                                self.__class__.__name__)

        self.__dict__.update(state)


class EODCancel(CancelPolicy):
    """
    This policy cancels open orders at the end of the day.  For now, Zipline
    will only apply this policy to minutely simulations.
    """
    def __init__(self, warn_on_cancel=True):
        self.warn_on_cancel = warn_on_cancel

    def should_cancel(self, event):
        return event == DAY_END


class NeverCancel(CancelPolicy):
    """
    Orders are never automatically canceled.
    """
    def __init__(self):
        self.warn_on_cancel = False

    def should_cancel(self, event):
        return False
