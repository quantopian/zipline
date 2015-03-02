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

# Label for the serialization version field in the state returned by
# __getstate__.
VERSION_LABEL = '_stateversion_'


class SerializeableZiplineObject(object):
    """
    This class implements the basic set and get state methods used for
    serialization. It also serves as a demarkation of which objects we
    serialize.
    """

    def __getstate__(self):
        """
        Many get_state methods need this one line of code.
        This method deduplicates the code calls.
        """
        state_dict = \
            {k: v for k, v in self.__dict__.iteritems()
                if not k.startswith('_')}
        return state_dict

    def __setstate__(self, state):
        """
        Many objects require only this code.
        """
        self.__dict__.update(state)
