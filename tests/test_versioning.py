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

import os
import pickle

from nose_parameterized import parameterized
from unittest import TestCase, skip

from zipline.finance.blotter import Order

from .serialization_cases import (
    object_serialization_cases,
    assert_dict_equal
)

base_state_dir = 'tests/resources/saved_state_archive'

BASE_STATE_DIR = os.path.join(
    os.path.dirname(__file__),
    'resources',
    'saved_state_archive')


class VersioningTestCase(TestCase):

    def load_state_from_disk(self, cls):
        state_dir = cls.__module__ + '.' + cls.__name__

        full_dir = BASE_STATE_DIR + '/' + state_dir

        state_files = \
            [f for f in os.listdir(full_dir) if 'State_Version_' in f]

        for f_name in state_files:
            f = open(full_dir + '/' + f_name, 'r')
            yield pickle.load(f)

    # Only test versioning in minutely mode right now
    @parameterized.expand(object_serialization_cases(skip_daily=True))
    @skip
    def test_object_serialization(self,
                                  _,
                                  cls,
                                  initargs,
                                  di_vars,
                                  comparison_method='dict'):

        # Make reference object
        obj = cls(*initargs)
        for k, v in di_vars.items():
            setattr(obj, k, v)

        # Fetch state
        state_versions = self.load_state_from_disk(cls)

        for version in state_versions:

            # For each version inflate a new object and ensure that it
            # matches the original.

            newargs = version['newargs']
            initargs = version['initargs']
            state = version['obj_state']

            if newargs is not None:
                obj2 = cls.__new__(cls, *newargs)
            else:
                obj2 = cls.__new__(cls)
            if initargs is not None:
                obj2.__init__(*initargs)

            obj2.__setstate__(state)
            for k, v in di_vars.items():
                setattr(obj2, k, v)

            # The ObjectId generated on instantiation of Order will
            # not be the same as the one loaded from saved state.
            if cls == Order:
                obj.__dict__['id'] = obj2.__dict__['id']

            if comparison_method == 'repr':
                self.assertEqual(obj.__repr__(), obj2.__repr__())
            elif comparison_method == 'to_dict':
                assert_dict_equal(obj.to_dict(), obj2.to_dict())
            else:
                assert_dict_equal(obj.__dict__, obj2.__dict__)
