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

from zipline.utils.serialization_utils import (
    loads_with_persistent_ids, dumps_with_persistent_ids
)

from nose_parameterized import parameterized
from unittest import TestCase

from .serialization_cases import (
    object_serialization_cases,
    assert_dict_equal,
    cases_env,
)


class PickleSerializationTestCase(TestCase):

    @parameterized.expand(object_serialization_cases())
    def test_object_serialization(self,
                                  _,
                                  cls,
                                  initargs,
                                  di_vars,
                                  comparison_method='dict'):

        obj = cls(*initargs)
        for k, v in di_vars.items():
            setattr(obj, k, v)
        state = dumps_with_persistent_ids(obj)

        obj2 = loads_with_persistent_ids(state, env=cases_env)
        for k, v in di_vars.items():
            setattr(obj2, k, v)

        if comparison_method == 'repr':
            self.assertEqual(obj.__repr__(), obj2.__repr__())
        elif comparison_method == 'to_dict':
            assert_dict_equal(obj.to_dict(), obj2.to_dict())
        else:
            assert_dict_equal(obj.__dict__, obj2.__dict__)
