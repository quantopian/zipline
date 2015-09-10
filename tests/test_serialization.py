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

from nose_parameterized import parameterized
from unittest import TestCase
from zipline.finance.trading import TradingEnvironment

from .serialization_cases import (
    object_serialization_cases,
    assert_dict_equal
)

from six import iteritems


def gather_bad_dicts(state):
    bad = []
    for k, v in iteritems(state):
        if not isinstance(v, dict):
            continue
        if type(v) != dict:
            bad.append((k, v))
        bad.extend(gather_bad_dicts(v))
    return bad


class SerializationTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()

    @classmethod
    def tearDownClass(cls):
        del cls.env

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

        state = obj.__getstate__()

        bad_dicts = gather_bad_dicts(state)
        bad_template = "type({0}) == {1}".format
        bad_msgs = [bad_template(k, type(v)) for k, v in bad_dicts]
        msg = "Only support bare dicts. " + ', '.join(bad_msgs)
        self.assertEqual(len(bad_dicts), 0, msg)
        # no state should have a dict subclass. Only regular PyDict

        if hasattr(obj, '__getinitargs__'):
            initargs = obj.__getinitargs__()
        else:
            initargs = None
        if hasattr(obj, '__getnewargs__'):
            newargs = obj.__getnewargs__()
        else:
            newargs = None

        if newargs is not None:
            obj2 = cls.__new__(cls, *newargs)
        else:
            obj2 = cls.__new__(cls)
        if initargs is not None:
            obj2.__init__(*initargs)
        obj2.__setstate__(state)

        for k, v in di_vars.items():
            setattr(obj2, k, v)

        if comparison_method == 'repr':
            self.assertEqual(obj.__repr__(), obj2.__repr__())
        elif comparison_method == 'to_dict':
            assert_dict_equal(obj.to_dict(), obj2.to_dict())
        else:
            assert_dict_equal(obj.__dict__, obj2.__dict__)
