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

"""
Tests for the zipline.assets package
"""

from unittest import TestCase

from zipline.assets._securities import Security


class SecurityTestCase(TestCase):

    def test_security_object(self):
        self.assertEquals({5061: 'foo'}[Security(5061)], 'foo')
        self.assertEquals(Security(5061), 5061)
        self.assertEquals(5061, Security(5061))

        self.assertEquals(Security(5061), Security(5061))
        self.assertEquals(int(Security(5061)), 5061)

        self.assertEquals(str(Security(5061)), 'Security(5061)')

        # TODO: we can't provide this property while subclassing
        # int. Need to subclass object to fix all the following
        # cases.
        # assert Security(5061) != 5061.0

        # with self.assertRaises(TypeError):
        #    assert Security(5061) + Security(5061)

        # with self.assertRaises(TypeError):
        #    print float(Security(5061))

        # with self.assertRaises(TypeError):
        #    print float(Security(5061))
