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

from datetime import datetime
import pytz

from copy import deepcopy

from zipline.utils.protocol_utils import ndict


def test_ndict():
    nd = ndict({})

    # Properties
    assert len(nd) == 0
    assert nd.keys() == []
    assert nd.values() == []
    assert list(nd.iteritems()) == []

    # Accessors
    nd['x'] = 1
    assert nd.x == 1
    assert nd['x'] == 1
    assert nd.get('y') is None
    assert nd.get('y', 'fizzpop') == 'fizzpop'
    assert 'x' in nd
    assert 'y' not in nd

    assert 'x' in nd
    assert 'y' not in nd

    # Mutability
    nd2 = ndict({'x': 1})
    assert nd2.x == 1
    nd2.x = 2
    assert nd2.x == 2

    # Class isolation
    assert '__init__' not in nd
    assert '__iter__' not in nd
    assert 'x' not in nd.__dict__
    assert nd.get('__init__') is None
    assert 'x' not in set(dir(nd))

    # Comparison
    nd2 = nd.copy()
    assert id(nd2) != id(nd)
    assert nd2 == nd
    nd2['z'] = 3
    assert nd2 != nd

    class ndictlike(object):
        x = 1

    assert {'x': 1} == nd
    assert ndictlike() != nd

    # Deletion
    del nd['x']
    assert 'x' not in nd
    assert nd.get('x') is None

    for n in xrange(1000):
        dt = datetime.utcnow().replace(tzinfo=pytz.utc)
        nd2 = ndict({"dt": dt,
                     "otherdata": "ishere" * 1000,
                     "maybeanint": 3})

    nd2.dt2 = dt


def test_ndict_deepcopy():
    def assert_correctly_copied(orig, copy):
        assert nd == nd_dc, \
            "Deepcopied ndict should have same keys and values."

        nd_dc.z = 3
        assert 'z' not in nd, "'z' also added to original ndict."

        nd_dc.y = 10
        assert nd_dc.y == 10, "value of copied ndict not correctly set."
        assert nd.y != 10, "value also set of original ndict."

    nd = ndict({'x': 1, 'y': 2})
    nd_dc = deepcopy(nd)
    assert_correctly_copied(nd, nd_dc)

    nd = ndict({'x': [1, 2, 3],
                'y': {1: 1}})
    nd_dc = deepcopy(nd)
    assert_correctly_copied(nd, nd_dc)
    nd_dc.x.append(4)
    assert nd_dc.x[-1] == 4, "not correctly appended to copied."
    assert nd.x[-1] != 4, "also copied to original."
