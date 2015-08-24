#
# Copyright 2013 Quantopian, Inc.
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

# This code is based on a unittest written by John Salvatier:
# https://github.com/pymc-devs/pymc/blob/pymc3/tests/test_examples.py

# Disable plotting
#

import glob
import imp
import matplotlib
from nose_parameterized import parameterized
import os
from unittest import TestCase

matplotlib.use('Agg')


def example_dir():
    import zipline
    d = os.path.dirname(zipline.__file__)
    return os.path.join(os.path.abspath(d), 'examples')


class ExamplesTests(TestCase):
    @parameterized.expand(((os.path.basename(f).replace('.', '_'), f) for f in
                           glob.glob(os.path.join(example_dir(), '*.py'))))
    def test_example(self, name, example):
        imp.load_source('__main__', os.path.basename(example), open(example))
