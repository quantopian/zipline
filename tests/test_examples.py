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

from zipline.utils import parse_args, run_pipeline

# Otherwise the next line sometimes complains about being run too late.
_multiprocess_can_split_ = False

matplotlib.use('Agg')


def example_dir():
    import zipline
    d = os.path.dirname(zipline.__file__)
    return os.path.join(os.path.abspath(d), 'examples')


class ExamplesTests(TestCase):
    # Test algorithms as if they are executed directly from the command line.
    @parameterized.expand(((os.path.basename(f).replace('.', '_'), f) for f in
                           glob.glob(os.path.join(example_dir(), '*.py'))))
    def test_example(self, name, example):
        imp.load_source('__main__', os.path.basename(example), open(example))

    # Test algorithm as if scripts/run_algo.py is being used.
    def test_example_run_pipline(self):
        example = os.path.join(example_dir(), 'buyapple.py')
        confs = ['-f', example, '--start', '2011-1-1', '--end', '2012-1-1']
        parsed_args = parse_args(confs)
        run_pipeline(**parsed_args)
