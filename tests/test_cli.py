#
# Copyright 2014 Quantopian, Inc.
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
from unittest import TestCase
from six import iteritems

from zipline.utils import parse_args
from zipline.utils import cli


class TestParseArgs(TestCase):
    def test_defaults(self):
        args = parse_args([])
        for k, v in iteritems(cli.DEFAULTS):
            self.assertEqual(v, args[k])

    def write_conf_file(self):
        #TODO add new CLI arguments
        conf_str = """
[Defaults]
algofile=test.py
symbols=test_symbols
start=1990-1-1
        """

        with open('test.conf', 'w') as fd:
            fd.write(conf_str)

    def test_conf_file(self):
        self.write_conf_file()
        try:
            args = parse_args(['-c', 'test.conf'])

            self.assertEqual(args['algofile'], 'test.py')
            self.assertEqual(args['symbols'], 'test_symbols')
            self.assertEqual(args['start'], '1990-1-1')
            self.assertEqual(args['end'], cli.DEFAULTS['end'])
        finally:
            os.remove('test.conf')

    def test_overwrite(self):
        self.write_conf_file()

        try:
            args = parse_args(['-c', 'test.conf', '--start', '1992-1-1',
                               '--algofile', 'test2.py'])

            # Overwritten values
            self.assertEqual(args['algofile'], 'test2.py')
            self.assertEqual(args['start'], '1992-1-1')
            # Non-overwritten values
            self.assertEqual(args['symbols'], 'test_symbols')
            # Default values
            self.assertEqual(args['end'], cli.DEFAULTS['end'])
        finally:
            os.remove('test.conf')
