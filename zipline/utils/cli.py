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

import sys
import os
import argparse
import ConfigParser
from copy import copy

import datetime
import pandas as pd

try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import TerminalFormatter
    from pygments.styles import STYLE_MAP
    PYGMENTS = True
except:
    PYGMENTS = False

import zipline

DEFAULTS = {
    'start': '2012-01-01',
    'end': '2012-12-31',
    'data_frequency': 'daily',
    'capital_base': '10e6',
    'source': 'yahoo',
    'symbols': 'AAPL'
}

def parse_args(argv):
    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify config file",
                             metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args(argv)

    defaults = copy(DEFAULTS)

    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults.update(dict(config.items("Defaults")))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        description="Zipline version %s." % zipline.__version__,
        parents=[conf_parser]
    )

    parser.set_defaults(**defaults)

    parser.add_argument('--algofile', '-f')
    parser.add_argument('--data-frequency',
                        choices=('minute', 'daily'))
    parser.add_argument('--start', '-s')
    parser.add_argument('--end', '-e')
    parser.add_argument('--capital_base')
    parser.add_argument('--source', choices=('yahoo',))
    parser.add_argument('--symbols')
    args = parser.parse_args(remaining_argv)

    return(vars(args))

def run_algo(print_algo=True, **kwargs):
    start = pd.Timestamp(kwargs['start'], tz='UTC')
    end = pd.Timestamp(kwargs['end'], tz='UTC')

    symbols = kwargs['symbols'].split(',')

    if kwargs['source'] == 'yahoo':
        source = zipline.data.load_bars_from_yahoo(stocks=symbols, start=start, end=end)
    else:
        raise NotImplementedError('Source %s not implemented.' % kwargs['source'])

    algo_fname = kwargs['algofile']
    with open(algo_fname, 'r') as fd:
        algo_text = fd.read()

    analyze_fname = os.path.splitext(algo_fname)[0] + '_analyze.py'
    if os.path.exists(analyze_fname):
        with open(analyze_fname, 'r') as fd:
            # Simply append
            algo_text += fd.read()

    if print_algo:
        if PYGMENTS:
            highlight(algo_text, PythonLexer(), TerminalFormatter(), outfile=sys.stdout)
        else:
            print algo_text

    algo = zipline.TradingAlgorithm(script=algo_text,
                                    capital_base=float(kwargs['capital_base']))

    perf = algo.run(source)

    return perf
