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
from copy import copy

from six import print_
from six.moves import configparser
import pandas as pd

try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import TerminalFormatter
    PYGMENTS = True
except:
    PYGMENTS = False

import zipline
from zipline.errors import NoSourceError

DEFAULTS = {
    'start': '2012-01-01',
    'end': '2012-12-31',
    'data_frequency': 'daily',
    'capital_base': '10e6',
    'source': 'yahoo',
    'symbols': 'AAPL'
}


def parse_args(argv, ipython_mode=False):
    """Parse list of arguments.

    If a config file is provided (via -c), it will read in the
    supplied options and overwrite any global defaults.

    All other directly supplied arguments will overwrite the config
    file settings.

    Arguments:
        * argv : list of strings
            List of arguments, e.g. ['-c', 'my.conf']
        * ipython_mode : bool <default=True>
            Whether to parse IPython specific arguments
            like --local_namespace

    Notes:
    Default settings can be found in zipline.utils.cli.DEFAULTS.

    """
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
        config = configparser.SafeConfigParser()
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
    parser.add_argument('--source', '-d')
    parser.add_argument('--source_time_column', '-t')
    parser.add_argument('--symbols')
    parser.add_argument('--output', '-o')
    parser.add_argument('--metadata_path', '-m')
    parser.add_argument('--metadata_index', '-x')
    if ipython_mode:
        parser.add_argument('--local_namespace', action='store_true')

    args = parser.parse_args(remaining_argv)

    return(vars(args))


def parse_cell_magic(line, cell):
    """Parse IPython magic
    """
    args_list = line.split(' ')
    args = parse_args(args_list, ipython_mode=True)

    local_namespace = args.pop('local_namespace', False)
    # By default, execute inside IPython namespace
    if not local_namespace:
        args['namespace'] = get_ipython().user_ns  # flake8: noqa

    # If we are running inside NB, do not output to file but create a
    # variable instead
    output_var_name = args.pop('output', None)

    perf = run_pipeline(print_algo=False, algo_text=cell, **args)

    if output_var_name is not None:
        get_ipython().user_ns[output_var_name] = perf  # flake8: noqa


def run_pipeline(print_algo=True, **kwargs):
    """Runs a full zipline pipeline given configuration keyword
    arguments.

    1. Load data (start and end dates can be provided a strings as
    well as the source and symobls).

    2. Instantiate algorithm (supply either algo_text or algofile
    kwargs containing initialize() and handle_data() functions). If
    algofile is supplied, will try to look for algofile_analyze.py and
    append it.

    3. Run algorithm (supply capital_base as float).

    4. Return performance dataframe.

    :Arguments:
        * print_algo : bool <default=True>
           Whether to print the algorithm to command line. Will use
           pygments syntax coloring if pygments is found.

    """
    start = pd.Timestamp(kwargs['start'], tz='UTC')
    end = pd.Timestamp(kwargs['end'], tz='UTC')

    symbols = kwargs['symbols'].split(',')
    asset_identifier = kwargs['metadata_index']
    if asset_identifier is None:
        asset_identifier = 'symbol'

    # Pull asset metadata
    asset_metadata = kwargs.get('asset_metadata', None)
    asset_metadata_path = kwargs['metadata_path']
    # Read in a CSV file, if applicable
    if asset_metadata_path is not None:
        if os.path.isfile(asset_metadata_path):
            asset_metadata = pd.read_csv(asset_metadata_path,
                                         index_col=asset_identifier)

    source_arg = kwargs['source']
    source_time_column = kwargs['source_time_column']
    if source_time_column is None:
        source_time_column = 'Date'

    if source_arg is None:
        raise NoSourceError()

    elif source_arg == 'yahoo':
        source = zipline.data.load_bars_from_yahoo(
            stocks=symbols, start=start, end=end)

    elif os.path.isfile(source_arg):
        source = zipline.data.load_prices_from_csv(
            filepath=source_arg,
            identifier_col=source_time_column
        )

    elif os.path.isdir(source_arg):
        source = zipline.data.load_prices_from_csv_folder(
            folderpath=source_arg,
            identifier_col=source_time_column
        )

    else:
        raise NotImplementedError(
            'Source %s not implemented.' % kwargs['source'])

    algo_text = kwargs.get('algo_text', None)
    if algo_text is None:
        # Expect algofile to be set
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
            highlight(algo_text, PythonLexer(), TerminalFormatter(),
                      outfile=sys.stdout)
        else:
            print_(algo_text)

    algo = zipline.TradingAlgorithm(script=algo_text,
                                    namespace=kwargs.get('namespace', {}),
                                    capital_base=float(kwargs['capital_base']),
                                    algo_filename=kwargs.get('algofile'),
                                    asset_metadata=asset_metadata,
                                    identifiers=symbols,
                                    start=start,
                                    end=end)

    perf = algo.run(source, overwrite_sim_params=False)

    output_fname = kwargs.get('output', None)
    if output_fname is not None:
        perf.to_pickle(output_fname)

    return perf
