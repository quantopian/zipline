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

"""
Module for manipulating and verifying the msgpack files.

Intended for testing update logic of msgpack files.
"""
import argparse
import datetime
import functools
import itertools
import os
import sys

import msgpack

import zipline.data.loader as loader
from zipline.utils.date_utils import tuple_to_date


DATA_SOURCES = {
    'benchmark': {
        'filename': '^GSPC_benchmark.msgpack',
        'download_func':
        functools.partial(loader.dump_benchmarks, '^GSPC')
    },
    'treasury': {
        'filename': 'treasury_curves.msgpack',
        'download_func': loader.dump_treasury_curves
    }
}


def last_date(source_name, args):
    data = loader.get_saved_data(DATA_SOURCES[source_name]['filename'])
    date = tuple_to_date(data[-1][0])
    print "Last saved {source_name} date is {date}".format(
        source_name=source_name, date=date)


def drop_before_date(source_name, last_date):
    """
    Loads the msgpack file for the given @source_name and drops values
    up to @last_date.

    Used so that we can test logic that updates the msgpack's to download
    current data if the data isn't current enough.
    """
    filename = DATA_SOURCES[source_name]['filename']
    data = loader.get_saved_data(filename)

    filtered_data = itertools.takewhile(
        lambda x: tuple_to_date(x[0]).date() <= last_date.date(), data)

    with loader.get_datafile(filename, mode='wb') as fp:
        fp.write(msgpack.dumps(list(filtered_data)))


def drop_wrapper(source_name, args):
    if not args.last_date:
        sys.exit("Last date must be supplied for drop.")
    last_date = datetime.datetime.strptime(args.last_date, "%Y-%m-%d")

    drop_before_date(source_name, last_date)


def force_load(source_name, args):
    if not os.path.exists(loader.DATA_PATH):
        os.makedirs(loader.DATA_PATH)

    filename = (DATA_SOURCES[source_name]['filename'])

    filepath = os.path.join(loader.DATA_PATH, filename)

    os.remove(filepath)
    DATA_SOURCES[source_name]['download_func']()


ACTIONS = {
    'last_date': last_date,
    'drop_before_date': drop_wrapper,
    'force_load': force_load
}


def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('--data-source',
                        action='append',
                        choices=DATA_SOURCES.keys(),
                        default=DATA_SOURCES.keys())
    parser.add_argument('--last-date')
    return parser


def main():
    parser = _make_parser()
    args = parser.parse_args()

    for source_name in args.data_source:
        ACTIONS[args.action](source_name, args)

if __name__ == "__main__":
    main()
