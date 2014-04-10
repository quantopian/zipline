import argparse

import os

import sys
sys.path.append('qexec')

import pandas as pd

import cProfile
from line_profiler import LineProfiler
import datetime

from functools import partial

from mem_util import get_memusage_mb


def profile_qexec(algo_text, results_file_name, start_date, end_date,
                  capital_base, granularity, profiler_type=None,
                  names_to_profile=None,
                  live_algo=False,
                  session_start_date=None,
                  inception_date=None,
                  data_delay=None):
    # Import inside profile function, so that modules that take a while
    # to import, e.g. tradingcalendar, don't trigger when there are
    # invalid parameters, which should be a quick fail.
    # TODO: Fix load time of tradingcalendar.
    from algo_profile import run_algo
    from qexec.algo.validation import unittest

    results, ok = unittest(algo_text, granularity)

    if not ok:
        for res in results:
            if not res['passed']:
                print res._data
        sys.exit('Did not pass validation.')

    algo_runner = partial(
        run_algo,
        algo_text,
        start_date,
        end_date,
        capital_base,
        granularity,
        session_start=session_start_date,
        inception_date=inception_date,
        data_delay=data_delay,
        live_algo=live_algo
    )
    if profiler_type == 'cProfile':
        results_dir = os.path.join('results', 'cprofile')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_file = os.path.join(results_dir, results_file_name)
        cProfile.runctx('algo_runner()',
                        globals(),
                        locals(),
                        results_file)
        print ("Wrote results to: {0}".format(results_file))

    elif profiler_type == 'line_profiler':
        results_dir = os.path.join('results', 'line_profiler')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_file = os.path.join(results_dir, results_file_name)
        profiler = LineProfiler()
        for name_to_profile in names_to_profile:
            name_parts = name_to_profile.split('.')
            obj = __import__(name_parts[0])
            for name in name_parts[1:]:
                obj = getattr(obj, name)
            profiler.add_function(obj)
        profiler.runctx('algo_runner()',
                        globals(),
                        locals())
        with open(results_file, 'w') as f:
            profiler.print_stats(stream=f)
        print ("Wrote results to: {0}".format(results_file))

    elif not profiler_type:
        algo_runner()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algofile', '-f', type=argparse.FileType('r'),
                        required=True)
    parser.add_argument('--data-frequency', default='minute',
                        choices=('minute', 'daily'))
    parser.add_argument('--start-date', default='2012-01-01')
    parser.add_argument('--end-date', default='2012-12-31')
    parser.add_argument('--start-epoch', type=int)
    parser.add_argument('--end-epoch', type=int)
    parser.add_argument('--capital-base', default='10e6')
    parser.add_argument('--live-algo', action='store_true', default=False)
    parser.add_argument('--session-start-date', default='2014-01-03')
    parser.add_argument('--inception-date', default='2013-12-03')
    parser.add_argument('--data-delay', type=int, default=15 * 60)
    parser.add_argument('--is-inception', action='store_true', default=False)
    parser.add_argument('--profiler-type', choices=('cProfile',
                                                    'line_profiler')),
    parser.add_argument(
        '--name-to-profile',
        action='append',
        default=[
            # Good proxy for overall performance/main gen
            'zipline.gens.tradesimulation.AlgorithmSimulator.transform',
            # Proxy for network time, will eventually have to change if
            # we change data access style.
            'pymongo.cursor.Cursor.next'
        ])

    return parser


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start_date, tz='UTC')
    end_date = pd.Timestamp(args.end_date, tz='UTC')

    if args.start_epoch:
        start_date = pd.Timestamp(args.start_epoch, tz='UTC')
    if args.end_epoch:
        end_date = pd.Timestamp(args.end_epoch, tz='UTC')

    algo_text = args.algofile.read()
    # Remove the extension
    algo_name_base = os.path.splitext(args.algofile.name)[0]
    algo_name = os.path.basename(algo_name_base)

    results_file_name = \
        'qexec-prof-{algo_name}-{commitish}-{granularity}-{time}'.format(
            algo_name=algo_name.replace('.', '_'),
            commitish='local',
            granularity=args.data_frequency,
            time=str(datetime.datetime.now()).replace(' ', '-').
            replace(':', '-'))

    if args.live_algo:
        session_start_date = pd.Timestamp(args.session_start_date, tz='UTC')
        inception_date = pd.Timestamp(args.inception_date, tz='UTC')

    else:
        session_start_date = None
        inception_date = None

    profile_qexec(
        algo_text,
        results_file_name,
        start_date,
        end_date,
        float(args.capital_base),
        args.data_frequency,
        args.profiler_type,
        args.name_to_profile,
        live_algo=args.live_algo,
        session_start_date=session_start_date,
        inception_date=inception_date,
        data_delay=args.data_delay
    )

    import objgraph
    print 'finished running algo, now doing memory inspection'
    maxrss_in_mb = get_memusage_mb()
    print "Max memory consumption={maxrss_in_mb}MB".format(
        maxrss_in_mb=maxrss_in_mb)
    print objgraph.show_growth(limit=20)
    print objgraph.show_most_common_types(limit=35)
