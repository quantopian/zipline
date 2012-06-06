from vbench.api import Benchmark
from datetime import datetime

setup = """
from zipline_bench_functions import *
"""

basic_zipline = Benchmark(
        'run_basic_zipline()',
        setup=setup,
        start_date=datetime(2012,5,15),
        name='basic_zipline_test'
)

load_ndict = Benchmark(
        'load_ndict()',
        setup=setup,
        start_date=datetime(2012,5,15),
        name='load_ndict'
)

mass_create_ndict = Benchmark(
        'mass_create_ndict()',
        setup=setup,
        start_date=datetime(2012,5,1),
        name='create_ndict'
)
