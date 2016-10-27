from importlib import import_module
import os

from toolz import merge

from zipline import run_algorithm


# These are used by test_examples.py to discover the examples to run.
from zipline.utils.calendars import register_calendar, get_calendar

EXAMPLE_MODULES = {}
for f in os.listdir(os.path.dirname(__file__)):
    if not f.endswith('.py') or f == '__init__.py':
        continue
    modname = f[:-len('.py')]
    mod = import_module('.' + modname, package=__name__)
    EXAMPLE_MODULES[modname] = mod
    globals()[modname] = mod

    # Remove noise from loop variables.
    del f, modname, mod


# Columns that we expect to be able to reliably deterministic
# Doesn't include fields that have UUIDS.
_cols_to_check = [
    'algo_volatility',
    'algorithm_period_return',
    'alpha',
    'benchmark_period_return',
    'benchmark_volatility',
    'beta',
    'capital_used',
    'ending_cash',
    'ending_exposure',
    'ending_value',
    'excess_return',
    'gross_leverage',
    'long_exposure',
    'long_value',
    'longs_count',
    'max_drawdown',
    'max_leverage',
    'net_leverage',
    'period_close',
    'period_label',
    'period_open',
    'pnl',
    'portfolio_value',
    'positions',
    'returns',
    'short_exposure',
    'short_value',
    'shorts_count',
    'sortino',
    'starting_cash',
    'starting_exposure',
    'starting_value',
    'trading_days',
    'treasury_period_return',
]


def run_example(example_name, environ):
    """
    Run an example module from zipline.examples.
    """
    mod = EXAMPLE_MODULES[example_name]

    register_calendar("YAHOO", get_calendar("NYSE"), force=True)

    return run_algorithm(
        initialize=getattr(mod, 'initialize', None),
        handle_data=getattr(mod, 'handle_data', None),
        before_trading_start=getattr(mod, 'before_trading_start', None),
        analyze=getattr(mod, 'analyze', None),
        bundle='test',
        environ=environ,
        # Provide a default capital base, but allow the test to override.
        **merge({'capital_base': 1e7}, mod._test_args())
    )
