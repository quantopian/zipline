from functools import partial
import json
from os.path import join as pjoin

import pandas as pd

from zipline.utils.paths import ensure_directory

from . import result_helpers

TimestampMS = partial(pd.Timestamp, unit='ms', tz='UTC')


class AlgorithmResult(object):
    """In-Memory Container for TradingAlgorithm outputs.
    """
    # Top-level metadata.
    _algo_id = None
    _capital_base = None
    _end_date = None
    _start_date = None

    # Daily time series.
    _risk = None
    _daily_performance = None
    _cumulative_performance = None
    _recorded_vars = None

    # Hierarchical frames.
    _transactions = None
    _orders = None
    _positions = None

    _frames = [
        'cumulative_performance',
        'daily_performance',
        'orders',
        'positions',
        'recorded_vars',
        'risk',
        'transactions',
    ]
    _scalars = [
        'algo_id',
        'capital_base',
        'end_date',
        'start_date',
    ]

    def __init__(self,
                 algo_id,
                 start_date,
                 end_date,
                 capital_base,
                 daily_performance,
                 cumulative_performance,
                 risk,
                 orders,
                 positions,
                 transactions,
                 recorded_vars):

        # Scalars
        self._algo_id = algo_id
        self._start_date = start_date
        self._end_date = end_date
        self._capital_base = capital_base

        # Single Row per Day Frames
        self._daily_performance = daily_performance
        self._cumulative_performance = cumulative_performance
        self._risk = risk
        self._recorded_vars = recorded_vars

        # Multiple Row per Day Frames
        self._orders = orders
        self._positions = positions
        self._transactions = transactions

    @classmethod
    def init_processor(cls):
        return result_helpers.default_result_stream_processor()

    @classmethod
    def from_stream(cls, result_iterator, progress_bar, algo_id):
        """
        Create an AlgorithmResult from a stream of algorithm result packets.
        """
        if progress_bar is None:
            progress_bar = result_helpers.NoProgressBar()

        processor = cls.init_processor()
        progress_bar.start()

        for msg in result_iterator:
            processor.handle_message(msg, progress_bar)

        finalized_results = processor.finalize()
        progress_bar.finish()

        return cls(algo_id=algo_id, **finalized_results)

    @property
    def algo_id(self):
        """
        Return the algorithm's ID
        """
        return self._algo_id

    @property
    def benchmark_security(self):
        """
        Return the security used as a benchmark for this algorithm.
        """
        return self._benchmark_security

    @property
    def capital_base(self):
        """
        Return the capital base for this algorithm.
        """
        return self._capital_base

    @property
    def launch_date(self):
        """
        The real-world date on which this algorithm launched.
        """
        return self._launch_date

    @property
    def start_date(self):
        """
        Return the start date for this algorithm.
        """
        return self._start_date

    @property
    def end_date(self):
        """
        Return the end date for this algorithm.
        """
        return self._end_date

    @property
    def daily_performance(self):
        """
        Return a DataFrame with a daily DatetimeIndex containing cumulative
        performance metrics for the algorithm

        Each row of the returned frame contains the following columns:

        * capital_used
        * starting_cash
        * ending_cash
        * starting_position_value
        * ending_position_value
        * pnl
        * portfolio_value
        * returns
        """
        return self._daily_performance

    @property
    def cumulative_performance(self):
        """
        Return a DataFrame with a daily DatetimeIndex containing cumulative
        performance metrics for the algorithm.

        Each row of the returned frame contains the following columns:

        * capital_used
        * starting_cash
        * ending_cash
        * starting_position_value
        * ending_position_value
        * pnl
        * portfolio_value
        * returns
        """
        return self._cumulative_performance

    @property
    def positions(self):
        """
        Return a datetime-indexed DataFrame representing a point-in-time
        record of positions held by the algorithm during the backtest.

        Each row of the returned frame contains the following columns:

        * amount
        * last_sale_price
        * cost_basis
        * sid
        """
        return self._positions

    @property
    def risk(self):
        """
        Return a DataFrame with a daily DatetimeIndex representing rolling risk
        metrics for the algorithm.

        Each row of the returned frame contains the following columns:

        * volatility
        * period_return
        * alpha
        * benchmark_period_return
        * benchmark_volatility
        * beta
        * excess_return
        * max_drawdown
        * period_label
        * sharpe
        * sortino
        * trading_days
        * treasury_period_return
        """
        return self._risk

    @property
    def orders(self):
        """
        Return a DataFrame representing a record of all orders made by the
        algorithm.

        Each row of the returned frame contains the following columns:

        * amount
        * commission
        * created_date
        * last_updated
        * filled
        * id
        * sid
        * status
        * limit
        * limit_reached
        * stop
        * stop_reached
        """
        return self._orders

    @property
    def transactions(self):
        """
        Return a DataFrame representing a record of transactions that occurred
        during the life of the algorithm. The returned frame is indexed by the
        date at which the transaction occurred.

        Each row of the returned frame contains the following columns:

        * amount
        * commission
        * date
        * order_id
        * price
        * sid
        """
        return self._transactions

    @property
    def recorded_vars(self):
        """
        Return a DataFrame containing the recorded variables for the algorithm.
        """
        return self._recorded_vars

    @property
    def attrs(self):
        """
        Return a list of public attributes on this object.
        """
        return self.frames + self.scalars

    @property
    def scalars(self):
        """
        Return a list of scalar attributes of on this object.
        """
        return self._scalars

    @property
    def frames(self):
        """
        Return a list of DataFrame attributes of on this object.
        """
        return self._frames

    def save(self, directory):
        """Save an AlgorithmResult to a directory.
        """
        ensure_directory(directory)
        with open(pjoin(directory, 'metadata.json'), 'w') as f:
            metadata = {
                'algo_id': self.algo_id,
                'capital_base': self.capital_base,
                'start_date': str(self.start_date),
                'end_date': str(self.end_date),
            }
            json.dump(metadata, f)

        write_frame = result_helpers.write_roundtrippable_csv
        for name in self._frames:
            path = pjoin(directory, name + '.csv')
            frame = getattr(self, name)
            write_frame(path, frame)

    @classmethod
    def load(cls, directory):
        """Load an AlgorithmResult from a directory.
        """
        with open(pjoin(directory, 'metadata.json'), 'r') as f:
            kwargs = json.load(f)

        for name in cls._frames:
            path = pjoin(directory, name + '.csv')
            read_frame = result_helpers.read_roundtrippable_csv
            kwargs[name] = read_frame(path)

        return cls(**kwargs)
