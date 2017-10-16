from contextlib import contextmanager
import re
import json

from six import iteritems, viewkeys
import toolz as tz

import numpy as np
import pandas as pd

from zipline.utils.numpy_utils import (
    float64_dtype,
    int64_dtype,
    object_dtype,
    bool_dtype,
)


class ResultStreamProcessor(object):
    """
    Helper for converting a Zipline result stream into an AlgorithmResult.
    """
    def __init__(self, extractors):
        self._extractors = extractors
        self._last_dt = None

    def handle_message(self, message, progress_bar):
        if 'daily_perf' not in message:
            # Only process daily perf packets.
            return

        dt = self._last_dt = message['daily_perf']['period_close'].normalize()
        for ext in self._extractors:
            ext.update(dt, message)

    def finalize(self):
        merged = tz.merge(e.finalize() for e in self._extractors)
        merged['end_date'] = self._last_dt
        return merged


def default_result_stream_processor():
    return ResultStreamProcessor([
        extract_daily_perf(),
        extract_cumulative_perf(),
        extract_risk(),
        extract_orders(),
        extract_positions(),
        extract_transactions(),
        extract_recorded_vars(),
        extract_top_level_metadata(),
    ])


class SchematizedPayloadExtractor(object):

    def __init__(self, name, path, dtypes, renames):
        self._name = name
        self._path = path
        assert viewkeys(renames) <= viewkeys(dtypes), (
            "Renames must be a subset of dtypes:\n renames=%s, dtypes=%s"
            % (renames, dtypes)
        )
        self._renames = renames
        self._dtypes = dtypes
        self._dts = []
        self._columns = {k: [] for k in self._dtypes}

    def update(self, dt, top_level_message):
        columns = self._columns
        dts = self._dts

        records = tz.get_in(self._path, top_level_message, no_default=True)
        if not isinstance(records, list):
            records = (records,)

        for message in records:
            dts.append(dt)
            for key in columns:
                columns[key].append(message[key])

    def finalize(self):
        index = pd.to_datetime(self._dts, utc=True)
        dtypes = self._dtypes
        arrays = {
            # Apply renames here.
            self._renames.get(k, k): _to_column(data, dtypes[k])
            for k, data in iteritems(self._columns)
        }
        result = pd.DataFrame(arrays, index=index)
        return {self._name: result}


def _to_column(data, dtype):
    arr = np.array(data, dtype=dtype)
    if dtype.kind == 'M':
        arr = pd.DatetimeIndex(arr, tz='UTC')
    return arr


def extract_daily_perf():
    path = ['daily_perf']
    dtypes = {
        # These fields should be kept in sync with extract_cumulative_perf.
        'capital_used':    float64_dtype,
        'ending_cash':     float64_dtype,
        'ending_value':    float64_dtype,
        'pnl':             float64_dtype,
        'portfolio_value': float64_dtype,
        'returns':         float64_dtype,
        # These fields are specific to daily perf.
        'starting_value':  float64_dtype,
        'starting_cash':   float64_dtype,
        'period_open':     np.dtype('datetime64[ms]'),
        'period_close':    np.dtype('datetime64[ms]'),
    }
    renames = {
        'capital_used': 'cash_flow',
        'ending_value': 'ending_position_value',
        'portfolio_value': 'ending_portfolio_value',
        'starting_value': 'starting_position_value',
    }
    return SchematizedPayloadExtractor(
        'daily_performance', path, dtypes, renames
    )


def extract_cumulative_perf():
    path = ['cumulative_perf']
    dtypes = {
        # These fields should be kept in sync with extract_daily_perf.
        'capital_used':    float64_dtype,
        'ending_cash':     float64_dtype,
        'ending_value':    float64_dtype,
        'pnl':             float64_dtype,
        'portfolio_value': float64_dtype,
        'returns':         float64_dtype,
    }
    renames = {
        'capital_used': 'cash_flow',
        'ending_value': 'ending_position_value',
        'portfolio_value': 'ending_portfolio_value',
    }
    return SchematizedPayloadExtractor(
        'cumulative_performance', path, dtypes, renames
    )


def extract_risk():
    path = ['cumulative_risk_metrics']
    dtypes = {
        'algo_volatility':         float64_dtype,
        'algorithm_period_return': float64_dtype,
        'alpha':                   float64_dtype,
        'benchmark_period_return': float64_dtype,
        'benchmark_volatility':    float64_dtype,
        'beta':                    float64_dtype,
        'excess_return':           float64_dtype,
        'max_drawdown':            float64_dtype,
        'period_label':            np.dtype('datetime64[D]'),
        'sharpe':                  float64_dtype,
        'sortino':                 float64_dtype,
        'treasury_period_return':  float64_dtype,
    }
    renames = {
        'algo_volatility': 'volatility',
        'algorithm_period_return': 'period_return',
    }
    return SchematizedPayloadExtractor('risk', path, dtypes, renames)


def extract_orders():
    path = ['daily_perf', 'orders']
    dtypes = {
        'amount': int64_dtype,
        'commission': float64_dtype,
        'created': np.dtype('datetime64[ms]'),
        'dt': np.dtype('datetime64[ms]'),
        'filled': int64_dtype,
        'id': object_dtype,
        'sid': int64_dtype,
        'status': int64_dtype,
        'limit': float64_dtype,
        'limit_reached': bool_dtype,
        'stop': float64_dtype,
        'stop_reached': bool_dtype,
    }
    renames = {
        'dt': 'last_modified',
    }
    return SchematizedPayloadExtractor('orders', path, dtypes, renames)


def extract_positions():
    path = ['daily_perf', 'positions']
    dtypes = {
        'amount': int64_dtype,
        'last_sale_price': float64_dtype,
        'cost_basis': float64_dtype,
        'sid': int64_dtype,
    }
    renames = {}
    return SchematizedPayloadExtractor('positions', path, dtypes, renames)


def extract_transactions():
    path = ['daily_perf', 'transactions']
    dtypes = {
        'amount': int64_dtype,
        'commission': float64_dtype,
        'dt': np.dtype('datetime64[ms]'),
        'order_id': object_dtype,
        'price': float64_dtype,
        'sid': int64_dtype,
    }
    renames = {'dt': 'date'}
    return SchematizedPayloadExtractor('transactions', path, dtypes, renames)


class RecordedVariablePayloadExtractor(object):
    """
    Extractor for float-dtype records that may have different sets of keys
    across records.
    """
    def __init__(self, name, path):
        self._name = name
        self._path = path
        self._records = []
        self._dts = []
        self._all_keys = set()

    def update(self, dt, top_level_message):
        self._dts.append(dt)
        message = tz.get_in(self._path, top_level_message, no_default=True)
        self._all_keys.update(message)
        self._records.append(message)

    def finalize(self):
        return {
            self._name: pd.DataFrame(
                index=self._dts,
                columns=self._all_keys,
                data=self._records,
                dtype=float64_dtype,
            ),
        }


def extract_recorded_vars():
    return RecordedVariablePayloadExtractor(
        name='recorded_vars', path=['daily_perf', 'recorded_vars'],
    )


class TopLevelMetaDataExtractor(object):
    def __init__(self):
        self._extracted = False
        self._start_date = None
        self._capital_base = None

    def update(self, dt, top_level_message):
        if not self._extracted:
            self._start_date = top_level_message['period_start']
            self._capital_base = top_level_message['capital_base']

    def finalize(self):
        return {
            'start_date': self._start_date,
            'capital_base': self._capital_base,
        }


def extract_top_level_metadata():
    return TopLevelMetaDataExtractor()


class ResultsNotAvailable(Exception):

    def __init__(self, algo_id):
        self.algo_id = algo_id

    def __str__(self):
        return ("Could not find results for algorithm with id '{}'. "
                "Please make sure it has run for at least a full day and has "
                "not experienced any errors.").format(self.algo_id)


class NoProgressBar(object):
    """Dummy standin for result progress bar.
    """

    def start(self):
        pass

    def update(self, arg):
        pass

    def finish(self):
        pass


ROUNDTRIP_HEADER_TEMPLATE = "# Roundtrip DType: {}"
ROUNDTRIP_HEADER_REGEX = re.compile(r"(# Roundtrip DType: )(\{.*\})")


def frame_dtypes_to_json(dtypes):
    strcols = dtypes.index.map(lambda x: isinstance(x, str))
    if not strcols.all():
        raise TypeError(
            "Can't serialize frame dtypes for non-string columns.\n"
            "Non strings: {}".format(dtypes.index[~strcols]),
        )

    return json.dumps({
        str(col): dtype.descr[0][1]
        for col, dtype in dtypes.iteritems()
    })


def format_roundtrip_dtype(dtypes):
    dtype_json = frame_dtypes_to_json(dtypes)
    return ROUNDTRIP_HEADER_TEMPLATE.format(dtype_json)


def parse_roundtrip_dtype(header_line):
    match = ROUNDTRIP_HEADER_REGEX.match(header_line)
    if match is None:
        raise ValueError(
            "Failed to parse dtype from header line: {!r}".format(header_line)
        )
    json_str = match.group(2)
    return recarray_dtype_from_json(json_str)


def recarray_dtype_from_json(json_str):
    d = json.loads(json_str)
    return np.dtype(list(d.items()))


@contextmanager
def maybe_open_file(path_or_file, mode):
    if hasattr(path_or_file, 'readline'):
        # Already a file-like object.
        yield path_or_file
    else:
        with open(path_or_file, mode) as f:
            yield f


def write_roundtrippable_csv(path_or_file, frame):
    header = format_roundtrip_dtype(frame.dtypes)
    with maybe_open_file(path_or_file, 'w') as definitely_a_file:
        definitely_a_file.write(header + '\n')
        frame.to_csv(definitely_a_file)


def read_roundtrippable_csv(path_or_file):
    with maybe_open_file(path_or_file, 'r') as definitely_a_file:
        first_line = definitely_a_file.readline().strip()
        dtype = parse_roundtrip_dtype(first_line)
        return pd.DataFrame.from_csv(definitely_a_file, dtype=dtype)
