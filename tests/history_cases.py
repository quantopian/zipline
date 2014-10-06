"""
Test case definitions for history tests.
"""

import pandas as pd
import numpy as np

from zipline.finance.trading import TradingEnvironment
from zipline.history.history import HistorySpec
from zipline.protocol import BarData
from zipline.utils.test_utils import to_utc


def mixed_frequency_expected_index(count, frequency):
    """
    Helper for enumerating expected indices for test_mixed_frequency.
    """
    env = TradingEnvironment.instance()
    minute = MIXED_FREQUENCY_MINUTES[count]

    if frequency == '1d':
        return [env.previous_open_and_close(minute)[1], minute]
    elif frequency == '1m':
        return [env.previous_market_minute(minute), minute]


def mixed_frequency_expected_data(count, frequency):
    """
    Helper for enumerating expected data test_mixed_frequency.
    """
    if frequency == '1d':
        # First day of this test is July 3rd, which is a half day.
        if count < 210:
            return [np.nan, count]
        else:
            return [209, count]
    elif frequency == '1m':
        if count == 0:
            return [np.nan, count]
        else:
            return [count - 1, count]


MIXED_FREQUENCY_MINUTES = TradingEnvironment.instance().market_minute_window(
    to_utc('2013-07-03 9:31AM'), 600,
)
ONE_MINUTE_PRICE_ONLY_SPECS = [
    HistorySpec(1, '1m', 'price', True, data_frequency='minute'),
]
DAILY_OPEN_CLOSE_SPECS = [
    HistorySpec(3, '1d', 'open_price', False, data_frequency='minute'),
    HistorySpec(3, '1d', 'close_price', False, data_frequency='minute'),
]
ILLIQUID_PRICES_SPECS = [
    HistorySpec(3, '1m', 'price', False, data_frequency='minute'),
    HistorySpec(5, '1m', 'price', True, data_frequency='minute'),
]
MIXED_FREQUENCY_SPECS = [
    HistorySpec(1, '1m', 'price', False, data_frequency='minute'),
    HistorySpec(2, '1m', 'price', False, data_frequency='minute'),
    HistorySpec(2, '1d', 'price', False, data_frequency='minute'),
]
MIXED_FIELDS_SPECS = [
    HistorySpec(3, '1m', 'price', True, data_frequency='minute'),
    HistorySpec(3, '1m', 'open_price', True, data_frequency='minute'),
    HistorySpec(3, '1m', 'close_price', True, data_frequency='minute'),
    HistorySpec(3, '1m', 'high', True, data_frequency='minute'),
    HistorySpec(3, '1m', 'low', True, data_frequency='minute'),
    HistorySpec(3, '1m', 'volume', True, data_frequency='minute'),
]


HISTORY_CONTAINER_TEST_CASES = {
    #      June 2013
    # Su Mo Tu We Th Fr Sa
    #                    1
    #  2  3  4  5  6  7  8
    #  9 10 11 12 13 14 15
    # 16 17 18 19 20 21 22
    # 23 24 25 26 27 28 29
    # 30

    'test one minute price only': {
        # A list of HistorySpec objects.
        'specs': ONE_MINUTE_PRICE_ONLY_SPECS,
        # Sids for the test.
        'sids': [1],
        # Start date for test.
        'dt': to_utc('2013-06-21 9:31AM'),
        # Sequency of updates to the container
        'updates': [
            BarData(
                {
                    1: {
                        'price': 5,
                        'dt': to_utc('2013-06-21 9:31AM'),
                    },
                },
            ),
            BarData(
                {
                    1: {
                        'price': 6,
                        'dt': to_utc('2013-06-21 9:32AM'),
                    },
                },
            ),
        ],
        # Expected results
        'expected': {
            ONE_MINUTE_PRICE_ONLY_SPECS[0].key_str: [
                pd.DataFrame(
                    data={
                        1: [5],
                    },
                    index=[
                        to_utc('2013-06-21 9:31AM'),
                    ],
                ),
                pd.DataFrame(
                    data={
                        1: [6],
                    },
                    index=[
                        to_utc('2013-06-21 9:32AM'),
                    ],
                ),
            ],
        },
    },

    'test daily open close': {
        # A list of HistorySpec objects.
        'specs': DAILY_OPEN_CLOSE_SPECS,

        # Sids for the test.
        'sids': [1],

        # Start date for test.
        'dt': to_utc('2013-06-21 9:31AM'),

        # Sequence of updates to the container
        'updates': [
            BarData(
                {
                    1: {
                        'open_price': 10,
                        'close_price': 11,
                        'dt': to_utc('2013-06-21 10:00AM'),
                    },
                },
            ),
            BarData(
                {
                    1: {
                        'open_price': 12,
                        'close_price': 13,
                        'dt': to_utc('2013-06-21 3:30PM'),
                    },
                },
            ),
            BarData(
                {
                    1: {
                        'open_price': 14,
                        'close_price': 15,
                        # Wait a full market day before the next bar.
                        # We should end up with nans for Monday the 24th.
                        'dt': to_utc('2013-06-25 9:31AM'),
                    },
                },
            ),
        ],

        # Dictionary mapping spec_key -> list of expected outputs
        'expected': {
            # open
            DAILY_OPEN_CLOSE_SPECS[0].key_str: [
                pd.DataFrame(
                    data={
                        1: [np.nan, np.nan, 10]
                    },
                    index=[
                        to_utc('2013-06-19 4:00PM'),
                        to_utc('2013-06-20 4:00PM'),
                        to_utc('2013-06-21 10:00AM'),
                    ],
                ),

                pd.DataFrame(
                    data={
                        1: [np.nan, np.nan, 10]
                    },
                    index=[
                        to_utc('2013-06-19 4:00PM'),
                        to_utc('2013-06-20 4:00PM'),
                        to_utc('2013-06-21 3:30PM'),
                    ],
                ),

                pd.DataFrame(
                    data={
                        1: [10, np.nan, 14]
                    },
                    index=[
                        to_utc('2013-06-21 4:00PM'),
                        to_utc('2013-06-24 4:00PM'),
                        to_utc('2013-06-25 9:31AM'),
                    ],
                ),
            ],
            # close
            DAILY_OPEN_CLOSE_SPECS[1].key_str: [
                pd.DataFrame(
                    data={
                        1: [np.nan, np.nan, 11]
                    },
                    index=[
                        to_utc('2013-06-19 4:00PM'),
                        to_utc('2013-06-20 4:00PM'),
                        to_utc('2013-06-21 10:00AM'),
                    ],
                ),

                pd.DataFrame(
                    data={
                        1: [np.nan, np.nan, 13]
                    },
                    index=[
                        to_utc('2013-06-19 4:00PM'),
                        to_utc('2013-06-20 4:00PM'),
                        to_utc('2013-06-21 3:30PM'),
                    ],
                ),

                pd.DataFrame(
                    data={
                        1: [13, np.nan, 15]
                    },
                    index=[
                        to_utc('2013-06-21 4:00PM'),
                        to_utc('2013-06-24 4:00PM'),
                        to_utc('2013-06-25 9:31AM'),
                    ],
                ),
            ],
        },
    },
    'test illiquid prices': {

        # A list of HistorySpec objects.
        'specs': ILLIQUID_PRICES_SPECS,

        # Sids for the test.
        'sids': [1],

        # Start date for test.
        'dt': to_utc('2013-06-28 9:31AM'),

        # Sequence of updates to the container
        'updates': [
            BarData(
                {
                    1: {
                        'price': 10,
                        'dt': to_utc('2013-06-28 9:31AM'),
                    },
                },
            ),
            BarData(
                {
                    1: {
                        'price': 11,
                        'dt': to_utc('2013-06-28 9:32AM'),
                    },
                },
            ),
            BarData(
                {
                    1: {
                        'price': 12,
                        'dt': to_utc('2013-06-28 9:33AM'),
                    },
                },
            ),
            BarData(
                {
                    1: {
                        'price': 13,
                        # Note: Skipping 9:34 to simulate illiquid bar/missing
                        # data.
                        'dt': to_utc('2013-06-28 9:35AM'),
                    },
                },
            ),
        ],

        # Dictionary mapping spec_key -> list of expected outputs
        'expected': {
            ILLIQUID_PRICES_SPECS[0].key_str: [
                pd.DataFrame(
                    data={
                        1: [np.nan, np.nan, 10],
                    },
                    index=[
                        to_utc('2013-06-27 3:59PM'),
                        to_utc('2013-06-27 4:00PM'),
                        to_utc('2013-06-28 9:31AM'),
                    ],
                ),

                pd.DataFrame(
                    data={
                        1: [np.nan, 10, 11],
                    },
                    index=[
                        to_utc('2013-06-27 4:00PM'),
                        to_utc('2013-06-28 9:31AM'),
                        to_utc('2013-06-28 9:32AM'),
                    ],
                ),

                pd.DataFrame(
                    data={
                        1: [10, 11, 12],
                    },
                    index=[
                        to_utc('2013-06-28 9:31AM'),
                        to_utc('2013-06-28 9:32AM'),
                        to_utc('2013-06-28 9:33AM'),
                    ],
                ),

                # Since there's no update for 9:34, this is called at 9:35.
                pd.DataFrame(
                    data={
                        1: [12, np.nan, 13],
                    },
                    index=[
                        to_utc('2013-06-28 9:33AM'),
                        to_utc('2013-06-28 9:34AM'),
                        to_utc('2013-06-28 9:35AM'),
                    ],
                ),
            ],

            ILLIQUID_PRICES_SPECS[1].key_str: [
                pd.DataFrame(
                    data={
                        1: [np.nan, np.nan, np.nan, np.nan, 10],
                    },
                    index=[
                        to_utc('2013-06-27 3:57PM'),
                        to_utc('2013-06-27 3:58PM'),
                        to_utc('2013-06-27 3:59PM'),
                        to_utc('2013-06-27 4:00PM'),
                        to_utc('2013-06-28 9:31AM'),
                    ],
                ),

                pd.DataFrame(
                    data={
                        1: [np.nan, np.nan, np.nan, 10, 11],
                    },
                    index=[
                        to_utc('2013-06-27 3:58PM'),
                        to_utc('2013-06-27 3:59PM'),
                        to_utc('2013-06-27 4:00PM'),
                        to_utc('2013-06-28 9:31AM'),
                        to_utc('2013-06-28 9:32AM'),
                    ],
                ),

                pd.DataFrame(
                    data={
                        1: [np.nan, np.nan, 10, 11, 12],
                    },
                    index=[
                        to_utc('2013-06-27 3:59PM'),
                        to_utc('2013-06-27 4:00PM'),
                        to_utc('2013-06-28 9:31AM'),
                        to_utc('2013-06-28 9:32AM'),
                        to_utc('2013-06-28 9:33AM'),
                    ],
                ),

                # Since there's no update for 9:34, this is called at 9:35.
                # The 12 value from 9:33 should be forward-filled.
                pd.DataFrame(
                    data={
                        1: [10, 11, 12, 12, 13],
                    },
                    index=[
                        to_utc('2013-06-28 9:31AM'),
                        to_utc('2013-06-28 9:32AM'),
                        to_utc('2013-06-28 9:33AM'),
                        to_utc('2013-06-28 9:34AM'),
                        to_utc('2013-06-28 9:35AM'),
                    ],
                ),
            ],
        },
    },

    'test mixed frequencies': {

        # A list of HistorySpec objects.
        'specs': MIXED_FREQUENCY_SPECS,

        # Sids for the test.
        'sids': [1],

        # Start date for test.
        #      July 2013
        # Su Mo Tu We Th Fr Sa
        #     1  2  3  4  5  6
        #  7  8  9 10 11 12 13
        # 14 15 16 17 18 19 20
        # 21 22 23 24 25 26 27
        # 28 29 30 31
        'dt': to_utc('2013-07-03 9:31AM'),

        # Sequence of updates to the container
        'updates': [
            BarData(
                {
                    1: {
                        'price': count,
                        'dt': dt,
                    }
                }
            )
            for count, dt in enumerate(MIXED_FREQUENCY_MINUTES)
        ],

        # Dictionary mapping spec_key -> list of expected outputs.
        'expected': {

            MIXED_FREQUENCY_SPECS[0].key_str: [
                pd.DataFrame(
                    data={
                        1: [count],
                    },
                    index=[minute],
                )
                for count, minute in enumerate(MIXED_FREQUENCY_MINUTES)
            ],

            MIXED_FREQUENCY_SPECS[1].key_str: [
                pd.DataFrame(
                    data={
                        1: mixed_frequency_expected_data(count, '1m'),
                    },
                    index=mixed_frequency_expected_index(count, '1m'),
                )
                for count in range(len(MIXED_FREQUENCY_MINUTES))
            ],

            MIXED_FREQUENCY_SPECS[2].key_str: [
                pd.DataFrame(
                    data={
                        1: mixed_frequency_expected_data(count, '1d'),
                    },
                    index=mixed_frequency_expected_index(count, '1d'),
                )
                for count in range(len(MIXED_FREQUENCY_MINUTES))
            ]
        },
    },

    'test multiple fields and sids': {

        # A list of HistorySpec objects.
        'specs': MIXED_FIELDS_SPECS,

        # Sids for the test.
        'sids': [1, 10],

        # Start date for test.
        'dt': to_utc('2013-06-28 9:31AM'),

        # Sequence of updates to the container
        'updates': [
            BarData(
                {
                    1: {
                        'dt': dt,
                        'price': count,
                        'open_price': count,
                        'close_price': count,
                        'high': count,
                        'low': count,
                        'volume': count,
                    },
                    10: {
                        'dt': dt,
                        'price': count * 10,
                        'open_price': count * 10,
                        'close_price': count * 10,
                        'high': count * 10,
                        'low': count * 10,
                        'volume': count * 10,
                    },
                },
            )
            for count, dt in enumerate([
                to_utc('2013-06-28 9:31AM'),
                to_utc('2013-06-28 9:32AM'),
                to_utc('2013-06-28 9:33AM'),
                # NOTE: No update for 9:34
                to_utc('2013-06-28 9:35AM'),
            ])
        ],

        # Dictionary mapping spec_key -> list of expected outputs
        'expected': dict(

            # Build a dict from a list of tuples.  Doing it this way because
            # there are two distinct cases we want to test: forward-fillable
            # fields and non-forward-fillable fields.
            [
                (
                    # Non forward-fill fields
                    key,
                    [
                        pd.DataFrame(
                            data={
                                1: [np.nan, np.nan, 0],
                                10: [np.nan, np.nan, 0],
                            },
                            index=[
                                to_utc('2013-06-27 3:59PM'),
                                to_utc('2013-06-27 4:00PM'),
                                to_utc('2013-06-28 9:31AM'),
                            ],

                        ),
                        pd.DataFrame(
                            data={
                                1: [np.nan, 0, 1],
                                10: [np.nan, 0, 10],
                            },
                            index=[
                                to_utc('2013-06-27 4:00PM'),
                                to_utc('2013-06-28 9:31AM'),
                                to_utc('2013-06-28 9:32AM'),
                            ],
                        ),

                        pd.DataFrame(
                            data={
                                1: [0, 1, 2],
                                10: [0, 10, 20],
                            },
                            index=[
                                to_utc('2013-06-28 9:31AM'),
                                to_utc('2013-06-28 9:32AM'),
                                to_utc('2013-06-28 9:33AM'),
                            ],

                        ),
                        pd.DataFrame(
                            data={
                                1: [2, np.nan, 3],
                                10: [20, np.nan, 30],
                            },
                            index=[
                                to_utc('2013-06-28 9:33AM'),
                                to_utc('2013-06-28 9:34AM'),
                                to_utc('2013-06-28 9:35AM'),
                            ],
                            # For volume, when we are missing data, we replace
                            # it with 0s to show that no trades occured.
                        ).fillna(0 if 'volume' in key else np.nan),
                    ],
                )
                for key in [spec.key_str for spec in MIXED_FIELDS_SPECS
                            if spec.field not in HistorySpec.FORWARD_FILLABLE]
            ]

            +  # Concatenate the expected results for non-ffillable with
               # expected result for ffillable.
            [
                (
                    # Forward-fillable fields
                    key,
                    [
                        pd.DataFrame(
                            data={
                                1: [np.nan, np.nan, 0],
                                10: [np.nan, np.nan, 0],
                            },
                            index=[
                                to_utc('2013-06-27 3:59PM'),
                                to_utc('2013-06-27 4:00PM'),
                                to_utc('2013-06-28 9:31AM'),
                            ],
                        ),

                        pd.DataFrame(
                            data={
                                1: [np.nan, 0, 1],
                                10: [np.nan, 0, 10],
                            },
                            index=[
                                to_utc('2013-06-27 4:00PM'),
                                to_utc('2013-06-28 9:31AM'),
                                to_utc('2013-06-28 9:32AM'),
                            ],
                        ),

                        pd.DataFrame(
                            data={
                                1: [0, 1, 2],
                                10: [0, 10, 20],
                            },
                            index=[
                                to_utc('2013-06-28 9:31AM'),
                                to_utc('2013-06-28 9:32AM'),
                                to_utc('2013-06-28 9:33AM'),
                            ],
                        ),

                        pd.DataFrame(
                            data={
                                1: [2, 2, 3],
                                10: [20, 20, 30],
                            },
                            index=[
                                to_utc('2013-06-28 9:33AM'),
                                to_utc('2013-06-28 9:34AM'),
                                to_utc('2013-06-28 9:35AM'),
                            ],
                        ),
                    ],
                )
                for key in [spec.key_str for spec in MIXED_FIELDS_SPECS
                            if spec.field in HistorySpec.FORWARD_FILLABLE]
            ]
        ),
    },
}
