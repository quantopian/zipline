from collections import deque

import pytz
import numpy as np
import pandas as pd

from datetime import datetime
from unittest import TestCase

from zipline.utils.test_utils import setup_logger

import zipline.utils.factory as factory

from zipline.test_algorithms import BatchTransformAlgorithm


class TestBatchTransform(TestCase):
    def setUp(self):
        self.sim_params = factory.create_simulation_parameters(
            start=datetime(1990, 1, 1, tzinfo=pytz.utc),
            end=datetime(1990, 1, 8, tzinfo=pytz.utc)
        )
        setup_logger(self)
        self.source, self.df = \
            factory.create_test_df_source(self.sim_params)

    def test_event_window(self):
        algo = BatchTransformAlgorithm(sim_params=self.sim_params)
        algo.run(self.source)
        wl = algo.window_length
        # The following assertion depend on window length of 3
        self.assertEqual(wl, 3)
        self.assertEqual(algo.history_return_price_class[:wl],
                         [None] * wl,
                         "First three iterations should return None." + "\n" +
                         "i.e. no returned values until window is full'" +
                         "%s" % (algo.history_return_price_class,))
        self.assertEqual(algo.history_return_price_decorator[:wl],
                         [None] * wl,
                         "First three iterations should return None." + "\n" +
                         "i.e. no returned values until window is full'" +
                         "%s" % (algo.history_return_price_decorator,))

        # After three Nones, the next value should be a data frame
        self.assertTrue(isinstance(
            algo.history_return_price_class[wl],
            pd.DataFrame)
        )

        # Test whether arbitrary fields can be added to datapanel
        field = algo.history_return_arbitrary_fields[-1]
        self.assertTrue(
            'arbitrary' in field.items,
            'datapanel should contain column arbitrary'
        )

        self.assertTrue(all(
            field['arbitrary'].values.flatten() ==
            [123] * algo.window_length),
            'arbitrary dataframe should contain only "test"'
        )

        for data in algo.history_return_sid_filter[wl:]:
            self.assertIn(0, data.columns)
            self.assertNotIn(1, data.columns)

        for data in algo.history_return_field_filter[wl:]:
            self.assertIn('price', data.items)
            self.assertNotIn('ignore', data.items)

        for data in algo.history_return_field_no_filter[wl:]:
            self.assertIn('price', data.items)
            self.assertIn('ignore', data.items)

        for data in algo.history_return_ticks[wl:]:
            self.assertTrue(isinstance(data, deque))

        for data in algo.history_return_not_full:
            self.assertIsNot(data, None)

        # test overloaded class
        for test_history in [algo.history_return_price_class,
                             algo.history_return_price_decorator]:
            # starting at window length, the window should contain
            # consecutive (of window length) numbers up till the end.
            for i in range(algo.window_length, len(test_history)):
                np.testing.assert_array_equal(
                    range(i - algo.window_length + 1, i + 1),
                    test_history[i].values.flatten()
                )

    def test_passing_of_args(self):
        algo = BatchTransformAlgorithm(1,
                                       kwarg='str', sim_params=self.sim_params)
        self.assertEqual(algo.args, (1,))
        self.assertEqual(algo.kwargs, {'kwarg': 'str'})

        algo.run(self.source)
        expected_item = ((1, ), {'kwarg': 'str'})
        self.assertEqual(
            algo.history_return_args,
            [
                # 1990-01-01 - market holiday, no event
                # 1990-01-02 - window not full
                None,
                # 1990-01-03 - window not full
                None,
                # 1990-01-04 - window not full, 3rd event
                None,
                # 1990-01-05 - window now full
                expected_item,
                # 1990-01-08 - window now full
                expected_item
            ])
