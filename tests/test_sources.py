from unittest2 import TestCase

import zipline.utils.factory as factory
from zipline.gens.tradegens import DataFrameSource

class TestDataFrameSource(TestCase):
    def test_streaming_of_df(self):
        source, df = factory.create_test_df_source()

        for expected_dt, expected_price in df.iterrows():
            sid0 = source.next()
            sid1 = source.next()

            assert expected_dt == sid0.dt == sid1.dt
            assert expected_price[0] == sid0.price
            assert expected_price[1] == sid1.price

    def test_sid_filtering(self):
        _, df = factory.create_test_df_source()
        source = DataFrameSource(df, sids=[0])
        assert 1 not in [event.sid for event in source], \
        "DataFrameSource should only stream selected sid 0, not sid 1."