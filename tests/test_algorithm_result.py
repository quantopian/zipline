import pandas as pd

from zipline.algorithm import TradingAlgorithm
from zipline.result import AlgorithmResult
import zipline.api as api
from zipline.testing.predicates import assert_equal
from zipline.testing.fixtures import WithDataPortal, ZiplineTestCase


def T(s):
    return pd.Timestamp(s, tz='UTC')


class ResultTestCase(WithDataPortal, ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = range(5)
    START_DATE = T('2017-01-02')
    END_DATE = T('2017-01-31')

    def test_roundtrip_to_disk(self):

        def initialize(context):
            context.assets = map(api.sid, self.ASSET_FINDER_EQUITY_SIDS)
            context.sign = 1

        def handle_data(context, data):
            for a in context.assets:
                api.order(a, 1 * context.sign)
            context.sign *= -1

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            data_frequency='daily',
            start=self.START_DATE,
            end=self.START_DATE + (5 * self.trading_calendar.day),
            env=self.env,
        )
        result = algo.run(data=self.data_portal)
        test_dir = self.tmpdir.getpath('result')

        result.save(test_dir)
        roundtripped = AlgorithmResult.load(test_dir)
        assert_equal(roundtripped, result)
