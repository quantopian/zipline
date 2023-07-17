import pandas as pd

from zipline.testing.fixtures import (
    WithCreateBarData,
    WithMakeAlgo,
    ZiplineTestCase,
)
import pytest

reference_missing_position_by_unexpected_type_algo = """
def initialize(context):
    pass

def handle_data(context, data):
    context.portfolio.positions["foobar"]
"""


class TestAPIShim(WithCreateBarData, WithMakeAlgo, ZiplineTestCase):

    START_DATE = pd.Timestamp("2016-01-05")
    END_DATE = pd.Timestamp("2016-01-28")
    SIM_PARAMS_DATA_FREQUENCY = "minute"

    sids = ASSET_FINDER_EQUITY_SIDS = 1, 2, 3

    @classmethod
    def init_class_fixtures(cls):
        super(TestAPIShim, cls).init_class_fixtures()

        cls.asset1 = cls.asset_finder.retrieve_asset(1)
        cls.asset2 = cls.asset_finder.retrieve_asset(2)
        cls.asset3 = cls.asset_finder.retrieve_asset(3)

    def create_algo(self, code, filename=None, sim_params=None):
        if sim_params is None:
            sim_params = self.sim_params

        return self.make_algo(
            script=code, sim_params=sim_params, algo_filename=filename
        )

    def test_reference_empty_position_by_unexpected_type(self):
        algo = self.create_algo(reference_missing_position_by_unexpected_type_algo)
        with pytest.raises(
            ValueError,
            match="Position lookup expected a value of type Asset but got str instead",
        ):
            algo.run()
