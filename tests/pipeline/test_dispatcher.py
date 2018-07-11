import os

from zipline.data import bundles
from zipline.pipeline import USEquityPricingLoader
from zipline.pipeline.data import (
    Column,
    DataSet,
    BoundColumn,
    USEquityPricing,
)
from zipline.pipeline.dispatcher import PipelineDispatcher
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.sentinels import NotSpecified
from zipline.testing import ZiplineTestCase
from zipline.testing.fixtures import WithAdjustmentReader
from zipline.testing.predicates import (
    assert_raises_str,
    assert_equal,
)
from zipline.utils.numpy_utils import float64_dtype


class FakeDataSet(DataSet):
    test_col = Column(float64_dtype)


class FakeColumn(BoundColumn):
    pass


class FakePipelineLoader(PipelineLoader):

    def load_adjusted_array(self, columns, dates, assets, mask):
        pass


class UnrelatedType(object):
    pass


class PipelineDispatcherTestCase(WithAdjustmentReader, ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(PipelineDispatcherTestCase, cls).init_class_fixtures()
        cls.default_pipeline_loader = USEquityPricingLoader(
            cls.bcolz_equity_daily_bar_reader,
            cls.adjustment_reader,
        )

    def test_load_not_registered(self):
        fake_col_instance = FakeColumn(
            float64_dtype,
            NotSpecified,
            FakeDataSet,
            'test',
            None,
            {},
        )
        fake_pl_instance = FakePipelineLoader()
        pipeline_dispatcher = PipelineDispatcher(
            column_loaders={fake_col_instance: fake_pl_instance}
        )

        expected_dict = {fake_col_instance: fake_pl_instance}
        assert_equal(pipeline_dispatcher.column_loaders, expected_dict)

        msg = "No pipeline loader registered for %s" % USEquityPricing.close
        with assert_raises_str(LookupError, msg):
            pipeline_dispatcher(USEquityPricing.close)

    def test_register_unrelated_type(self):
        pipeline_dispatcher = PipelineDispatcher()
        fake_pl_instance = FakePipelineLoader()

        msg = "Data provided is neither a BoundColumn nor a DataSet"
        with assert_raises_str(TypeError, msg):
            pipeline_dispatcher.register(UnrelatedType, fake_pl_instance)

    def test_passive_registration(self):
        pipeline_dispatcher = PipelineDispatcher()
        assert_equal(pipeline_dispatcher.column_loaders, {})

        # imitate user registering a custom pipeline loader first
        custom_loader = FakePipelineLoader()
        pipeline_dispatcher.register(USEquityPricing.close, custom_loader)
        expected_dict = {USEquityPricing.close: custom_loader}
        assert_equal(pipeline_dispatcher.column_loaders, expected_dict)

        # now check that trying to register something else won't change it
        pipeline_dispatcher.register(
            USEquityPricing.close, self.default_pipeline_loader
        )
        assert_equal(pipeline_dispatcher.column_loaders, expected_dict)

    def test_normal_ops(self):
        fake_loader_instance = FakePipelineLoader()
        fake_col_instance = FakeColumn(
            float64_dtype,
            NotSpecified,
            FakeDataSet,
            'test',
            None,
            {},
        )
        pipeline_dispatcher = PipelineDispatcher(
            column_loaders={
                fake_col_instance: fake_loader_instance
            },
            dataset_loaders={
                FakeDataSet: fake_loader_instance
            }
        )
        expected_dict = {
            fake_col_instance: fake_loader_instance,
            FakeDataSet.test_col: fake_loader_instance,
        }
        assert_equal(pipeline_dispatcher.column_loaders, expected_dict)

        pipeline_dispatcher.register(
            USEquityPricing.close, fake_loader_instance
        )
        expected_dict = {
            fake_col_instance: fake_loader_instance,
            FakeDataSet.test_col: fake_loader_instance,
            USEquityPricing.close: fake_loader_instance,
        }
        assert_equal(pipeline_dispatcher.column_loaders, expected_dict)

        assert_equal(
            pipeline_dispatcher(fake_col_instance), fake_loader_instance
        )
        assert_equal(
            pipeline_dispatcher(FakeDataSet.test_col), fake_loader_instance
        )
        assert_equal(
            pipeline_dispatcher(USEquityPricing.close), fake_loader_instance
        )
