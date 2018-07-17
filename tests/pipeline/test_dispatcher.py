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


class PipelineDispatcherTestCase(ZiplineTestCase):

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
            {fake_col_instance: fake_pl_instance}
        )

        expected_dict = {fake_col_instance: fake_pl_instance}
        assert_equal(pipeline_dispatcher._column_loaders, expected_dict)

        msg = "No pipeline loader registered for %s" % USEquityPricing.close
        with assert_raises_str(LookupError, msg):
            pipeline_dispatcher(USEquityPricing.close)

    def test_register_unrelated_type(self):
        fake_pl_instance = FakePipelineLoader()

        msg = "%s is neither a BoundColumn nor a DataSet" % UnrelatedType
        with assert_raises_str(TypeError, msg):
            PipelineDispatcher(
                {UnrelatedType: fake_pl_instance}
            )

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
        pipeline_dispatcher = PipelineDispatcher({
                fake_col_instance: fake_loader_instance,
                FakeDataSet: fake_loader_instance
        })

        expected_dict = {
            fake_col_instance: fake_loader_instance,
            FakeDataSet.test_col: fake_loader_instance,
        }
        assert_equal(pipeline_dispatcher._column_loaders, expected_dict)
        assert_equal(
            pipeline_dispatcher(fake_col_instance), fake_loader_instance
        )
        assert_equal(
            pipeline_dispatcher(FakeDataSet.test_col), fake_loader_instance
        )
