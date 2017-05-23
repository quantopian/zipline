from zipline.pipeline import Pipeline, run_chunked_pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Returns
from zipline.testing import ZiplineTestCase
from zipline.testing.fixtures import WithEquityPricingPipelineEngine


class ChunkedPipelineTestCase(WithEquityPricingPipelineEngine,
                              ZiplineTestCase):

    def test_run_chunked_pipeline(self):
        """
        Test that running a pipeline in chunks produces the same result as if
        it were run all at once
        """
        pipe = Pipeline(
            columns={
                'close': USEquityPricing.close.latest,
                'returns': Returns(window_length=2),
            },
        )
        sessions = self.nyse_calendar.all_sessions
        start_date = sessions[sessions.get_loc(self.START_DATE) + 2]

        pipeline_result = self.pipeline_engine.run_pipeline(
            pipe,
            start_date=start_date,
            end_date=self.END_DATE,
        )
        chunked_result = run_chunked_pipeline(
            engine=self.pipeline_engine,
            pipeline=pipe,
            start_date=start_date,
            end_date=self.END_DATE,
            chunksize=22
        )
        self.assertTrue(chunked_result.equals(pipeline_result))
