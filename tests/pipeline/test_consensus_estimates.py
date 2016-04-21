"""
Tests for the reference loader for ConsensusEstimates.
"""
import blaze as bz
from blaze.compute.core import swap_resources_into_scope
import pandas as pd
from six import iteritems

from zipline.pipeline.common import (
    ACTUAL_VALUE_FIELD_NAME,
    COUNT_FIELD_NAME,
    FISCAL_QUARTER_FIELD_NAME,
    FISCAL_YEAR_FIELD_NAME,
    HIGH_FIELD_NAME,
    LOW_FIELD_NAME,
    MEAN_FIELD_NAME,
    NEXT_COUNT,
    NEXT_FISCAL_QUARTER,
    NEXT_FISCAL_YEAR,
    NEXT_HIGH,
    NEXT_LOW,
    NEXT_RELEASE_DATE,
    NEXT_STANDARD_DEVIATION,
    PREVIOUS_ACTUAL_VALUE,
    PREVIOUS_COUNT,
    PREVIOUS_FISCAL_QUARTER,
    PREVIOUS_FISCAL_YEAR,
    PREVIOUS_HIGH,
    PREVIOUS_LOW,
    PREVIOUS_MEAN, NEXT_MEAN,
    PREVIOUS_RELEASE_DATE,
    PREVIOUS_STANDARD_DEVIATION,
    RELEASE_DATE_FIELD_NAME,
    STANDARD_DEVIATION_FIELD_NAME,
    SID_FIELD_NAME)
from zipline.pipeline.data import ConsensusEstimates
from zipline.pipeline.loaders.consensus_estimates import (
    ConsensusEstimatesLoader
)
from zipline.pipeline.loaders.blaze import BlazeConsensusEstimatesLoader
from zipline.pipeline.loaders.utils import (
    zip_with_floats
)
from zipline.testing.fixtures import (
    ZiplineTestCase,
    WithNextAndPreviousEventDataLoader
)

consensus_estimates_cases = [
    # K1--K2--A1--A2.
    pd.DataFrame({
        ACTUAL_VALUE_FIELD_NAME: (100, 200),
        STANDARD_DEVIATION_FIELD_NAME: (.5, .6),
        COUNT_FIELD_NAME: (1, 2),
        FISCAL_QUARTER_FIELD_NAME: (1, 1),
        HIGH_FIELD_NAME: (.6, .7),
        MEAN_FIELD_NAME: (.1, .2),
        FISCAL_YEAR_FIELD_NAME: (2014, 2014),
        LOW_FIELD_NAME: (.05, .06),
    }),
    # K1--K2--A2--A1.
    pd.DataFrame({
        ACTUAL_VALUE_FIELD_NAME: (200, 300),
        STANDARD_DEVIATION_FIELD_NAME: (.6, .7),
        COUNT_FIELD_NAME: (2, 3),
        FISCAL_QUARTER_FIELD_NAME: (1, 1),
        HIGH_FIELD_NAME: (.7, .8),
        MEAN_FIELD_NAME: (.2, .3),
        FISCAL_YEAR_FIELD_NAME: (2014, 2014),
        LOW_FIELD_NAME: (.06, .07),
    }),
    # K1--A1--K2--A2.
    pd.DataFrame({
        ACTUAL_VALUE_FIELD_NAME: (300, 400),
        STANDARD_DEVIATION_FIELD_NAME: (.7, .8),
        COUNT_FIELD_NAME: (3, 4),
        FISCAL_QUARTER_FIELD_NAME: (1, 1),
        HIGH_FIELD_NAME: (.8, .9),
        MEAN_FIELD_NAME: (.3, .4),
        FISCAL_YEAR_FIELD_NAME: (2014, 2014),
        LOW_FIELD_NAME: (.07, .08),
    }),
    # K1 == K2.
    pd.DataFrame({
        ACTUAL_VALUE_FIELD_NAME: (400, 500),
        STANDARD_DEVIATION_FIELD_NAME: (.8, .9),
        COUNT_FIELD_NAME: (4, 5),
        FISCAL_QUARTER_FIELD_NAME: (1, 1),
        HIGH_FIELD_NAME: (.9, 1.0),
        MEAN_FIELD_NAME: (.4, .5),
        FISCAL_YEAR_FIELD_NAME: (2014, 2014),
        LOW_FIELD_NAME: (.08, .09),
    }),
    pd.DataFrame(
        columns=[ACTUAL_VALUE_FIELD_NAME,
                 STANDARD_DEVIATION_FIELD_NAME,
                 COUNT_FIELD_NAME,
                 FISCAL_QUARTER_FIELD_NAME,
                 HIGH_FIELD_NAME,
                 MEAN_FIELD_NAME,
                 FISCAL_YEAR_FIELD_NAME,
                 LOW_FIELD_NAME],
        dtype='datetime64[ns]'
    ),
]

prev_actual_value = [
    ['NaN', 100, 200],
    ['NaN', 300, 200],
    ['NaN', 300, 400],
    ['NaN', 400, 500],
    ['NaN']
]

next_standard_deviation = [
    ['NaN', .5, .6, 'NaN'],
    ['NaN', .6, .7, .6, 'NaN'],
    ['NaN', .7, 'NaN', .8, 'NaN'],
    ['NaN', .8, .9, 'NaN'],
    ['NaN']
]

prev_standard_deviation = [
    ['NaN', .5, .6],
    ['NaN', .7, .6],
    ['NaN', .7, .8],
    ['NaN', .8, .9],
    ['NaN']
]

next_count = [
    ['NaN', 1, 2, 'NaN'],
    ['NaN', 2, 3, 2, 'NaN'],
    ['NaN', 3, 'NaN', 4, 'NaN'],
    ['NaN', 4, 5, 'NaN'],
    ['NaN']
]

prev_count = [
    ['NaN', 1, 2],
    ['NaN', 3, 2],
    ['NaN', 3, 4],
    ['NaN', 4, 5],
    ['NaN']
]

next_fiscal_quarter = [
    ['NaN', 1, 1, 'NaN'],
    ['NaN', 1, 1, 1, 'NaN'],
    ['NaN', 1, 'NaN', 1, 'NaN'],
    ['NaN', 1, 1, 'NaN'],
    ['NaN']
]

prev_fiscal_quarter = [
    ['NaN', 1, 1],
    ['NaN', 1, 1],
    ['NaN', 1, 1],
    ['NaN', 1, 1],
    ['NaN']
]

next_high = [
    ['NaN', .6, .7, 'NaN'],
    ['NaN', .7, .8, .7, 'NaN'],
    ['NaN', .8, 'NaN', .9, 'NaN'],
    ['NaN', .9, 1.0, 'NaN'],
    ['NaN']
]

prev_high = [
    ['NaN', .6, .7],
    ['NaN', .8, .7],
    ['NaN', .8, .9],
    ['NaN', .9, 1.0],
    ['NaN']
]

next_mean = [
    ['NaN', .1, .2, 'NaN'],
    ['NaN', .2, .3, .2, 'NaN'],
    ['NaN', .3, 'NaN', .4, 'NaN'],
    ['NaN', .4, .5, 'NaN'],
    ['NaN']
]

prev_mean = [
    ['NaN', .1, .2],
    ['NaN', .3, .2],
    ['NaN', .3, .4],
    ['NaN', .4, .5],
    ['NaN']
]

next_fiscal_year = [
    ['NaN', 2014, 2014, 'NaN'],
    ['NaN', 2014, 2014, 2014, 'NaN'],
    ['NaN', 2014, 'NaN', 2014, 'NaN'],
    ['NaN', 2014, 2014, 'NaN'],
    ['NaN']
]

prev_fiscal_year = [
    ['NaN', 2014, 2014],
    ['NaN', 2014, 2014],
    ['NaN', 2014, 2014],
    ['NaN', 2014, 2014],
    ['NaN']
]

next_low = [
    ['NaN', .05, .06, 'NaN'],
    ['NaN', .06, .07, .06, 'NaN'],
    ['NaN', .07, 'NaN', .08, 'NaN'],
    ['NaN', .08, .09, 'NaN'],
    ['NaN']
]

prev_low = [
    ['NaN', .05, .06],
    ['NaN', .07, .06],
    ['NaN', .07, .08],
    ['NaN', .08, .09],
    ['NaN']
]

field_name_to_expected_col = {
    PREVIOUS_ACTUAL_VALUE: prev_actual_value,
    PREVIOUS_STANDARD_DEVIATION: prev_standard_deviation,
    NEXT_STANDARD_DEVIATION: next_standard_deviation,
    PREVIOUS_COUNT: prev_count,
    NEXT_COUNT: next_count,
    PREVIOUS_FISCAL_QUARTER: prev_fiscal_quarter,
    NEXT_FISCAL_QUARTER: next_fiscal_quarter,
    PREVIOUS_HIGH: prev_high,
    NEXT_HIGH: next_high,
    PREVIOUS_MEAN: prev_mean,
    NEXT_MEAN: next_mean,
    PREVIOUS_FISCAL_YEAR: prev_fiscal_year,
    NEXT_FISCAL_YEAR: next_fiscal_year,
    PREVIOUS_LOW: prev_low,
    NEXT_LOW: next_low
}


class ConsensusEstimatesLoaderTestCase(WithNextAndPreviousEventDataLoader,
                                       ZiplineTestCase):
    """
    Tests for loading the consensus estimates data.
    """
    pipeline_columns = {
        PREVIOUS_ACTUAL_VALUE:
            ConsensusEstimates.previous_actual_value.latest,
        NEXT_RELEASE_DATE:
            ConsensusEstimates.next_release_date.latest,
        PREVIOUS_RELEASE_DATE:
            ConsensusEstimates.previous_release_date.latest,
        PREVIOUS_STANDARD_DEVIATION:
            ConsensusEstimates.previous_standard_deviation.latest,
        NEXT_STANDARD_DEVIATION:
            ConsensusEstimates.next_standard_deviation.latest,
        PREVIOUS_COUNT:
            ConsensusEstimates.previous_count.latest,
        NEXT_COUNT:
            ConsensusEstimates.next_count.latest,
        PREVIOUS_FISCAL_QUARTER:
            ConsensusEstimates.previous_fiscal_quarter.latest,
        NEXT_FISCAL_QUARTER:
            ConsensusEstimates.next_fiscal_quarter.latest,
        PREVIOUS_HIGH:
            ConsensusEstimates.previous_high.latest,
        NEXT_HIGH:
            ConsensusEstimates.next_high.latest,
        PREVIOUS_MEAN:
            ConsensusEstimates.previous_mean.latest,
        NEXT_MEAN:
            ConsensusEstimates.next_mean.latest,
        PREVIOUS_FISCAL_YEAR:
            ConsensusEstimates.previous_fiscal_year.latest,
        NEXT_FISCAL_YEAR:
            ConsensusEstimates.next_fiscal_year.latest,
        PREVIOUS_LOW:
            ConsensusEstimates.previous_low.latest,
        NEXT_LOW:
            ConsensusEstimates.next_low.latest
    }

    @classmethod
    def get_dataset(cls):
        return {sid:
                pd.concat([
                    cls.base_cases[sid].rename(columns={
                        'other_date': RELEASE_DATE_FIELD_NAME
                    }),
                    df
                ], axis=1)
                for sid, df in enumerate(consensus_estimates_cases)}

    loader_type = ConsensusEstimatesLoader

    def setup(self, dates):
        cols = {
            PREVIOUS_RELEASE_DATE:
                self.get_expected_previous_event_dates(dates),
            NEXT_RELEASE_DATE: self.get_expected_next_event_dates(dates)
        }
        for field_name in field_name_to_expected_col:
            cols[field_name] = self.get_sids_to_frames(
                zip_with_floats, field_name_to_expected_col[field_name],
                self.prev_date_intervals
                if field_name.startswith("previous")
                else self.next_date_intervals,
                dates
            )
        return cols


class BlazeConsensusEstimatesLoaderTestCase(ConsensusEstimatesLoaderTestCase):
    loader_type = BlazeConsensusEstimatesLoader

    def pipeline_event_loader_args(self, dates):
        _, mapping = super(
            BlazeConsensusEstimatesLoaderTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        frames = []
        for sid, df in iteritems(mapping):
            frame = df.copy()
            frame[SID_FIELD_NAME] = sid
            frames.append(frame)
        return bz.data(pd.concat(frames).reset_index(drop=True)),


class BlazeConsensusEstimatesLoaderNotInteractiveTestCase(
    BlazeConsensusEstimatesLoaderTestCase
):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """

    def pipeline_event_loader_args(self, dates):
        (bound_expr,) = super(
            BlazeConsensusEstimatesLoaderNotInteractiveTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})
