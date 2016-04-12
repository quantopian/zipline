"""
Tests for the reference loader for Buyback Authorizations.
"""
import blaze as bz
from blaze.compute.core import swap_resources_into_scope
import pandas as pd
from six import iteritems

from zipline.pipeline.common import(
    BUYBACK_ANNOUNCEMENT_FIELD_NAME,
    BUYBACK_TYPE_FIELD_NAME,
    DAYS_SINCE_PREV,
    PREVIOUS_BUYBACK_ANNOUNCEMENT,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
    VALUE_FIELD_NAME,
    VALUE_TYPE_FIELD_NAME,
    PREVIOUS_VALUE,
    PREVIOUS_VALUE_TYPE,
    PREVIOUS_BUYBACK_TYPE,
)
from zipline.pipeline.data import BuybackAuthorizations
from zipline.pipeline.factors.events import BusinessDaysSinceBuybackAuth
from zipline.pipeline.loaders.buyback_auth import BuybackAuthorizationsLoader
from zipline.pipeline.loaders.blaze import BlazeBuybackAuthorizationsLoader
from zipline.pipeline.loaders.utils import (
    zip_with_floats,
    zip_with_dates,
    get_values_for_date_ranges,
    zip_with_strs)
from zipline.testing.fixtures import (
    WithPipelineEventDataLoader, ZiplineTestCase
)

date_intervals = [
    [['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-09'],
     ['2014-01-10', '2014-01-31']]
]

buyback_authorizations_cases = [
    pd.DataFrame({
        VALUE_FIELD_NAME: [1, 15],
        VALUE_TYPE_FIELD_NAME: ["$M", "Mshares"],
        BUYBACK_TYPE_FIELD_NAME: ["New", "Additional"],
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04',
                                                         '2014-01-09'])
    }),
    pd.DataFrame(
        columns=[VALUE_FIELD_NAME,
                 VALUE_TYPE_FIELD_NAME,
                 BUYBACK_TYPE_FIELD_NAME,
                 BUYBACK_ANNOUNCEMENT_FIELD_NAME,
                 TS_FIELD_NAME],
        dtype='datetime64[ns]'
    ),
]


def get_expected_previous_values(zip_date_index_with_vals,
                                 dates,
                                 vals_for_date_intervals):
    return pd.DataFrame({
        0: get_values_for_date_ranges(zip_date_index_with_vals,
                                      vals_for_date_intervals,
                                      date_intervals,
                                      dates),
        1: zip_date_index_with_vals(dates, ['NaN'] * len(dates)),
    }, index=dates)


class BuybackAuthLoaderTestCase(WithPipelineEventDataLoader, ZiplineTestCase):
    """
    Test for cash buyback authorizations dataset.
    """
    pipeline_columns = {
        PREVIOUS_VALUE:
            BuybackAuthorizations.previous_value.latest,
        PREVIOUS_BUYBACK_ANNOUNCEMENT:
            BuybackAuthorizations.previous_date.latest,
        PREVIOUS_VALUE_TYPE:
            BuybackAuthorizations.previous_value_type.latest,
        PREVIOUS_BUYBACK_TYPE:
            BuybackAuthorizations.previous_buyback_type.latest,
        DAYS_SINCE_PREV:
            BusinessDaysSinceBuybackAuth(),
    }

    @classmethod
    def get_sids(cls):
        return range(2)

    @classmethod
    def get_dataset(cls):
        return {sid: frame
                for sid, frame
                in enumerate(buyback_authorizations_cases)}

    loader_type = BuybackAuthorizationsLoader

    def setup(self, dates):
        cols = {
            PREVIOUS_VALUE: self.get_sids_to_frames(zip_with_floats,
                                                    [['NaN', 1, 15]],
                                                    date_intervals,
                                                    dates),
            PREVIOUS_BUYBACK_ANNOUNCEMENT: self.get_sids_to_frames(
                zip_with_dates,
                [['NaT', '2014-01-04', '2014-01-09']],
                date_intervals,
                dates),
            PREVIOUS_VALUE_TYPE: self.get_sids_to_frames(
                zip_with_strs, [["", "$M", "Mshares"]], date_intervals, dates
            ),
            PREVIOUS_BUYBACK_TYPE: self.get_sids_to_frames(
                zip_with_strs, [["", "New", "Additional"]], date_intervals,
                dates
            )
        }
        cols[DAYS_SINCE_PREV] = self._compute_busday_offsets(
            cols[PREVIOUS_BUYBACK_ANNOUNCEMENT]
        )
        return cols


class BlazeBuybackAuthLoaderTestCase(BuybackAuthLoaderTestCase):
    """ Test case for loading via blaze.
    """
    loader_type = BlazeBuybackAuthorizationsLoader

    def pipeline_event_loader_args(self, dates):
        _, mapping = super(
            BlazeBuybackAuthLoaderTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return (bz.data(pd.concat(
            pd.DataFrame({
                BUYBACK_ANNOUNCEMENT_FIELD_NAME:
                    frame[BUYBACK_ANNOUNCEMENT_FIELD_NAME],
                VALUE_FIELD_NAME:
                    frame[VALUE_FIELD_NAME],
                VALUE_TYPE_FIELD_NAME:
                    frame[VALUE_TYPE_FIELD_NAME],
                BUYBACK_TYPE_FIELD_NAME:
                    frame[BUYBACK_TYPE_FIELD_NAME],
                TS_FIELD_NAME:
                    frame[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
            })
            for sid, frame in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeBuybackAuthLoaderNotInteractiveTestCase(
        BlazeBuybackAuthLoaderTestCase):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """
    def pipeline_event_loader_args(self, dates):
        (bound_expr,) = super(
            BlazeBuybackAuthLoaderNotInteractiveTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})
