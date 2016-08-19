"""
Tests for the reference loader for Buyback Authorizations.
"""
import blaze as bz
from blaze.compute.core import swap_resources_into_scope
import pandas as pd
from six import iteritems

from zipline.pipeline.common import(
    DAYS_SINCE_PREV,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.data import BuybackAuthorizations
from zipline.pipeline.factors.events import BusinessDaysSinceBuybackAuth
from zipline.pipeline.loaders.buyback_auth import (
    BUYBACK_AMOUNT_FIELD_NAME,
    BUYBACK_ANNOUNCEMENT_FIELD_NAME,
    BUYBACK_TYPE_FIELD_NAME,
    BUYBACK_UNIT_FIELD_NAME,
    BuybackAuthorizationsLoader,
)
from zipline.pipeline.loaders.blaze import BlazeBuybackAuthorizationsLoader
from zipline.pipeline.loaders.utils import (
    zip_with_dates,
    zip_with_floats,
    zip_with_strs
)
from zipline.testing.fixtures import (
    WithPipelineEventDataLoader, ZiplineTestCase
)

PREVIOUS_BUYBACK_AMOUNT = 'previous_value'
PREVIOUS_BUYBACK_ANNOUNCEMENT = 'previous_buyback_announcement'
PREVIOUS_BUYBACK_CASH = 'previous_buyback_cash'
PREVIOUS_BUYBACK_SHARE_COUNT = 'previous_buyback_share_count'
PREVIOUS_BUYBACK_TYPE = 'previous_buyback_type'
PREVIOUS_BUYBACK_UNIT = 'previous_buyback_unit'

date_intervals = [
    [['2014-01-01', '2014-01-04'], ['2014-01-05', '2014-01-09'],
     ['2014-01-10', '2014-01-31']]
]

buyback_authorizations_cases = [
    pd.DataFrame({
        BUYBACK_AMOUNT_FIELD_NAME: [1, 15],
        BUYBACK_UNIT_FIELD_NAME: ["$M", "Mshares"],
        BUYBACK_TYPE_FIELD_NAME: ["New", "Additional"],
        TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
        BUYBACK_ANNOUNCEMENT_FIELD_NAME: pd.to_datetime(['2014-01-04',
                                                         '2014-01-09'])
    }),
    pd.DataFrame(
        columns=[BUYBACK_AMOUNT_FIELD_NAME,
                 BUYBACK_UNIT_FIELD_NAME,
                 BUYBACK_TYPE_FIELD_NAME,
                 BUYBACK_ANNOUNCEMENT_FIELD_NAME,
                 TS_FIELD_NAME],
        dtype='datetime64[ns]'
    ),
]


class BuybackAuthLoaderTestCase(WithPipelineEventDataLoader, ZiplineTestCase):
    """
    Test for cash buyback authorizations dataset.
    """
    pipeline_columns = {
        PREVIOUS_BUYBACK_AMOUNT:
            BuybackAuthorizations.previous_amount.latest,
        PREVIOUS_BUYBACK_ANNOUNCEMENT:
            BuybackAuthorizations.previous_date.latest,
        PREVIOUS_BUYBACK_UNIT:
            BuybackAuthorizations.previous_unit.latest,
        PREVIOUS_BUYBACK_TYPE:
            BuybackAuthorizations.previous_type.latest,
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
            PREVIOUS_BUYBACK_AMOUNT: self.get_sids_to_frames(zip_with_floats,
                                                             [['NaN', 1, 15]],
                                                             date_intervals,
                                                             dates,
                                                             'float',
                                                             'NaN'),
            PREVIOUS_BUYBACK_ANNOUNCEMENT: self.get_sids_to_frames(
                zip_with_dates,
                [['NaT', '2014-01-04', '2014-01-09']],
                date_intervals,
                dates,
                'datetime64[ns]',
                'NaN'
            ),
            PREVIOUS_BUYBACK_UNIT: self.get_sids_to_frames(
                zip_with_strs,
                [[None, "$M", "Mshares"]],
                date_intervals,
                dates,
                'category',
                None
            ),
            PREVIOUS_BUYBACK_TYPE: self.get_sids_to_frames(
                zip_with_strs,
                [[None, "New", "Additional"]],
                date_intervals,
                dates,
                'category',
                None
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
                BUYBACK_AMOUNT_FIELD_NAME:
                    frame[BUYBACK_AMOUNT_FIELD_NAME],
                BUYBACK_UNIT_FIELD_NAME:
                    frame[BUYBACK_UNIT_FIELD_NAME],
                BUYBACK_TYPE_FIELD_NAME:
                    frame[BUYBACK_TYPE_FIELD_NAME],
                TS_FIELD_NAME:
                    frame[TS_FIELD_NAME],
                SID_FIELD_NAME: sid,
            })
            for sid, frame in iteritems(mapping)
        ).reset_index(drop=True)),)


class BlazeBuybackAuthLoaderNotInteractiveTestCase(
        BlazeBuybackAuthLoaderTestCase
):
    """Test case for passing a non-interactive symbol and a dict of resources.
    """
    def pipeline_event_loader_args(self, dates):
        (bound_expr,) = super(
            BlazeBuybackAuthLoaderNotInteractiveTestCase,
            self,
        ).pipeline_event_loader_args(dates)
        return swap_resources_into_scope(bound_expr, {})
