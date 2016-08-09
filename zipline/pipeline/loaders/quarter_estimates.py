import numpy as np
import pandas as pd
from six import viewvalues
from toolz import groupby
from zipline.pipeline.common import (
    EVENT_DATE_FIELD_NAME,
    FISCAL_QUARTER_FIELD_NAME,
    FISCAL_YEAR_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.loaders.frame import DataFrameLoader

import line_profiler
from zipline.pipeline.loaders.utils import choose_rows_by_indexer

PREVIOUS_FISCAL_QUARTER = 'previous_fiscal_quarter'

PREVIOUS_FISCAL_YEAR = 'previous_fiscal_year'

NEXT_FISCAL_QUARTER = 'next_fiscal_quarter'

NEXT_FISCAL_YEAR = 'next_fiscal_year'

FISCAL_QUARTER = 'fiscal_quarter'

FISCAL_YEAR = 'fiscal_year'

ALL_DATES = 'dates'

prof = line_profiler.LineProfiler()


#@profile
def calc_forward_shift(yrs, qtrs, num_qtrs_shift):
    """
    Calculate the number of years to shift forward and the new quarter in the
    shifted year.

    Parameters
    ----------
    qtr : int
        The starting quarter.
    num_qtr_shift : int
        The number of quarters to shift forward.
    yr : int
        The starting year.

    Returns
    -------
    s : pd.Series
        A series containins the new year and quarter.
    """

    result_qtrs = (qtrs + num_qtrs_shift) % 4
    result_years = yrs + (qtrs + num_qtrs_shift) // 4
    to_adjust = result_qtrs[result_qtrs == 0].index
    result_years.iloc[to_adjust] -= 1
    result_qtrs.iloc[to_adjust] = 4
    return result_years, result_qtrs


#@profile
def calc_backward_shift(yrs, qtrs, num_qtrs_shift):
    """
    Calculate the number of years to shift backward and the new quarter in the
    shifted year.

    Parameters
    ----------
    qtr : int
        The starting quarter.
    num_qtr_shift : int
        The number of quarters to shift backward.
    yr : int
        The starting year.

    Returns
    -------
    s : pd.Series
        A series containins the new year and quarter.
    """
    result_qtrs = 4 - (num_qtrs_shift - qtrs) % 4
    # Must subtract 1 year since we go backwards at least `qtr` number of
    # quarters
    result_years = yrs - (num_qtrs_shift - qtrs) // 4 - 1
    no_yr_boundary_crossed = qtrs[qtrs > num_qtrs_shift].index
    result_years.iloc[no_yr_boundary_crossed] = yrs.iloc[no_yr_boundary_crossed]
    result_qtrs.iloc[no_yr_boundary_crossed] = qtrs.iloc[no_yr_boundary_crossed] - num_qtrs_shift
    return result_years, result_qtrs


def required_event_fields(columns):
    """
    Compute the set of resource columns required to serve
    ``next_value_columns`` and ``previous_value_columns``.
    """
    # These metadata columns are used to align event indexers.
    return {
        TS_FIELD_NAME,
        SID_FIELD_NAME,
        EVENT_DATE_FIELD_NAME,
        FISCAL_QUARTER_FIELD_NAME,
        FISCAL_YEAR_FIELD_NAME
    }.union(
        # We also expect any of the field names that our loadable columns
        # are mapped to.
        viewvalues(columns),
    )


def validate_column_specs(events, columns):
    """
    Verify that the columns of ``events`` can be used by an EventsLoader to
    serve the BoundColumns described by ``next_value_columns`` and
    ``previous_value_columns``.
    """
    required = required_event_fields(columns)
    received = set(events.columns)
    missing = required - received
    if missing:
        raise ValueError(
            "EventsLoader missing required columns {missing}.\n"
            "Got Columns: {received}\n"
            "Expected Columns: {required}".format(
                missing=sorted(missing),
                received=sorted(received),
                required=sorted(required),
            )
        )


class QuarterEstimatesLoader(PipelineLoader):
    def __init__(self,
                 estimates,
                 base_column_name_map):
        validate_column_specs(
            estimates,
            base_column_name_map
        )

        self.estimates = estimates[
            estimates[EVENT_DATE_FIELD_NAME].notnull() &
            estimates[FISCAL_QUARTER_FIELD_NAME].notnull() &
            estimates[FISCAL_YEAR_FIELD_NAME].notnull()
        ]

        self.base_column_name_map = base_column_name_map

    def load_quarters(self, num_quarters, dates_sids, final_releases_per_qtr):
        pass

    #@profile
    def load_adjusted_array(self, columns, dates, assets, mask):
        groups = groupby(lambda x: x.dataset.num_quarters, columns)
        out = {}
        date_values = pd.DataFrame({'dates': dates})
        date_values['key'] = 1
        self.estimates['key'] = 1
        merged = pd.merge(date_values, self.estimates, on='key')
        asset_df = pd.DataFrame({SID_FIELD_NAME: assets})
        asset_df['key'] = 1
        dates_sids = pd.merge(date_values, asset_df, on='key')
        merged.drop('key', axis=1, inplace=True)
        dates_sids.drop('key', axis=1, inplace=True)
        for num_quarters in groups:
            name_map = {c: self.base_column_name_map[getattr(c.dataset.__base__, c.name)] for c in columns}

            columns = groups[num_quarters]
            # First, group by sid, fiscal year, and fiscal quarter and only
            # keep the last estimate made.
            final_releases_per_qtr = merged[merged[TS_FIELD_NAME] <=
                                            merged.dates].sort(
                ['dates', TS_FIELD_NAME]
            ).groupby(
                ['dates', SID_FIELD_NAME, FISCAL_YEAR, FISCAL_QUARTER]
            ).last()
            final_releases_per_qtr = final_releases_per_qtr.reset_index()

            result = self.load_quarters(num_quarters,
                                        dates_sids,
                                        final_releases_per_qtr)

            for c in columns:
                column_name = name_map[c]
                # Need to pass a DataFrame that has dates as the index and
                # all sids as columns with column values being the value in
                # 'result' for column c
                loader = DataFrameLoader(
                    c,
                    result.pivot(index='dates',
                                 columns=SID_FIELD_NAME,
                                 values=column_name),
                    adjustments=None
                )
                out[c] = loader.load_adjusted_array([c], dates, assets, mask)[c]
        return out


class NextQuartersEstimatesLoader(QuarterEstimatesLoader):

    #@profile
    def load_quarters(self, num_quarters, dates_sids, final_releases_per_qtr):
        # Filter for releases that are after each simulation date.
        eligible_next_releases = final_releases_per_qtr[
            final_releases_per_qtr[EVENT_DATE_FIELD_NAME] >=
            final_releases_per_qtr['dates']
        ]

        eligible_next_releases.sort(EVENT_DATE_FIELD_NAME)
        # For each sid, get the upcoming release/year/quarter.
        next_releases = eligible_next_releases.groupby(
            ['dates', SID_FIELD_NAME]
        ).nth(0).reset_index()  # We use nth here to avoid forward filling
        # NaNs, which `first()` will do.
        next_releases = next_releases.rename(
            columns={FISCAL_YEAR: NEXT_FISCAL_YEAR,
                     FISCAL_QUARTER: NEXT_FISCAL_QUARTER}
        )
        # `next_qtr` is already the next quarter over,
        # so we should offest `num_shifts` by 1.
        (next_releases[FISCAL_YEAR],
         next_releases[FISCAL_QUARTER]) = calc_forward_shift(
            next_releases[NEXT_FISCAL_YEAR],
            next_releases[NEXT_FISCAL_QUARTER], (num_quarters - 1)
        )
        # Merge to get the rows we care about for each date
        result = dates_sids.merge(next_releases,
                                  on=(['dates', SID_FIELD_NAME]),
                                  how='left')
        return result


class PreviousQuartersEstimatesLoader(QuarterEstimatesLoader):
    def __init__(self,
                 estimates,
                 columns):
        super(PreviousQuartersEstimatesLoader, self).__init__(estimates, columns)

    #@profile
    def load_quarters(self, num_quarters, dates_sids, final_releases_per_qtr):
        # Filter for releases that are before each simulation date.
        eligible_previous_releases = final_releases_per_qtr[
            final_releases_per_qtr[EVENT_DATE_FIELD_NAME] <=
            final_releases_per_qtr['dates']
        ]

        eligible_previous_releases.sort(EVENT_DATE_FIELD_NAME)
        # For each sid, get the latest release we knew about prior to
        # each simulation date.
        previous_releases = eligible_previous_releases.groupby(
            ['dates', SID_FIELD_NAME]
        ).nth(-1).reset_index()  # We use nth here to avoid forward filling
        # NaNs, which `last()` will do.

        previous_releases = previous_releases.rename(columns={
            FISCAL_YEAR: PREVIOUS_FISCAL_YEAR,
            FISCAL_QUARTER: PREVIOUS_FISCAL_QUARTER
        })

        (previous_releases[FISCAL_YEAR],
         previous_releases[FISCAL_QUARTER]) = \
            calc_backward_shift(
            previous_releases[PREVIOUS_FISCAL_YEAR], previous_releases[
                    PREVIOUS_FISCAL_QUARTER], (num_quarters - 1)
        )
        # Merge to get the rows we care about for each date
        result = dates_sids.merge(previous_releases,
                                  on=(['dates', SID_FIELD_NAME]), how='left')
        return result

