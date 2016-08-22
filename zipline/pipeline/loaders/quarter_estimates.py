from abc import abstractmethod
from collections import defaultdict
from functools import partial
import numpy as np
from numpy.ma import asarray
import pandas as pd
from six import viewvalues
from toolz import groupby, curry
from zipline.lib.adjusted_array import AdjustedArray
from zipline.lib.adjustment import (Datetime641DArrayOverwrite,
                                    Float641DArrayOverwrite)

from zipline.pipeline.common import (
    EVENT_DATE_FIELD_NAME,
    FISCAL_QUARTER_FIELD_NAME,
    FISCAL_YEAR_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.utils.numpy_utils import datetime64ns_dtype
from zipline.utils.pandas_utils import cross_product
from zipline.pipeline.loaders.utils import last_in_date_group, ffill_across_cols

NEXT_FISCAL_QUARTER = 'next_fiscal_quarter'
NEXT_FISCAL_YEAR = 'next_fiscal_year'
PREVIOUS_FISCAL_QUARTER = 'previous_fiscal_quarter'
PREVIOUS_FISCAL_YEAR = 'previous_fiscal_year'
SIMULTATION_DATES = 'dates'


def normalize_quarters(years, quarters):
    return years * 4 + quarters - 1


def split_normalized_quarters(normalized_quarters):
    years = normalized_quarters // 4
    quarters = normalized_quarters % 4
    return years, quarters + 1


def required_estimates_fields(columns):
    """
    Compute the set of resource columns required to serve
    `columns`.
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
    Verify that the columns of ``events`` can be used by a
    QuarterEstimatesLoader to serve the BoundColumns described by
    `columns`.
    """
    required = required_estimates_fields(columns)
    received = set(events.columns)
    missing = required - received
    if missing:
        raise ValueError(
            "QuarterEstimatesLoader missing required columns {missing}.\n"
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

    @abstractmethod
    def load_quarters(self, num_quarters, last, dates):
        pass

    def get_adjustments(self, result, col_result, last,
                        column_name,
                        column, mask,
                        assets):
        adjustments = defaultdict(list)
        if column.dtype == datetime64ns_dtype:
            overwrite = Datetime641DArrayOverwrite
        else:
            overwrite = Float641DArrayOverwrite
        for sid_idx, sid in enumerate(assets):
            sid_result = result[result.index.get_level_values(
                SID_FIELD_NAME
            ) == sid]
            sid_result = sid_result.reset_index(
                level='shifted_normalized_quarters'
            )  # Remove qtrs from index to find shifts
            # Figure out where we think quarters are changing.
            qtr_shifts = sid_result[
                sid_result['shifted_normalized_quarters'] !=
                sid_result['shifted_normalized_quarters'].shift(1)
            ]
            # Iterate backwards. No adjustment for 1st quarter.
            for row_indexer in list(reversed(qtr_shifts.index))[:-1]:
                # We want to write the values for this row's quarter over
                # everything that comes before this quarter when we are at
                # the date before this quarter starts.
                qtr_start_idx = last.index.get_loc(row_indexer[0])
                quarter = qtr_shifts.loc[row_indexer][
                    'shifted_normalized_quarters'
                ]
                adjustments[qtr_start_idx] = \
                    [overwrite(0,
                               qtr_start_idx - 1,  # get index date
                               sid_idx,
                               sid_idx,
                               last[column_name, quarter,
                                    sid][:qtr_start_idx].values)
                     ]

        return AdjustedArray(
                col_result.values.astype(column.dtype),
                mask,
                dict(adjustments),
                column.missing_value,
            )

    def load_adjusted_array(self, columns, dates, assets, mask):
        # TODO: how can we enforce that datasets have the num_quarters
        # attribute, given that they're created dynamically?
        groups = groupby(lambda x: x.dataset.num_quarters, columns)
        groups_columns = dict(groups)
        if (pd.Series(groups_columns) < 0).any():
            raise ValueError("Must pass a number of quarters >= 0")
        out = {}
        date_values = pd.DataFrame({SIMULTATION_DATES: dates})
        # dates column must be of type datetime64[ns] in order for subsequent
        # comparisons to work correctly.
        date_values[SIMULTATION_DATES] = date_values[
            SIMULTATION_DATES
        ].astype('datetime64[ns]')
        self.estimates['normalized_quarters'] = normalize_quarters(
            self.estimates[FISCAL_YEAR_FIELD_NAME],
            self.estimates[FISCAL_QUARTER_FIELD_NAME],
        ).astype(float)
        for num_quarters, columns in groups_columns.iteritems():
            name_map = {c:
                        self.base_column_name_map[
                            getattr(c.dataset.__base__, c.name)
                        ] for c in columns}
            # Determine the last piece of information we know for each column
            # on each date in the index.
            last = last_in_date_group(self.estimates, True, dates,
                                      assets,
                                      extra_groupers=[
                                          'normalized_quarters'])
            # Forward fill values for each quarter.
            ffill_across_cols(last, columns)
            stacked = last.stack(1).stack(1)

            result = self.load_quarters(num_quarters, stacked)

            for c in columns:
                column_name = name_map[c]
                col_result = result[
                    column_name
                ].reset_index(1, drop=True).unstack(1).reindex(dates)
                adjusted_array = self.get_adjustments(result,
                                                      col_result,
                                                      last,
                                                      column_name,
                                                      c,
                                                      mask,
                                                      assets)
                out[c] = adjusted_array
        return out


class NextQuartersEstimatesLoader(QuarterEstimatesLoader):

    def load_quarters(self, num_quarters, stacked):
        # Filter for releases that are on or after each simulation date and
        # determine the next quarter by picking out the upcoming release for
        # each date in the index.
        stacked = stacked.sort(EVENT_DATE_FIELD_NAME)
        next_releases = stacked.loc[
            stacked[EVENT_DATE_FIELD_NAME] >= stacked.index.get_level_values(
                0
            )].groupby(level=[0, 2]).nth(0)
        next_releases[
            'shifted_normalized_quarters'
        ] = next_releases.index.get_level_values(
            'normalized_quarters'
        ) + (num_quarters - 1)
        next_releases = next_releases.set_index([
            next_releases.index.get_level_values(0),  # dates
            'shifted_normalized_quarters',
            next_releases.index.get_level_values(2)  # sids
        ])
        return stacked.loc[next_releases.index]


class PreviousQuartersEstimatesLoader(QuarterEstimatesLoader):
    def __init__(self,
                 estimates,
                 columns):
        super(PreviousQuartersEstimatesLoader, self).__init__(estimates,
                                                              columns)

    def load_quarters(self, num_quarters, dates_sids, final_releases_per_qtr):
        # Filter for releases that are on or before each simulation date.
        eligible_previous_releases = final_releases_per_qtr[
            final_releases_per_qtr[EVENT_DATE_FIELD_NAME] <=
            final_releases_per_qtr[SIMULTATION_DATES]
        ]
        # For each sid, get the latest release.
        eligible_previous_releases.sort(EVENT_DATE_FIELD_NAME)
        previous_releases = eligible_previous_releases.groupby(
            [SIMULTATION_DATES, SID_FIELD_NAME]
        ).nth(-1).reset_index()  # We use nth here to avoid forward filling
        # NaNs, which `last()` will do.
        previous_releases = previous_releases.rename(columns={
            FISCAL_YEAR_FIELD_NAME: PREVIOUS_FISCAL_YEAR,
            FISCAL_QUARTER_FIELD_NAME: PREVIOUS_FISCAL_QUARTER
        })
        # The previous fiscal quarter is already our starting point,
        # so we should offset `num_quarters` by 1.
        (previous_releases[FISCAL_YEAR_FIELD_NAME],
         previous_releases[FISCAL_QUARTER_FIELD_NAME]) = shift_quarters(
            -(num_quarters - 1),
            previous_releases[PREVIOUS_FISCAL_YEAR],
            previous_releases[PREVIOUS_FISCAL_QUARTER],
        )
        # Do a left merge to get values for each date.
        result = dates_sids.merge(previous_releases,
                                  on=([SIMULTATION_DATES,
                                       SID_FIELD_NAME]),
                                  how='left')
        return result
