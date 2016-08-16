from abc import abstractmethod
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
from zipline.utils.pandas_utils import cross_product

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


def shift_quarters(by, years, quarters):
    return split_normalized_quarters(normalize_quarters(years, quarters) + by)


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
    def load_quarters(self, num_quarters, dates_sids, final_releases_per_qtr):
        pass

    def load_adjusted_array(self, columns, dates, assets, mask):
        # TODO: how can we enforce that datasets have the num_quarters
        # attribute, given that they're created dynamically?
        groups = groupby(lambda x: x.dataset.num_quarters, columns)
        groups_columns = dict(groups)
        if (pd.Series(groups_columns.keys()) < 0).any():
            raise ValueError("Must pass a number of quarters >= 0")
        out = {}
        date_values = pd.DataFrame({SIMULTATION_DATES: dates})
        # dates column must be of type datetime64[ns] in order for subsequent
        # comparisons to work correctly.
        date_values[SIMULTATION_DATES] = date_values[
            SIMULTATION_DATES
        ].astype('datetime64[ns]')
        estimates_all_dates = cross_product(date_values, self.estimates)
        asset_df = pd.DataFrame({SID_FIELD_NAME: assets})
        dates_sids = cross_product(date_values, asset_df)
        for num_quarters, columns in groups_columns.iteritems():
            name_map = {c:
                        self.base_column_name_map[
                            getattr(c.dataset.__base__, c.name)
                        ] for c in columns}

            # First, determine which estimates we would have known about on
            # each date. Then, Sort by timestamp and group to find the latest
            # estimate for each quarter.
            final_releases_per_qtr = estimates_all_dates[
                estimates_all_dates[TS_FIELD_NAME] <=
                estimates_all_dates.dates
            ].sort([TS_FIELD_NAME]).groupby(
                [SIMULTATION_DATES,
                 SID_FIELD_NAME,
                 FISCAL_YEAR_FIELD_NAME,
                 FISCAL_QUARTER_FIELD_NAME]
            ).nth(-1).reset_index()

            result = self.load_quarters(num_quarters,
                                        dates_sids,
                                        final_releases_per_qtr)

            for c in columns:
                column_name = name_map[c]
                # Pivot to get a DataFrame with dates as the index and
                # sids as the columns.
                loader = DataFrameLoader(
                    c,
                    result.pivot(index=SIMULTATION_DATES,
                                 columns=SID_FIELD_NAME,
                                 values=column_name),
                    adjustments=None
                )
                out[c] = loader.load_adjusted_array([c],
                                                    dates,
                                                    assets,
                                                    mask)[c]
        return out


class NextQuartersEstimatesLoader(QuarterEstimatesLoader):

    def load_quarters(self, num_quarters, dates_sids, final_releases_per_qtr):
        # Filter for releases that are on or after each simulation date.
        eligible_next_releases = final_releases_per_qtr[
            final_releases_per_qtr[EVENT_DATE_FIELD_NAME] >=
            final_releases_per_qtr[SIMULTATION_DATES]
        ]
        # For each sid, get the upcoming release.
        eligible_next_releases.sort(EVENT_DATE_FIELD_NAME)
        next_releases = eligible_next_releases.groupby(
            [SIMULTATION_DATES, SID_FIELD_NAME]
        ).nth(0).reset_index()  # We use nth here to avoid forward filling
        # NaNs, which `first()` will do.
        next_releases = next_releases.rename(
            columns={FISCAL_YEAR_FIELD_NAME: NEXT_FISCAL_YEAR,
                     FISCAL_QUARTER_FIELD_NAME: NEXT_FISCAL_QUARTER}
        )
        # The next fiscal quarter is already our starting point,
        # so we should offset `num_quarters` by 1.
        (next_releases[FISCAL_YEAR_FIELD_NAME],
         next_releases[FISCAL_QUARTER_FIELD_NAME]) = shift_quarters(
            (num_quarters - 1),
            next_releases[NEXT_FISCAL_YEAR],
            next_releases[NEXT_FISCAL_QUARTER],
        )
        # Do a left merge to get values for each date.
        result = dates_sids.merge(next_releases,
                                  on=([SIMULTATION_DATES, SID_FIELD_NAME]),
                                  how='left')
        return result


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
