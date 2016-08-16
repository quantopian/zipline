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

ALL_DATES = 'dates'
FISCAL_QUARTER = 'fiscal_quarter'
FISCAL_YEAR = 'fiscal_year'
NEXT_FISCAL_QUARTER = 'next_fiscal_quarter'
NEXT_FISCAL_YEAR = 'next_fiscal_year'
PREVIOUS_FISCAL_QUARTER = 'previous_fiscal_quarter'
PREVIOUS_FISCAL_YEAR = 'previous_fiscal_year'
SIMULTATION_DATES = 'dates'


def calc_forward_shift(yrs, qtrs, num_qtrs_shift):
    """
    Calculate the new years and quarters based on on shifting the specified
    number of quarters forward.

    Parameters
    ----------
    yrs : np.Series
        The starting years.
    qtrs : np.Series
        The starting quarters.
    num_qtrs_shift : int
        The number of quarters to shift forward.


    Returns
    -------
    result_years : pd.Series
        A series contains the new years.
    result_qtrs : pd.Series
        A series that contains the new quarters.
    """

    if num_qtrs_shift < 0:
        raise AssertionError("Must pass a number of quarters >= 0")
    result_qtrs = (qtrs + num_qtrs_shift) % 4
    result_years = yrs + (qtrs + num_qtrs_shift) // 4
    # When we get 0, this actually means we're in Q1 of the previous year,
    # so we need to adjust.
    to_adjust = result_qtrs[result_qtrs == 0].index
    result_years.iloc[to_adjust] -= 1
    result_qtrs.iloc[to_adjust] = 4
    return result_years, result_qtrs


def calc_backward_shift(yrs, qtrs, num_qtrs_shift):
    """
    Calculate the new years and quarters based shifting the specified number
    of quarters backwards.

    Parameters
    ----------
    yrs : np.Series
        The starting years.
    qtrs : np.Series
        The starting quarters.
    num_qtrs_shift : int
        The number of quarters to shift backward.


    Returns
    -------
    result_years : pd.Series
        A series contains the new years.
    result_qtrs : pd.Series
        A series that contains the new quarters.
    """

    if num_qtrs_shift < 0:
        raise AssertionError("Must pass a number of quarters >= 0")
    result_qtrs = 4 - (num_qtrs_shift - qtrs) % 4
    # Subtract 1 year since we go backwards at least `qtrs` number of quarters.
    result_years = yrs - (num_qtrs_shift - qtrs) // 4 - 1
    # Find cases where we aren't shifting enough quarters backwards to cross
    # a year boundary and correct for these.
    no_yr_boundary_crossed = qtrs[qtrs > num_qtrs_shift].index
    # Set the year back to the original.
    result_years.iloc[no_yr_boundary_crossed] = yrs.iloc[
        no_yr_boundary_crossed
    ]
    result_qtrs.iloc[no_yr_boundary_crossed] = qtrs.iloc[
                                                   no_yr_boundary_crossed
                                               ] - num_qtrs_shift
    return result_years, result_qtrs


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


def cross_product(df1, df2):
    df1['key'] = 1
    df2['key'] = 1
    merged = pd.merge(df1, df2, on='key')
    return merged.drop('key', axis=1)


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

    def load_adjusted_array(self, columns, dates, assets, mask):
        # TODO: how can we enforce that datasets have the num_quarters
        # attribute, given that they're created dynamically?
        groups = groupby(lambda x: x.dataset.num_quarters, columns)
        out = {}
        date_values = pd.DataFrame({SIMULTATION_DATES: dates})
        estimates_all_dates = cross_product(date_values, self.estimates)
        asset_df = pd.DataFrame({SID_FIELD_NAME: assets})
        dates_sids = cross_product(date_values, asset_df)
        for num_quarters in groups:
            name_map = {c:
                        self.base_column_name_map[
                            getattr(c.dataset.__base__, c.name)
                        ] for c in columns}

            columns = groups[num_quarters]
            # First, determine which estimates we would have known about on
            # each date. Then, Sort by timestamp and group to find the latest
            # estimate for each quarter.
            final_releases_per_qtr = estimates_all_dates[
                estimates_all_dates[TS_FIELD_NAME] <=
                estimates_all_dates.dates
            ].sort([TS_FIELD_NAME]).groupby(
                [SIMULTATION_DATES,
                 SID_FIELD_NAME,
                 FISCAL_YEAR,
                 FISCAL_QUARTER]
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
            columns={FISCAL_YEAR: NEXT_FISCAL_YEAR,
                     FISCAL_QUARTER: NEXT_FISCAL_QUARTER}
        )
        # The next fiscal quarter is already our starting point,
        # so we should offset `num_quarters` by 1.
        (next_releases[FISCAL_YEAR],
         next_releases[FISCAL_QUARTER]) = calc_forward_shift(
            next_releases[NEXT_FISCAL_YEAR],
            next_releases[NEXT_FISCAL_QUARTER], (num_quarters - 1)
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
            FISCAL_YEAR: PREVIOUS_FISCAL_YEAR,
            FISCAL_QUARTER: PREVIOUS_FISCAL_QUARTER
        })
        # The previous fiscal quarter is already our starting point,
        # so we should offset `num_quarters` by 1.
        (previous_releases[FISCAL_YEAR],
         previous_releases[FISCAL_QUARTER]) = calc_backward_shift(
            previous_releases[PREVIOUS_FISCAL_YEAR],
            previous_releases[PREVIOUS_FISCAL_QUARTER],
            (num_quarters - 1)
        )
        # Do a left merge to get values for each date.
        result = dates_sids.merge(previous_releases,
                                  on=([SIMULTATION_DATES,
                                       SID_FIELD_NAME]),
                                  how='left')
        return result
