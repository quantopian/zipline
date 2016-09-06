from abc import abstractmethod
import numpy as np
from six import viewvalues
from toolz import groupby
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
from zipline.utils.numpy_utils import datetime64ns_dtype
from zipline.pipeline.loaders.utils import (
    ffill_across_cols,
    last_in_date_group
)

NORMALIZED_QUARTERS = 'normalized_quarters'

SHIFTED_NORMALIZED_QTRS = 'shifted_normalized_quarters'

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
                 name_map):
        validate_column_specs(
            estimates,
            name_map
        )

        self.estimates = estimates[
            estimates[EVENT_DATE_FIELD_NAME].notnull() &
            estimates[FISCAL_QUARTER_FIELD_NAME].notnull() &
            estimates[FISCAL_YEAR_FIELD_NAME].notnull()
        ]
        self.estimates[NORMALIZED_QUARTERS] = normalize_quarters(
            self.estimates[FISCAL_YEAR_FIELD_NAME],
            self.estimates[FISCAL_QUARTER_FIELD_NAME],
        )

        self.name_map = name_map

    @abstractmethod
    def load_quarters(self, num_quarters, last, dates):
        raise NotImplementedError('load_quarters')

    def get_requested_data_for_col(self, stacked_last_per_qtr, idx, dates):
        """
        Selects the requested data for each date.

        Parameters
        ----------
        stacked_last_per_qtr : pd.DataFrame
            The latest estimate known per sid per date per quarter with the
            dates, normalized quarter, and sid as the index.
        idx : pd.MultiIndex
            The index of the row of the requested quarter from each date for
            each sid.
        dates : pd.DatetimeIndex
            The calendar dates for which estimates data is requested.

        Returns
        --------
        requested_qtr_data : pd.DataFrame
            The DataFrame with final values for the requested quarter for all
            columns; `dates` are the index and columns are a MultiIndex with
            sids at the top level and the dataset columns on the bottom.
        """
        requested_qtr_data = stacked_last_per_qtr.loc[idx]
        # We no longer need this in the index, but we do need it as a column
        # to calculate adjustments.
        requested_qtr_data = requested_qtr_data.reset_index(
            SHIFTED_NORMALIZED_QTRS
        )
        (requested_qtr_data[FISCAL_YEAR_FIELD_NAME],
         requested_qtr_data[FISCAL_QUARTER_FIELD_NAME]) = \
            split_normalized_quarters(
                requested_qtr_data[SHIFTED_NORMALIZED_QTRS]
            )
        # Move sids into the columns. Once we're left with just dates
        # as the index, we can reindex by all dates so that we have a
        # value for each calendar date.
        requested_qtr_data = requested_qtr_data.unstack(
            SID_FIELD_NAME
        ).reindex(dates)
        return requested_qtr_data

    def get_adjustments(self,
                        zero_qtr_idx,
                        requested_qtr_idx,
                        stacked_last_per_qtr,
                        last_per_qtr,
                        dates,
                        column_name,
                        column,
                        mask,
                        assets):
        """
        Creates an AdjustedArray from the given estimates data for the given
        dates.

        Parameters
        ----------
        zero_qtr_idx : pd.MultiIndex
            The index of the row of the zeroth (immediately next/previous)
            quarter from each date for each sid.
        requested_qtr_idx : pd.MultiIndex
            The index of the row of the requested quarter from each date for
            each sid.
        stacked_last_per_qtr : pd.DataFrame
            The latest estimate known per sid per date per quarter with the
            dates, normalized quarter, and sid as the index.
        last_per_qtr : pd.DataFrame
            The latest estimate known per sid per date per quarter with
            dates as the index and normalized quarter and sid in the columns
            MultiIndex; allows easy access to the timeline of estimates
            across all dates for a sid for a particular quarter.
        dates : pd.DatetimeIndex
            The calendar dates for which estimates data is requested.
        column_name : string
            The name of the column for which the AdjustedArray is being
            computed.
        column : BoundColumn
            The column for which the AdjustedArray is being computed.
        mask :
        assets :

        Returns
        -------
        adjusted_array : AdjustedArray
            The array of data and overwrites for the given column.
        """
        adjustments = {}
        requested_qtr_data = self.get_requested_data_for_col(
            stacked_last_per_qtr, requested_qtr_idx, dates
        )
        zero_qtr_data = stacked_last_per_qtr.loc[zero_qtr_idx]
        # We no longer need this in the index, but we do need it as a column
        # to calculate adjustments.
        zero_qtr_data = zero_qtr_data.reset_index(NORMALIZED_QUARTERS)
        if column.dtype == datetime64ns_dtype:
            overwrite = Datetime641DArrayOverwrite
        else:
            overwrite = Float641DArrayOverwrite
        for sid_idx, sid in enumerate(assets):
            zero_qtr_sid_data = zero_qtr_data[
                zero_qtr_data.index.get_level_values(SID_FIELD_NAME) == sid
            ]
            # Determine where quarters are changing for this sid.
            qtr_shifts = zero_qtr_sid_data[
                zero_qtr_sid_data[NORMALIZED_QUARTERS] !=
                zero_qtr_sid_data[NORMALIZED_QUARTERS].shift(1)
            ]
            # On dates where we don't have any information about quarters,
            # we will get nulls, and each of these will be interpreted as
            # quarter shifts. We need to remove these here.
            qtr_shifts = qtr_shifts[
                qtr_shifts[NORMALIZED_QUARTERS].notnull()
            ]
            # For the given sid, determine which quarters we have estimates
            # for.
            quarters_with_estimates_for_sid = last_per_qtr.xs(
                sid, axis=1, level=SID_FIELD_NAME
            ).groupby(axis=1, level=1).first().columns.values
            for row_indexer in list(qtr_shifts.index):
                # Find the starting index of the quarter that comes right
                # after this row. This isn't the starting index of the
                # requested quarter, but simply the date we cross over into a
                # new quarter.
                next_qtr_start_idx = dates.searchsorted(
                    zero_qtr_data.loc[
                        row_indexer
                    ][EVENT_DATE_FIELD_NAME],
                    side='left'
                    if isinstance(self, PreviousQuartersEstimatesLoader)
                    else 'right'
                )

                # Only add adjustments if the next quarter starts somewhere in
                # our date index for this sid. Our 'next' quarter can never
                # start at index 0; a starting index of 0 means that the next
                # quarter's event date was NaT.
                if 0 < next_qtr_start_idx < len(dates):
                    # Find the quarter being requested in the quarter we're
                    # crossing into.
                    requested_quarter = requested_qtr_data[
                        SHIFTED_NORMALIZED_QTRS
                    ][sid].iloc[next_qtr_start_idx]

                    # If there are estimates for the requested quarter,
                    # overwrite all values going up to the starting index of
                    # that quarter with estimates for that quarter.
                    if requested_quarter in quarters_with_estimates_for_sid:
                        if isinstance(self, NextQuartersEstimatesLoader):
                            adjustments[next_qtr_start_idx] = \
                                [overwrite(
                                    0,
                                    # overwrite thru last qtr
                                    next_qtr_start_idx - 1,
                                    sid_idx,
                                    sid_idx,
                                    last_per_qtr[
                                        column_name,
                                        requested_quarter,
                                        sid
                                    ][0:next_qtr_start_idx].values)]
                        else:
                            self.overwrite_with_null(adjustments, column,
                                                     last_per_qtr,
                                                     next_qtr_start_idx,
                                                     overwrite,
                                                     sid_idx)
                    # There are no estimates for the quarter. Overwrite all
                    # values going up to the starting index of that quarter
                    # with the missing value for this column.
                    else:
                        self.overwrite_with_null(adjustments, column,
                                                 last_per_qtr,
                                                 next_qtr_start_idx, overwrite,
                                                 sid_idx)

        return AdjustedArray(
            requested_qtr_data[column_name].values.astype(column.dtype),
            mask,
            dict(adjustments),
            column.missing_value,
        )


    def overwrite_with_null(self, adjustments, column, last_per_qtr,
                            next_qtr_start_idx, overwrite, sid_idx):
        adjustments[next_qtr_start_idx] = [
            overwrite(
                0,
                next_qtr_start_idx - 1,
                sid_idx,
                sid_idx,
                np.full(
                    len(
                        last_per_qtr.index[:next_qtr_start_idx]
                    ),
                    column.missing_value,
                )
            )
        ]


    def load_adjusted_array(self, columns, dates, assets, mask):
        groups = groupby(lambda x: x.dataset.num_quarters, columns)
        if any(num_qtr < 0 for num_qtr in groups):  # use any
            raise ValueError(
                "Passed invalid number of quarters %s; "
                "must pass a number of quarters >= 0" % ','.join(
                    qtr for qtr in groups if qtr < 0
                )

            )
        out = {}

        for num_quarters, columns in groups.items():
            # Determine the last piece of information we know for each column
            # on each date in the index for each sid and quarter.
            last_per_qtr = last_in_date_group(
                # TODO pass bools by keyword
                self.estimates, True, dates, assets,
                extra_groupers=[NORMALIZED_QUARTERS]
            )

            # Forward fill values for each quarter/sid/dataset column.
            ffill_across_cols(last_per_qtr, columns, self.name_map)
            # Stack quarter and sid into the index.
            stacked_last_per_qtr = last_per_qtr.stack([SID_FIELD_NAME,
                                                       NORMALIZED_QUARTERS])
            # Set date index name for ease of reference
            # TODO: pass bool by keyword
            stacked_last_per_qtr.index.set_names(SIMULTATION_DATES, 0, True)
            # Determine which quarter is next/previous for each date.
            shifted_qtr_data = self.load_quarters(num_quarters,
                                                  stacked_last_per_qtr)
            zero_qtr_idx = shifted_qtr_data.index
            requested_qtr_idx = shifted_qtr_data.set_index([
                shifted_qtr_data.index.get_level_values(
                    SIMULTATION_DATES
                ),
                shifted_qtr_data.index.get_level_values(
                    SID_FIELD_NAME
                ),
                shifted_qtr_data[SHIFTED_NORMALIZED_QTRS]
            ]).index

            for c in columns:
                column_name = self.name_map[c.name]
                adjusted_array = self.get_adjustments(zero_qtr_idx,
                                                      requested_qtr_idx,
                                                      stacked_last_per_qtr,
                                                      last_per_qtr,
                                                      dates,
                                                      column_name,
                                                      c,
                                                      mask,
                                                      assets)
                out[c] = adjusted_array
        return out


class NextQuartersEstimatesLoader(QuarterEstimatesLoader):

    def load_quarters(self, num_quarters, stacked_last_per_qtr):
        # Filter for releases that are on or after each simulation date and
        # determine the next quarter by picking out the upcoming release for
        # each date in the index.
        stacked_last_per_qtr = stacked_last_per_qtr.sort(
            EVENT_DATE_FIELD_NAME
        )
        # We reset the index here because in pandas3, a groupby on the index
        # will set the index to just the items in the groupby, so we will lose
        # the normalized quarters.
        next_releases_per_date = stacked_last_per_qtr.loc[
            stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] >=
            stacked_last_per_qtr.index.get_level_values(SIMULTATION_DATES)
        ].reset_index(NORMALIZED_QUARTERS).groupby(
            level=[SIMULTATION_DATES, SID_FIELD_NAME]
        ).nth(0).set_index(NORMALIZED_QUARTERS, append=True)
        next_releases_per_date[
            SHIFTED_NORMALIZED_QTRS
        ] = next_releases_per_date.index.get_level_values(
            NORMALIZED_QUARTERS
        ) + (num_quarters - 1)
        return next_releases_per_date


class PreviousQuartersEstimatesLoader(QuarterEstimatesLoader):

    def load_quarters(self, num_quarters, stacked_last_per_qtr):
        # Filter for releases that are on or before each simulation date and
        # determine the previous quarter by picking out the upcoming release
        # for each date in the index.
        stacked_last_per_qtr = stacked_last_per_qtr.sort(EVENT_DATE_FIELD_NAME)
        # We reset the index here because in pandas3, a groupby on the index
        # will set the index to just the items in the groupby, so we will lose
        # the normalized quarters.
        previous_releases_per_date = stacked_last_per_qtr.loc[
            stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] <=
            stacked_last_per_qtr.index.get_level_values(
                SIMULTATION_DATES
        )].reset_index(NORMALIZED_QUARTERS).groupby(
            level=[SIMULTATION_DATES, SID_FIELD_NAME]
        ).nth(-1).set_index(NORMALIZED_QUARTERS, append=True)
        previous_releases_per_date[
            SHIFTED_NORMALIZED_QTRS
        ] = previous_releases_per_date.index.get_level_values(
            NORMALIZED_QUARTERS
        ) - (num_quarters - 1)
        return previous_releases_per_date
