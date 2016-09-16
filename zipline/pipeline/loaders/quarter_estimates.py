from collections import defaultdict
from abc import abstractmethod
import numpy as np
import pandas as pd
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


INVALID_NUM_QTRS_MESSAGE = "Passed invalid number of quarters %s; " \
                           "must pass a number of quarters >= 0"
NEXT_FISCAL_QUARTER = 'next_fiscal_quarter'
NEXT_FISCAL_YEAR = 'next_fiscal_year'
NORMALIZED_QUARTERS = 'normalized_quarters'
PREVIOUS_FISCAL_QUARTER = 'previous_fiscal_quarter'
PREVIOUS_FISCAL_YEAR = 'previous_fiscal_year'
SHIFTED_NORMALIZED_QTRS = 'shifted_normalized_quarters'
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
    """
    An abstract pipeline loader for estimates data that can load data a
    variable number of quarters forwards/backwards from calendar dates
    depending on the `num_quarters` attribute of the columns' dataset.

    Parameters
    ----------
    estimates : pd.DataFrame
        The raw estimates data.
        ``estimates`` must contain at least 5 columns:
            sid : int64
                The asset id associated with each estimate.

            event_date : datetime64[ns]
                The date on which the event that the estimate is for will/has
                occurred..

            timestamp : datetime64[ns]
                The date on which we learned about the estimate.

            fiscal_quarter : int64
                The quarter during which the event has/will occur.

            fiscal_year : int64
                The year during which the event has/will occur.

    name_map : dict[str -> str]
        A map of names of BoundColumns that this loader will load to the
        names of the corresponding columns in `events`.
    """
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
    def get_zeroth_quarter_idx(self, num_quarters, last, dates):
        raise NotImplementedError('get_zeroth_quarter_idx')

    @abstractmethod
    def get_shifted_qtrs(self, zero_qtrs, num_quarters):
        raise NotImplementedError('get_shifted_qtrs')

    def get_requested_quarter_data(self, stacked_last_per_qtr, idx, dates):
        """
        Selects the requested data for each date.

        Parameters
        ----------
        stacked_last_per_qtr : pd.DataFrame
            The latest estimate known  with the dates, normalized quarter, and
            sid as the index.
        idx : pd.MultiIndex
            The index of the row of the requested quarter from each date for
            each sid.
        dates : pd.DatetimeIndex
            The calendar dates for which estimates data is requested.

        Returns
        --------
        requested_qtr_data : pd.DataFrame
            The DataFrame with the latest values for the requested quarter
            for all columns; `dates` are the index and columns are a MultiIndex
            with sids at the top level and the dataset columns on the bottom.
        """
        requested_qtr_data = stacked_last_per_qtr.loc[idx]
        # We've lost the index names when doing `loc`, so set them here.
        requested_qtr_data.index = requested_qtr_data.index.set_names(
            idx.names
        )
        requested_qtr_data = requested_qtr_data.reset_index(
            SHIFTED_NORMALIZED_QTRS
        )
        # Calculate the actual year/quarter being requested and add those in
        # as columns.
        (requested_qtr_data[FISCAL_YEAR_FIELD_NAME],
         requested_qtr_data[FISCAL_QUARTER_FIELD_NAME]) = \
            split_normalized_quarters(
                requested_qtr_data[SHIFTED_NORMALIZED_QTRS]
            )
        # Once we're left with just dates as the index, we can reindex by all
        # dates so that we have a value for each calendar date.
        return requested_qtr_data.unstack(SID_FIELD_NAME).reindex(dates)

    def get_adjustments(self,
                        zero_qtr_data,
                        requested_qtr_data,
                        last_per_qtr,
                        dates,
                        assets,
                        columns):
        """
        Creates an AdjustedArray from the given estimates data for the given
        dates.

        Parameters
        ----------
        zero_qtr_data : pd.DataFrame
            The 'time zero' data for each calendar date per sid.
        requested_qtr_data : pd.DataFrame
            The requested quarter data for each calendar date per sid.
        last_per_qtr : pd.DataFrame
            A DataFrame with a column MultiIndex of [self.estimates.columns,
            normalized_quarters, sid] that allows easily getting the timeline
            of estimates for a particular sid for a particular quarter.
        dates : pd.DatetimeIndex
            The calendar dates for which estimates data is requested.
        assets : pd.Int64Index
            An index of all the assets from the raw data.
        columns : list of BoundColumn
            The columns for which adjustments need to be calculated.

        Returns
        -------
        adjusted_array : AdjustedArray
            The array of data and overwrites for the given column.
        """
        col_to_overwrites = defaultdict(dict)
        # We no longer need NORMALIZED_QUARTERS in the index, but we do need it
        # as a column to calculate adjustments.
        zero_qtr_data = zero_qtr_data.reset_index(NORMALIZED_QUARTERS)

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
            qtrs_with_estimates_for_sid = last_per_qtr.xs(
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
                    self.create_overwrite_for_quarter(
                        col_to_overwrites,
                        next_qtr_start_idx,
                        last_per_qtr,
                        qtrs_with_estimates_for_sid,
                        requested_qtr_data,
                        sid,
                        sid_idx,
                        columns,
                    )
        return col_to_overwrites

    def create_overwrite_for_quarter(self,
                                     overwrites,
                                     next_qtr_start_idx,
                                     last_per_qtr,
                                     quarters_with_estimates_for_sid,
                                     requested_qtr_data,
                                     sid,
                                     sid_idx,
                                     columns):
        """
        Add entries to the dictionary of columns to adjustments for the given
        sid and the given quarter.

        Parameters
        ----------
        col_to_overwrites : dict [column_name -> list of ArrayAdjustment]
            A dictionary mapping column names to all overwrites for those
            columns.
        next_qtr_start_idx : int
            The index of the first day of the next quarter in the calendar
            dates.
        last_per_qtr : pd.DataFrame
            A DataFrame with a column MultiIndex of [self.estimates.columns,
            normalized_quarters, sid] that allows easily getting the timeline
            of estimates for a particular sid for a particular quarter; this
            is particularly useful for getting adjustments for 'next'
            estimates.
        quarters_with_estimates_for_sid : np.array
            An array of all quarters for which there are estimates for the
            given sid.
        sid : int
            The sid for which to create overwrites.
        sid_idx : int
            The index of the sid in `assets`.
        columns : list of BoundColumn
            The columns for which to create overwrites.
        """
        overwrites_dict = {}
        for col in columns:
            if col.dtype == datetime64ns_dtype:
                overwrites_dict[col] = Datetime641DArrayOverwrite
            else:
                overwrites_dict[col] = Float641DArrayOverwrite

        # Find the quarter being requested in the quarter we're
        # crossing into.
        requested_quarter = requested_qtr_data[
            SHIFTED_NORMALIZED_QTRS
        ][sid].iloc[next_qtr_start_idx]
        for col in columns:
            column_name = self.name_map[col.name]
            # If there are estimates for the requested quarter,
            # overwrite all values going up to the starting index of
            # that quarter with estimates for that quarter.
            if requested_quarter in quarters_with_estimates_for_sid:
                overwrites[column_name][next_qtr_start_idx] = \
                    [self.create_overwrite_for_estimate(
                        col,
                        column_name,
                        last_per_qtr,
                        next_qtr_start_idx,
                        overwrites_dict[col],
                        requested_quarter,
                        sid,
                        sid_idx
                    )]
            # There are no estimates for the quarter. Overwrite all
            # values going up to the starting index of that quarter
            # with the missing value for this column.
            else:
                overwrites[column_name][next_qtr_start_idx] =\
                    [self.overwrite_with_null(
                        col,
                        last_per_qtr.index,
                        next_qtr_start_idx,
                        overwrites_dict[col],
                        sid_idx
                    )]

    def overwrite_with_null(self,
                            column,
                            dates,
                            next_qtr_start_idx,
                            overwrite,
                            sid_idx):
        return overwrite(
            0,
            next_qtr_start_idx - 1,
            sid_idx,
            sid_idx,
            np.full(
                len(
                    dates[:next_qtr_start_idx]
                ),
                column.missing_value,
                dtype=column.dtype
            ))

    def load_adjusted_array(self, columns, dates, assets, mask):
        # Separate out getting the columns' datasets and the datasets'
        # num_quarters attributes to ensure that we're catching the right
        # AttributeError.
        col_to_datasets = {col: col.dataset for col in columns}
        try:
            groups = groupby(lambda col: col_to_datasets[col].num_quarters,
                             col_to_datasets)
        except AttributeError:
            raise AttributeError("Datasets loaded via the "
                                 "QuarterEstimatesLoader must define a "
                                 "`num_quarters` attribute that defines how "
                                 "many quarters out the loader should load "
                                 "the data relative to `dates`.")
        if any(num_qtr < 0 for num_qtr in groups):
            raise ValueError(
                INVALID_NUM_QTRS_MESSAGE % ','.join(
                    str(qtr) for qtr in groups if qtr < 0
                )

            )
        out = {}
        # To optimize performance, only work below on assets that are
        # actually in the raw data.
        assets_with_data = set.intersection(
            set(assets), set(self.estimates[SID_FIELD_NAME])
        )
        for num_quarters, columns in groups.items():
            last_per_qtr, stacked_last_per_qtr = self.get_last_data_per_qtr(
                assets_with_data, columns, dates
            )
            # Determine which quarter is immediately next/previous for each
            # date.
            zeroth_quarter_idx = self.get_zeroth_quarter_idx(
                num_quarters, stacked_last_per_qtr
            )
            zero_qtr_data = stacked_last_per_qtr.loc[zeroth_quarter_idx]
            # Doing it this way because creating a MultiIndex from scratch
            # results in being unable to unstack sids because of duplicate
            # values, even though the MultiIndex is created with the same
            # exact values as below - possible pandas bug.
            requested_qtr_idx = zero_qtr_data.reset_index(
                NORMALIZED_QUARTERS
            ).set_index(
                pd.Series(self.get_shifted_qtrs(
                    zeroth_quarter_idx.get_level_values(NORMALIZED_QUARTERS),
                    num_quarters
                ), name=SHIFTED_NORMALIZED_QTRS),
                append=True
            ).index
            requested_qtr_data = self.get_requested_quarter_data(
                stacked_last_per_qtr, requested_qtr_idx, dates
            )

            # Calculate all adjustments for the given quarter and accumulate
            # them for each column.
            col_to_adjustments = self.get_adjustments(zero_qtr_data,
                                                      requested_qtr_data,
                                                      last_per_qtr,
                                                      dates,
                                                      assets_with_data,
                                                      columns)
            for col in columns:
                column_name = self.name_map[col.name]
                # We may have dropped assets if they never have any data for
                # the requested quarter.
                df = pd.DataFrame(data=requested_qtr_data[column_name],
                                  index=dates,
                                  columns=assets,
                                  dtype=col.dtype)

                out[col] = AdjustedArray(
                    df.values.astype(col.dtype),
                    mask,
                    dict(col_to_adjustments[column_name]),
                    col.missing_value,
                )
        return out

    def get_last_data_per_qtr(self, assets_with_data, columns, dates):
        """
        Determine the last piece of information we know for each column on each
        date in the index for each sid and quarter.

        Parameters
        ----------
        assets_with_data : pd.Index
            Index of all assets that appear in the raw data given to the
            loader.
        columns : iterable of BoundColumn
            The columns that need to be loaded from the raw data.
        dates : pd.DatetimeIndex
            The calendar of dates for which data should be loaded.

        Returns
        -------
        stacked_last_per_qtr : pd.DataFrame
            A DataFrame indexed by [dates, sid, normalized_quarters] that has
            the latest information for each row of the index, sorted by event
            date.
        last_per_qtr : pd.DataFrame
            A DataFrame with columns that are a MultiIndex of [
            self.estimates.columns, normalized_quarters, sid].
        """
        # Get a DataFrame indexed by date with a MultiIndex of columns of [
        # self.estimates.columns, normalized_quarters, sid], where each cell
        # contains the latest data for that day.
        last_per_qtr = last_in_date_group(
            self.estimates, dates, assets_with_data, reindex=True,
            extra_groupers=[NORMALIZED_QUARTERS]
        )
        # Forward fill values for each quarter/sid/dataset column.
        ffill_across_cols(last_per_qtr, columns, self.name_map)
        # Stack quarter and sid into the index.
        stacked_last_per_qtr = last_per_qtr.stack([SID_FIELD_NAME,
                                                   NORMALIZED_QUARTERS])
        # Set date index name for ease of reference
        stacked_last_per_qtr.index.set_names(SIMULTATION_DATES,
                                             level=0,
                                             inplace=True)
        stacked_last_per_qtr = stacked_last_per_qtr.sort(
            EVENT_DATE_FIELD_NAME
        )
        stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] = pd.to_datetime(
            stacked_last_per_qtr[EVENT_DATE_FIELD_NAME]
        )
        return last_per_qtr, stacked_last_per_qtr


class NextQuartersEstimatesLoader(QuarterEstimatesLoader):
    def create_overwrite_for_estimate(self,
                                      column,
                                      column_name,
                                      last_per_qtr,
                                      next_qtr_start_idx,
                                      overwrite,
                                      requested_quarter,
                                      sid,
                                      sid_idx):
        return overwrite(
            0,
            # overwrite thru last qtr
            next_qtr_start_idx - 1,
            sid_idx,
            sid_idx,
            last_per_qtr[
                column_name,
                requested_quarter,
                sid
            ][0:next_qtr_start_idx].values)

    def get_shifted_qtrs(self, zero_qtrs, num_quarters):
        return zero_qtrs + (num_quarters - 1)

    def get_zeroth_quarter_idx(self, num_quarters, stacked_last_per_qtr):
        """
        Filters for releases that are on or after each simulation date and
        determines the next quarter by picking out the upcoming release for
        each date in the index.

        Parameters
        ----------
        num_quarters : int
            Number of quarters to go out in the future.
        stacked_last_per_qtr : pd.DataFrame
            A DataFrame with index of calendar dates, sid, and normalized
            quarters with each row being the latest estimate for the row's
            index values, sorted by event date.

        Returns
        -------
        next_releases_per_date_index : pd.MultiIndex
            An index of calendar dates, sid, and normalized quarters, for only
            the rows that have a next event.
        """

        # We reset the index here because in pandas3, a groupby on the index
        # will set the index to just the items in the groupby, so we will lose
        # the normalized quarters.
        next_releases_per_date = stacked_last_per_qtr.loc[
            stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] >=
            stacked_last_per_qtr.index.get_level_values(SIMULTATION_DATES)
        ].reset_index(NORMALIZED_QUARTERS).groupby(
            level=[SIMULTATION_DATES, SID_FIELD_NAME]
        ).nth(0).set_index(NORMALIZED_QUARTERS, append=True)
        return next_releases_per_date.index


class PreviousQuartersEstimatesLoader(QuarterEstimatesLoader):
    def create_overwrite_for_estimate(self,
                                      column,
                                      column_name,
                                      dates,
                                      next_qtr_start_idx,
                                      overwrite,
                                      requested_quarter,
                                      sid,
                                      sid_idx):
        return self.overwrite_with_null(column,
                                        dates,
                                        next_qtr_start_idx,
                                        overwrite,
                                        sid_idx)

    def get_shifted_qtrs(self, zero_qtrs, num_quarters):
        return zero_qtrs - (num_quarters - 1)

    def get_zeroth_quarter_idx(self, num_quarters, stacked_last_per_qtr):
        """
        Filters for releases that are on or after each simulation date and
        determines the previous quarter by picking out the most recent
        release relative to each date in the index.

        Parameters
        ----------
        num_quarters : int
            Number of quarters to go out in the past.
        stacked_last_per_qtr : pd.DataFrame
            A DataFrame with index of calendar dates, sid, and normalized
            quarters with each row being the latest estimate for the row's
            index values, sorted by event date.

        Returns
        -------
        previous_releases_per_date_index : pd.MultiIndex
            An index of calendar dates, sid, and normalized quarters, for only
            the rows that have a previous event.
        """

        # We reset the index here because in pandas3, a groupby on the index
        # will set the index to just the items in the groupby, so we will lose
        # the normalized quarters.
        previous_releases_per_date = stacked_last_per_qtr.loc[
            stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] <=
            stacked_last_per_qtr.index.get_level_values(SIMULTATION_DATES)
        ].reset_index(NORMALIZED_QUARTERS).groupby(
            level=[SIMULTATION_DATES, SID_FIELD_NAME]
        ).nth(-1).set_index(NORMALIZED_QUARTERS, append=True)
        return previous_releases_per_date.index
