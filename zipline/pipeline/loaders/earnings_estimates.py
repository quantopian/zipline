from abc import abstractmethod, abstractproperty
from collections import defaultdict

import numpy as np
import pandas as pd
from six import viewvalues
from toolz import groupby

from zipline.lib.adjusted_array import AdjustedArray
from zipline.lib.adjustment import (
    Datetime641DArrayOverwrite,
    Datetime64Overwrite,
    Float641DArrayOverwrite,
    Float64Overwrite,
)

from zipline.pipeline.common import (
    EVENT_DATE_FIELD_NAME,
    FISCAL_QUARTER_FIELD_NAME,
    FISCAL_YEAR_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype
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
SIMULATION_DATES = 'dates'


def normalize_quarters(years, quarters):
    return years * 4 + quarters - 1


def split_normalized_quarters(normalized_quarters):
    years = normalized_quarters // 4
    quarters = normalized_quarters % 4
    return years, quarters + 1


# These metadata columns are used to align event indexers.
metadata_columns = frozenset({
    TS_FIELD_NAME,
    SID_FIELD_NAME,
    EVENT_DATE_FIELD_NAME,
    FISCAL_QUARTER_FIELD_NAME,
    FISCAL_YEAR_FIELD_NAME,
})


def required_estimates_fields(columns):
    """
    Compute the set of resource columns required to serve
    `columns`.
    """
    # We also expect any of the field names that our loadable columns
    # are mapped to.
    return metadata_columns.union(viewvalues(columns))


def validate_column_specs(events, columns):
    """
    Verify that the columns of ``events`` can be used by a
    EarningsEstimatesLoader to serve the BoundColumns described by
    `columns`.
    """
    required = required_estimates_fields(columns)
    received = set(events.columns)
    missing = required - received
    if missing:
        raise ValueError(
            "EarningsEstimatesLoader missing required columns {missing}.\n"
            "Got Columns: {received}\n"
            "Expected Columns: {required}".format(
                missing=sorted(missing),
                received=sorted(received),
                required=sorted(required),
            )
        )


class EarningsEstimatesLoader(PipelineLoader):
    """
    An abstract pipeline loader for estimates data that can load data a
    variable number of quarters forwards/backwards from calendar dates
    depending on the `num_announcements` attribute of the columns' dataset.

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

        self.array_overwrites_dict = {
            datetime64ns_dtype: Datetime641DArrayOverwrite,
            float64_dtype: Float641DArrayOverwrite,
        }
        self.scalar_overwrites_dict = {
            datetime64ns_dtype: Datetime64Overwrite,
            float64_dtype: Float64Overwrite,
        }

        self.name_map = name_map

    @abstractmethod
    def get_zeroth_quarter_idx(self, num_announcements, last, dates):
        raise NotImplementedError('get_zeroth_quarter_idx')

    @abstractmethod
    def get_shifted_qtrs(self, zero_qtrs, num_announcements):
        raise NotImplementedError('get_shifted_qtrs')

    @abstractmethod
    def create_overwrite_for_estimate(self,
                                      column,
                                      column_name,
                                      last_per_qtr,
                                      next_qtr_start_idx,
                                      requested_quarter,
                                      sid,
                                      sid_idx):
        raise NotImplementedError('create_overwrite_for_estimate')

    @abstractproperty
    def searchsorted_side(self):
        return NotImplementedError('searchsorted_side')

    def get_requested_quarter_data(self,
                                   zero_qtr_data,
                                   zeroth_quarter_idx,
                                   stacked_last_per_qtr,
                                   num_announcements,
                                   dates):
        """
        Selects the requested data for each date.

        Parameters
        ----------
        zero_qtr_data : pd.DataFrame
            The 'time zero' data for each calendar date per sid.
        zeroth_quarter_idx : pd.Index
            An index of calendar dates, sid, and normalized quarters, for only
            the rows that have a next or previous earnings estimate.
        stacked_last_per_qtr : pd.DataFrame
            The latest estimate known with the dates, normalized quarter, and
            sid as the index.
        dates : pd.DatetimeIndex
            The calendar dates for which estimates data is requested.

        Returns
        --------
        requested_qtr_data : pd.DataFrame
            The DataFrame with the latest values for the requested quarter
            for all columns; `dates` are the index and columns are a MultiIndex
            with sids at the top level and the dataset columns on the bottom.
        """
        zero_qtr_data_idx = zero_qtr_data.index
        requested_qtr_idx = pd.MultiIndex.from_arrays(
            [
                zero_qtr_data_idx.get_level_values(0),
                zero_qtr_data_idx.get_level_values(1),
                self.get_shifted_qtrs(
                    zeroth_quarter_idx.get_level_values(
                        NORMALIZED_QUARTERS,
                    ),
                    num_announcements,
                ),
            ],
            names=[
                zero_qtr_data_idx.names[0],
                zero_qtr_data_idx.names[1],
                SHIFTED_NORMALIZED_QTRS,
            ],
        )
        requested_qtr_data = stacked_last_per_qtr.loc[requested_qtr_idx]
        requested_qtr_data = requested_qtr_data.reset_index(
            SHIFTED_NORMALIZED_QTRS,
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
        zero_qtr_data.sort_index(inplace=True)
        # Here we want to get the LAST record from each group of records
        # corresponding to a single quarter. This is to ensure that we select
        # the most up-to-date event date in case the event date changes.
        quarter_shifts = zero_qtr_data.groupby(
            level=[SID_FIELD_NAME, NORMALIZED_QUARTERS]
        ).nth(-1)

        sid_to_idx = dict(zip(assets, range(len(assets))))

        for column in columns:
            column_name = self.name_map[column.name]
            col_to_overwrites[column_name] = defaultdict(list)

        def collect_adjustments(group):
            next_qtr_start_indices = dates.searchsorted(
                group[EVENT_DATE_FIELD_NAME].values,
                side=self.searchsorted_side,
            )
            sid = int(group.name)
            qtrs_with_estimates = group.index.get_level_values(
                NORMALIZED_QUARTERS
            ).values
            for idx in next_qtr_start_indices:
                if 0 < idx < len(dates):
                    # Only add adjustments if the next quarter starts somewhere
                    # in our date index for this sid. Our 'next' quarter can
                    # never start at index 0; a starting index of 0 means that
                    # the next quarter's event date was NaT.
                    self.create_overwrite_for_quarter(
                        col_to_overwrites,
                        idx,
                        last_per_qtr,
                        qtrs_with_estimates,
                        requested_qtr_data,
                        sid,
                        sid_to_idx[sid],
                        columns,
                    )

        quarter_shifts.groupby(level=SID_FIELD_NAME).apply(collect_adjustments)
        return col_to_overwrites

    def create_overwrite_for_quarter(self,
                                     col_to_overwrites,
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

        # Find the quarter being requested in the quarter we're
        # crossing into.
        requested_quarter = requested_qtr_data[
            SHIFTED_NORMALIZED_QTRS, sid,
        ].iloc[next_qtr_start_idx]
        for col in columns:
            column_name = self.name_map[col.name]
            # If there are estimates for the requested quarter,
            # overwrite all values going up to the starting index of
            # that quarter with estimates for that quarter.
            if requested_quarter in quarters_with_estimates_for_sid:
                col_to_overwrites[column_name][next_qtr_start_idx].append(
                    self.create_overwrite_for_estimate(
                        col,
                        column_name,
                        last_per_qtr,
                        next_qtr_start_idx,
                        requested_quarter,
                        sid,
                        sid_idx
                    ),
                )
            # There are no estimates for the quarter. Overwrite all
            # values going up to the starting index of that quarter
            # with the missing value for this column.
            else:
                col_to_overwrites[column_name][next_qtr_start_idx].append(
                    self.overwrite_with_null(
                        col,
                        last_per_qtr.index,
                        next_qtr_start_idx,
                        sid_idx
                    ),
                )

    def overwrite_with_null(self,
                            column,
                            dates,
                            next_qtr_start_idx,
                            sid_idx):
        return self.scalar_overwrites_dict[column.dtype](
            0,
            next_qtr_start_idx - 1,
            sid_idx,
            sid_idx,
            column.missing_value
        )

    def load_adjusted_array(self, columns, dates, assets, mask):
        # Separate out getting the columns' datasets and the datasets'
        # num_announcements attributes to ensure that we're catching the right
        # AttributeError.
        col_to_datasets = {col: col.dataset for col in columns}
        try:
            groups = groupby(lambda col:
                             col_to_datasets[col].num_announcements,
                             col_to_datasets)
        except AttributeError:
            raise AttributeError("Datasets loaded via the "
                                 "EarningsEstimatesLoader must define a "
                                 "`num_announcements` attribute that defines "
                                 "how many quarters out the loader should load"
                                 " the data relative to `dates`.")
        if any(num_qtr < 0 for num_qtr in groups):
            raise ValueError(
                INVALID_NUM_QTRS_MESSAGE % ','.join(
                    str(qtr) for qtr in groups if qtr < 0
                )

            )
        out = {}
        # To optimize performance, only work below on assets that are
        # actually in the raw data.
        assets_with_data = set(assets) & set(self.estimates[SID_FIELD_NAME])
        last_per_qtr, stacked_last_per_qtr = self.get_last_data_per_qtr(
            assets_with_data,
            columns,
            dates
        )
        # Determine which quarter is immediately next/previous for each
        # date.
        zeroth_quarter_idx = self.get_zeroth_quarter_idx(stacked_last_per_qtr)
        zero_qtr_data = stacked_last_per_qtr.loc[zeroth_quarter_idx]

        for num_announcements, columns in groups.items():
            requested_qtr_data = self.get_requested_quarter_data(
                zero_qtr_data,
                zeroth_quarter_idx,
                stacked_last_per_qtr,
                num_announcements,
                dates,
            )

            # Calculate all adjustments for the given quarter and accumulate
            # them for each column.
            col_to_adjustments = self.get_adjustments(zero_qtr_data,
                                                      requested_qtr_data,
                                                      last_per_qtr,
                                                      dates,
                                                      assets_with_data,
                                                      columns)

            # Lookup the asset indexer once, this is so we can reindex
            # the assets returned into the assets requested for each column.
            # This depends on the fact that our column multiindex has the same
            # sids for each field. This allows us to do the lookup once on
            # level 1 instead of doing the lookup each time per value in
            # level 0.
            asset_indexer = assets.get_indexer_for(
                requested_qtr_data.columns.levels[1],
            )
            for col in columns:
                column_name = self.name_map[col.name]
                # allocate the empty output with the correct missing value
                output_array = np.full(
                    (len(dates), len(assets)),
                    col.missing_value,
                    dtype=col.dtype,
                )
                # overwrite the missing value with values from the computed
                # data
                output_array[
                    :,
                    asset_indexer,
                ] = requested_qtr_data[column_name].values

                out[col] = AdjustedArray(
                    output_array,
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
            self.estimates,
            dates,
            assets_with_data,
            reindex=True,
            extra_groupers=[NORMALIZED_QUARTERS],
        )
        # Forward fill values for each quarter/sid/dataset column.
        ffill_across_cols(last_per_qtr, columns, self.name_map)
        # Stack quarter and sid into the index.
        stacked_last_per_qtr = last_per_qtr.stack(
            [SID_FIELD_NAME, NORMALIZED_QUARTERS],
        )
        # Set date index name for ease of reference
        stacked_last_per_qtr.index.set_names(
            SIMULATION_DATES,
            level=0,
            inplace=True,
        )
        stacked_last_per_qtr = stacked_last_per_qtr.sort_values(
            EVENT_DATE_FIELD_NAME,
        )
        stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] = pd.to_datetime(
            stacked_last_per_qtr[EVENT_DATE_FIELD_NAME]
        )
        return last_per_qtr, stacked_last_per_qtr


class NextEarningsEstimatesLoader(EarningsEstimatesLoader):
    searchsorted_side = 'right'

    def create_overwrite_for_estimate(self,
                                      column,
                                      column_name,
                                      last_per_qtr,
                                      next_qtr_start_idx,
                                      requested_quarter,
                                      sid,
                                      sid_idx):
        return self.array_overwrites_dict[column.dtype](
            0,
            # overwrite thru last qtr
            next_qtr_start_idx - 1,
            sid_idx,
            sid_idx,
            last_per_qtr[
                column_name,
                requested_quarter,
                sid,
            ].values[:next_qtr_start_idx],
        )

    def get_shifted_qtrs(self, zero_qtrs, num_announcements):
        return zero_qtrs + (num_announcements - 1)

    def get_zeroth_quarter_idx(self, stacked_last_per_qtr):
        """
        Filters for releases that are on or after each simulation date and
        determines the next quarter by picking out the upcoming release for
        each date in the index.

        Parameters
        ----------
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
        next_releases_per_date = stacked_last_per_qtr.loc[
            stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] >=
            stacked_last_per_qtr.index.get_level_values(SIMULATION_DATES)
        ].groupby(
            level=[SIMULATION_DATES, SID_FIELD_NAME],
            as_index=False,
            # Here we take advantage of the fact that `stacked_last_per_qtr` is
            # sorted by event date.
        ).nth(0)
        return next_releases_per_date.index


class PreviousEarningsEstimatesLoader(EarningsEstimatesLoader):
    searchsorted_side = 'left'

    def create_overwrite_for_estimate(self,
                                      column,
                                      column_name,
                                      dates,
                                      next_qtr_start_idx,
                                      requested_quarter,
                                      sid,
                                      sid_idx):
        return self.overwrite_with_null(
            column,
            dates,
            next_qtr_start_idx,
            sid_idx,
        )

    def get_shifted_qtrs(self, zero_qtrs, num_announcements):
        return zero_qtrs - (num_announcements - 1)

    def get_zeroth_quarter_idx(self, stacked_last_per_qtr):
        """
        Filters for releases that are on or after each simulation date and
        determines the previous quarter by picking out the most recent
        release relative to each date in the index.

        Parameters
        ----------
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
        previous_releases_per_date = stacked_last_per_qtr.loc[
            stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] <=
            stacked_last_per_qtr.index.get_level_values(SIMULATION_DATES)
        ].groupby(
            level=[SIMULATION_DATES, SID_FIELD_NAME],
            as_index=False,
            # Here we take advantage of the fact that `stacked_last_per_qtr` is
            # sorted by event date.
        ).nth(-1)
        return previous_releases_per_date.index
