from abc import abstractmethod

from interface import implements
import numpy as np
import pandas as pd
from toolz import groupby

from zipline.lib.adjusted_array import AdjustedArray
from zipline.lib.adjustment import (
    Datetime641DArrayOverwrite,
    Datetime64Overwrite,
    Float641DArrayOverwrite,
    Float64Multiply,
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
from zipline.utils.date_utils import make_utc_aware
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype
from zipline.pipeline.loaders.utils import (
    ffill_across_cols,
    last_in_date_group,
)

INVALID_NUM_QTRS_MESSAGE = (
    "Passed invalid number of quarters %s; " "must pass a number of quarters >= 0"
)
NEXT_FISCAL_QUARTER = "next_fiscal_quarter"
NEXT_FISCAL_YEAR = "next_fiscal_year"
NORMALIZED_QUARTERS = "normalized_quarters"
PREVIOUS_FISCAL_QUARTER = "previous_fiscal_quarter"
PREVIOUS_FISCAL_YEAR = "previous_fiscal_year"
SHIFTED_NORMALIZED_QTRS = "shifted_normalized_quarters"
SIMULATION_DATES = "dates"


def normalize_quarters(years, quarters):
    return years * 4 + quarters - 1


def split_normalized_quarters(normalized_quarters):
    years = normalized_quarters // 4
    quarters = normalized_quarters % 4
    return years, quarters + 1


# These metadata columns are used to align event indexers.
metadata_columns = frozenset(
    {
        TS_FIELD_NAME,
        SID_FIELD_NAME,
        EVENT_DATE_FIELD_NAME,
        FISCAL_QUARTER_FIELD_NAME,
        FISCAL_YEAR_FIELD_NAME,
    }
)


def required_estimates_fields(columns):
    """Compute the set of resource columns required to serve
    `columns`.
    """
    # We also expect any of the field names that our loadable columns
    # are mapped to.
    return metadata_columns.union(columns.values())


def validate_column_specs(events, columns):
    """Verify that the columns of ``events`` can be used by a
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


def add_new_adjustments(adjustments_dict, adjustments, column_name, ts):
    try:
        adjustments_dict[column_name][ts].extend(adjustments)
    except KeyError:
        adjustments_dict[column_name][ts] = adjustments


class EarningsEstimatesLoader(implements(PipelineLoader)):
    """An abstract pipeline loader for estimates data that can load data a
    variable number of quarters forwards/backwards from calendar dates
    depending on the `num_announcements` attribute of the columns' dataset.
    If split adjustments are to be applied, a loader, split-adjusted columns,
    and the split-adjusted asof-date must be supplied.

    Parameters
    ----------
    estimates : pd.DataFrame
        The raw estimates data; must contain at least 5 columns:
            sid : int64
                The asset id associated with each estimate.

            event_date : datetime64[ns]
                The date on which the event that the estimate is for will/has
                occurred.

            timestamp : datetime64[ns]
                The datetime where we learned about the estimate.

            fiscal_quarter : int64
                The quarter during which the event has/will occur.

            fiscal_year : int64
                The year during which the event has/will occur.

    name_map : dict[str -> str]
        A map of names of BoundColumns that this loader will load to the
        names of the corresponding columns in `events`.
    """

    def __init__(self, estimates, name_map):
        validate_column_specs(estimates, name_map)

        self.estimates = estimates[
            estimates[EVENT_DATE_FIELD_NAME].notnull()
            & estimates[FISCAL_QUARTER_FIELD_NAME].notnull()
            & estimates[FISCAL_YEAR_FIELD_NAME].notnull()
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
    def get_zeroth_quarter_idx(self, stacked_last_per_qtr):
        raise NotImplementedError("get_zeroth_quarter_idx")

    @abstractmethod
    def get_shifted_qtrs(self, zero_qtrs, num_announcements):
        raise NotImplementedError("get_shifted_qtrs")

    @abstractmethod
    def create_overwrite_for_estimate(
        self,
        column,
        column_name,
        last_per_qtr,
        next_qtr_start_idx,
        requested_quarter,
        sid,
        sid_idx,
        col_to_split_adjustments,
        split_adjusted_asof_idx,
    ):
        raise NotImplementedError("create_overwrite_for_estimate")

    @property
    @abstractmethod
    def searchsorted_side(self):
        return NotImplementedError("searchsorted_side")

    def get_requested_quarter_data(
        self,
        zero_qtr_data,
        zeroth_quarter_idx,
        stacked_last_per_qtr,
        num_announcements,
        dates,
    ):
        """Selects the requested data for each date.

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
        num_announcements : int
            The number of annoucements out the user requested relative to
            each date in the calendar dates.
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

        requested_qtr_data = stacked_last_per_qtr.reindex(index=requested_qtr_idx)
        requested_qtr_data = requested_qtr_data.reset_index(
            SHIFTED_NORMALIZED_QTRS,
        )
        # Calculate the actual year/quarter being requested and add those in
        # as columns.
        (
            requested_qtr_data[FISCAL_YEAR_FIELD_NAME],
            requested_qtr_data[FISCAL_QUARTER_FIELD_NAME],
        ) = split_normalized_quarters(requested_qtr_data[SHIFTED_NORMALIZED_QTRS])
        # Once we're left with just dates as the index, we can reindex by all
        # dates so that we have a value for each calendar date.
        return requested_qtr_data.unstack(SID_FIELD_NAME).reindex(dates)

    def get_split_adjusted_asof_idx(self, dates):
        """Compute the index in `dates` where the split-adjusted-asof-date
        falls. This is the date up to which, and including which, we will
        need to unapply all adjustments for and then re-apply them as they
        come in. After this date, adjustments are applied as normal.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            The calendar dates over which the Pipeline is being computed.

        Returns
        -------
        split_adjusted_asof_idx : int
            The index in `dates` at which the data should be split.
        """
        split_adjusted_asof_idx = dates.searchsorted(self._split_adjusted_asof)
        # make_utc_aware(pd.DatetimeIndex(self._split_adjusted_asof))
        # The split-asof date is after the date index.
        if split_adjusted_asof_idx == len(dates):
            split_adjusted_asof_idx = len(dates) - 1
        if self._split_adjusted_asof.tzinfo is not None:
            if self._split_adjusted_asof < dates[0]:
                split_adjusted_asof_idx = -1
        else:
            if self._split_adjusted_asof < dates[0]:
                split_adjusted_asof_idx = -1
        return split_adjusted_asof_idx

    def collect_overwrites_for_sid(
        self,
        group,
        dates,
        requested_qtr_data,
        last_per_qtr,
        sid_idx,
        columns,
        all_adjustments_for_sid,
        sid,
    ):
        """Given a sid, collect all overwrites that should be applied for this
        sid at each quarter boundary.

        Parameters
        ----------
        group : pd.DataFrame
            The data for `sid`.
        dates : pd.DatetimeIndex
            The calendar dates for which estimates data is requested.
        requested_qtr_data : pd.DataFrame
            The DataFrame with the latest values for the requested quarter
            for all columns.
        last_per_qtr : pd.DataFrame
            A DataFrame with a column MultiIndex of [self.estimates.columns,
            normalized_quarters, sid] that allows easily getting the timeline
            of estimates for a particular sid for a particular quarter.
        sid_idx : int
            The sid's index in the asset index.
        columns : list of BoundColumn
            The columns for which the overwrites should be computed.
        all_adjustments_for_sid : dict[int -> AdjustedArray]
            A dictionary of the integer index of each timestamp into the date
            index, mapped to adjustments that should be applied at that
            index for the given sid (`sid`). This dictionary is modified as
            adjustments are collected.
        sid : int
            The sid for which overwrites should be computed.
        """
        # If data was requested for only 1 date, there can never be any
        # overwrites, so skip the extra work.
        if len(dates) == 1:
            return

        next_qtr_start_indices = dates.searchsorted(
            pd.DatetimeIndex(group[EVENT_DATE_FIELD_NAME]),
            side=self.searchsorted_side,
        )

        qtrs_with_estimates = group.index.get_level_values(NORMALIZED_QUARTERS).values
        for idx in next_qtr_start_indices:
            if 0 < idx < len(dates):
                # Find the quarter being requested in the quarter we're
                # crossing into.
                requested_quarter = requested_qtr_data[
                    SHIFTED_NORMALIZED_QTRS,
                    sid,
                ].iloc[idx]
                # Only add adjustments if the next quarter starts somewhere
                # in our date index for this sid. Our 'next' quarter can
                # never start at index 0; a starting index of 0 means that
                # the next quarter's event date was NaT.
                self.create_overwrites_for_quarter(
                    all_adjustments_for_sid,
                    idx,
                    last_per_qtr,
                    qtrs_with_estimates,
                    requested_quarter,
                    sid,
                    sid_idx,
                    columns,
                )

    def get_adjustments_for_sid(
        self,
        group,
        dates,
        requested_qtr_data,
        last_per_qtr,
        sid_to_idx,
        columns,
        col_to_all_adjustments,
        **kwargs,
    ):
        """

        Parameters
        ----------
        group : pd.DataFrame
            The data for the given sid.
        dates : pd.DatetimeIndex
            The calendar dates for which estimates data is requested.
        requested_qtr_data : pd.DataFrame
            The DataFrame with the latest values for the requested quarter
            for all columns.
        last_per_qtr : pd.DataFrame
            A DataFrame with a column MultiIndex of [self.estimates.columns,
            normalized_quarters, sid] that allows easily getting the timeline
            of estimates for a particular sid for a particular quarter.
        sid_to_idx : dict[int -> int]
            A dictionary mapping sid to he sid's index in the asset index.
        columns : list of BoundColumn
            The columns for which the overwrites should be computed.
        col_to_all_adjustments : dict[int -> AdjustedArray]
            A dictionary of the integer index of each timestamp into the date
            index, mapped to adjustments that should be applied at that
            index. This dictionary is for adjustments for ALL sids. It is
            modified as adjustments are collected.
        kwargs :
            Additional arguments used in collecting adjustments; unused here.
        """
        # Collect all adjustments for a given sid.
        all_adjustments_for_sid = {}
        sid = int(group.name)
        self.collect_overwrites_for_sid(
            group,
            dates,
            requested_qtr_data,
            last_per_qtr,
            sid_to_idx[sid],
            columns,
            all_adjustments_for_sid,
            sid,
        )
        self.merge_into_adjustments_for_all_sids(
            all_adjustments_for_sid, col_to_all_adjustments
        )

    def merge_into_adjustments_for_all_sids(
        self, all_adjustments_for_sid, col_to_all_adjustments
    ):
        """Merge adjustments for a particular sid into a dictionary containing
        adjustments for all sids.

        Parameters
        ----------
        all_adjustments_for_sid : dict[int -> AdjustedArray]
            All adjustments for a particular sid.
        col_to_all_adjustments : dict[int -> AdjustedArray]
            All adjustments for all sids.
        """

        for col_name in all_adjustments_for_sid:
            if col_name not in col_to_all_adjustments:
                col_to_all_adjustments[col_name] = {}
            for ts in all_adjustments_for_sid[col_name]:
                adjs = all_adjustments_for_sid[col_name][ts]
                add_new_adjustments(col_to_all_adjustments, adjs, col_name, ts)

    def get_adjustments(
        self,
        zero_qtr_data,
        requested_qtr_data,
        last_per_qtr,
        dates,
        assets,
        columns,
        **kwargs,
    ):
        """Creates an AdjustedArray from the given estimates data for the given
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
        kwargs :
            Additional keyword arguments that should be forwarded to
            `get_adjustments_for_sid` and to be used in computing adjustments
            for each sid.

        Returns
        -------
        col_to_all_adjustments : dict[int -> AdjustedArray]
            A dictionary of all adjustments that should be applied.
        """

        zero_qtr_data.sort_index(inplace=True)
        # Here we want to get the LAST record from each group of records
        # corresponding to a single quarter. This is to ensure that we select
        # the most up-to-date event date in case the event date changes.
        quarter_shifts = zero_qtr_data.groupby(
            level=[SID_FIELD_NAME, NORMALIZED_QUARTERS]
        ).nth(-1)

        col_to_all_adjustments = {}
        sid_to_idx = dict(zip(assets, range(len(assets))))
        quarter_shifts.groupby(level=SID_FIELD_NAME).apply(
            self.get_adjustments_for_sid,
            dates,
            requested_qtr_data,
            last_per_qtr,
            sid_to_idx,
            columns,
            col_to_all_adjustments,
            **kwargs,
        )
        return col_to_all_adjustments

    def create_overwrites_for_quarter(
        self,
        col_to_overwrites,
        next_qtr_start_idx,
        last_per_qtr,
        quarters_with_estimates_for_sid,
        requested_quarter,
        sid,
        sid_idx,
        columns,
    ):
        """Add entries to the dictionary of columns to adjustments for the given
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
        requested_quarter : float
            The quarter for which the overwrite should be created.
        sid : int
            The sid for which to create overwrites.
        sid_idx : int
            The index of the sid in `assets`.
        columns : list of BoundColumn
            The columns for which to create overwrites.
        """
        for col in columns:
            column_name = self.name_map[col.name]
            if column_name not in col_to_overwrites:
                col_to_overwrites[column_name] = {}
            # If there are estimates for the requested quarter,
            # overwrite all values going up to the starting index of
            # that quarter with estimates for that quarter.
            if requested_quarter in quarters_with_estimates_for_sid:
                adjs = self.create_overwrite_for_estimate(
                    col,
                    column_name,
                    last_per_qtr,
                    next_qtr_start_idx,
                    requested_quarter,
                    sid,
                    sid_idx,
                )
                add_new_adjustments(
                    col_to_overwrites, adjs, column_name, next_qtr_start_idx
                )
            # There are no estimates for the quarter. Overwrite all
            # values going up to the starting index of that quarter
            # with the missing value for this column.
            else:
                adjs = [self.overwrite_with_null(col, next_qtr_start_idx, sid_idx)]
                add_new_adjustments(
                    col_to_overwrites, adjs, column_name, next_qtr_start_idx
                )

    def overwrite_with_null(self, column, next_qtr_start_idx, sid_idx):
        return self.scalar_overwrites_dict[column.dtype](
            0, next_qtr_start_idx - 1, sid_idx, sid_idx, column.missing_value
        )

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        # Separate out getting the columns' datasets and the datasets'
        # num_announcements attributes to ensure that we're catching the right
        # AttributeError.
        col_to_datasets = {col: col.dataset for col in columns}
        try:
            groups = groupby(
                lambda col: col_to_datasets[col].num_announcements, col_to_datasets
            )
        except AttributeError as exc:
            raise AttributeError(
                "Datasets loaded via the "
                "EarningsEstimatesLoader must define a "
                "`num_announcements` attribute that defines "
                "how many quarters out the loader should load"
                " the data relative to `dates`."
            ) from exc
        if any(num_qtr < 0 for num_qtr in groups):
            raise ValueError(
                INVALID_NUM_QTRS_MESSAGE
                % ",".join(str(qtr) for qtr in groups if qtr < 0)
            )
        out = {}
        # To optimize performance, only work below on assets that are
        # actually in the raw data.
        data_query_cutoff_times = domain.data_query_cutoff_for_sessions(dates)
        assets_with_data = set(sids) & set(self.estimates[SID_FIELD_NAME])
        last_per_qtr, stacked_last_per_qtr = self.get_last_data_per_qtr(
            assets_with_data,
            columns,
            dates,
            data_query_cutoff_times,
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
            col_to_adjustments = self.get_adjustments(
                zero_qtr_data, requested_qtr_data, last_per_qtr, dates, sids, columns
            )

            # Lookup the asset indexer once, this is so we can reindex
            # the assets returned into the assets requested for each column.
            # This depends on the fact that our column pd.MultiIndex has the same
            # sids for each field. This allows us to do the lookup once on
            # level 1 instead of doing the lookup each time per value in
            # level 0.
            # asset_indexer = sids.get_indexer_for(
            #     requested_qtr_data.columns.levels[1],
            # )
            for col in columns:
                column_name = self.name_map[col.name]
                # allocate the empty output with the correct missing value
                # shape = len(dates), len(sids)
                # output_array = np.full(shape=shape,
                #                        fill_value=col.missing_value,
                #                        dtype=col.dtype)
                # overwrite the missing value with values from the computed data
                try:
                    output_array = (
                        requested_qtr_data[column_name]
                        .reindex(sids, axis=1)
                        .to_numpy()
                        .astype(col.dtype)
                    )
                except Exception:
                    output_array = (
                        requested_qtr_data[column_name]
                        .reindex(sids, axis=1)
                        .to_numpy(na_value=col.missing_value)
                        .astype(col.dtype)
                    )

                # except ValueError:
                #     np.copyto(output_array[:, asset_indexer],
                #               requested_qtr_data[column_name].to_numpy(na_value=output_array.dtype),
                #               casting='unsafe')
                out[col] = AdjustedArray(
                    output_array,
                    # There may not be any adjustments at all (e.g. if
                    # len(date) == 1), so provide a default.
                    dict(col_to_adjustments.get(column_name, {})),
                    col.missing_value,
                )
        return out

    def get_last_data_per_qtr(
        self, assets_with_data, columns, dates, data_query_cutoff_times
    ):
        """Determine the last piece of information we know for each column on each
        date in the index for each sid and quarter.

        Parameters
        ----------
        assets_with_data : pd.Index
            Index of all assets that appear in the raw data given to the
            loader.
        columns : iterable of BoundColumn
            The columns that need to be loaded from the raw data.
        data_query_cutoff_times : pd.DatetimeIndex
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
        # Get a DataFrame indexed by date with a MultiIndex of columns of
        # [self.estimates.columns, normalized_quarters, sid], where each cell
        # contains the latest data for that day.
        last_per_qtr = last_in_date_group(
            self.estimates,
            data_query_cutoff_times,
            assets_with_data,
            reindex=True,
            extra_groupers=[NORMALIZED_QUARTERS],
        )
        last_per_qtr.index = dates
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
        stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] = pd.to_datetime(
            stacked_last_per_qtr[EVENT_DATE_FIELD_NAME]
        )
        stacked_last_per_qtr = stacked_last_per_qtr.sort_values(EVENT_DATE_FIELD_NAME)
        return last_per_qtr, stacked_last_per_qtr


class NextEarningsEstimatesLoader(EarningsEstimatesLoader):
    searchsorted_side = "right"

    def create_overwrite_for_estimate(
        self,
        column,
        column_name,
        last_per_qtr,
        next_qtr_start_idx,
        requested_quarter,
        sid,
        sid_idx,
        col_to_split_adjustments=None,
        split_adjusted_asof_idx=None,
    ):
        return [
            self.array_overwrites_dict[column.dtype](
                0,
                next_qtr_start_idx - 1,
                sid_idx,
                sid_idx,
                last_per_qtr[
                    column_name,
                    requested_quarter,
                    sid,
                ].values[:next_qtr_start_idx],
            )
        ]

    def get_shifted_qtrs(self, zero_qtrs, num_announcements):
        return zero_qtrs + (num_announcements - 1)

    def get_zeroth_quarter_idx(self, stacked_last_per_qtr):
        """Filters for releases that are on or after each simulation date and
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
        next_releases_per_date = (
            stacked_last_per_qtr.loc[
                stacked_last_per_qtr[EVENT_DATE_FIELD_NAME]
                >= stacked_last_per_qtr.index.get_level_values(SIMULATION_DATES)
            ]
            .groupby(
                level=[SIMULATION_DATES, SID_FIELD_NAME],
                as_index=False,
                # Here we take advantage of the fact that `stacked_last_per_qtr` is
                # sorted by event date.
            )
            .nth(0)
        )
        return next_releases_per_date.index


class PreviousEarningsEstimatesLoader(EarningsEstimatesLoader):
    searchsorted_side = "left"

    def create_overwrite_for_estimate(
        self,
        column,
        column_name,
        dates,
        next_qtr_start_idx,
        requested_quarter,
        sid,
        sid_idx,
        col_to_split_adjustments=None,
        split_adjusted_asof_idx=None,
        split_dict=None,
    ):
        return [
            self.overwrite_with_null(
                column,
                next_qtr_start_idx,
                sid_idx,
            )
        ]

    def get_shifted_qtrs(self, zero_qtrs, num_announcements):
        return zero_qtrs - (num_announcements - 1)

    def get_zeroth_quarter_idx(self, stacked_last_per_qtr):
        """Filters for releases that are on or after each simulation date and
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
        previous_releases_per_date = (
            stacked_last_per_qtr.loc[
                stacked_last_per_qtr[EVENT_DATE_FIELD_NAME]
                <= stacked_last_per_qtr.index.get_level_values(SIMULATION_DATES)
            ]
            .groupby(
                level=[SIMULATION_DATES, SID_FIELD_NAME],
                as_index=False,
                # Here we take advantage of the fact that `stacked_last_per_qtr` is
                # sorted by event date.
            )
            .nth(-1)
        )
        return previous_releases_per_date.index


def validate_split_adjusted_column_specs(name_map, columns):
    to_be_split = set(columns)
    available = set(name_map.keys())
    extra = to_be_split - available
    if extra:
        raise ValueError(
            "EarningsEstimatesLoader got the following extra columns to be "
            "split-adjusted: {extra}.\n"
            "Got Columns: {to_be_split}\n"
            "Available Columns: {available}".format(
                extra=sorted(extra),
                to_be_split=sorted(to_be_split),
                available=sorted(available),
            )
        )


class SplitAdjustedEstimatesLoader(EarningsEstimatesLoader):
    """Estimates loader that loads data that needs to be split-adjusted.

    Parameters
    ----------
    split_adjustments_loader : SQLiteAdjustmentReader
        The loader to use for reading split adjustments.
    split_adjusted_column_names : iterable of str
        The column names that should be split-adjusted.
    split_adjusted_asof : pd.Timestamp
        The date that separates data into 2 halves: the first half is the set
        of dates up to and including the split_adjusted_asof date. All
        adjustments occurring during this first half are applied  to all
        dates in this first half. The second half is the set of dates after
        the split_adjusted_asof date. All adjustments occurring during this
        second half are applied sequentially as they appear in the timeline.
    """

    def __init__(
        self,
        estimates,
        name_map,
        split_adjustments_loader,
        split_adjusted_column_names,
        split_adjusted_asof,
    ):
        validate_split_adjusted_column_specs(name_map, split_adjusted_column_names)
        self._split_adjustments = split_adjustments_loader
        self._split_adjusted_column_names = split_adjusted_column_names
        self._split_adjusted_asof = split_adjusted_asof
        self._split_adjustment_dict = {}
        super(SplitAdjustedEstimatesLoader, self).__init__(estimates, name_map)

    @abstractmethod
    def collect_split_adjustments(
        self,
        adjustments_for_sid,
        requested_qtr_data,
        dates,
        sid,
        sid_idx,
        sid_estimates,
        split_adjusted_asof_idx,
        pre_adjustments,
        post_adjustments,
        requested_split_adjusted_columns,
    ):
        raise NotImplementedError("collect_split_adjustments")

    def get_adjustments_for_sid(
        self,
        group,
        dates,
        requested_qtr_data,
        last_per_qtr,
        sid_to_idx,
        columns,
        col_to_all_adjustments,
        split_adjusted_asof_idx=None,
        split_adjusted_cols_for_group=None,
    ):
        """Collects both overwrites and adjustments for a particular sid.

        Parameters
        ----------
        split_adjusted_asof_idx : int
            The integer index of the date on which the data was split-adjusted.
        split_adjusted_cols_for_group : list of str
            The names of requested columns that should also be split-adjusted.
        """
        all_adjustments_for_sid = {}
        sid = int(group.name)
        self.collect_overwrites_for_sid(
            group,
            dates,
            requested_qtr_data,
            last_per_qtr,
            sid_to_idx[sid],
            columns,
            all_adjustments_for_sid,
            sid,
        )
        (
            pre_adjustments,
            post_adjustments,
        ) = self.retrieve_split_adjustment_data_for_sid(
            dates, sid, split_adjusted_asof_idx
        )
        sid_estimates = self.estimates[self.estimates[SID_FIELD_NAME] == sid]
        # We might not have any overwrites but still have
        # adjustments, and we will need to manually add columns if
        # that is the case.
        for col_name in split_adjusted_cols_for_group:
            if col_name not in all_adjustments_for_sid:
                all_adjustments_for_sid[col_name] = {}

        self.collect_split_adjustments(
            all_adjustments_for_sid,
            requested_qtr_data,
            dates,
            sid,
            sid_to_idx[sid],
            sid_estimates,
            split_adjusted_asof_idx,
            pre_adjustments,
            post_adjustments,
            split_adjusted_cols_for_group,
        )
        self.merge_into_adjustments_for_all_sids(
            all_adjustments_for_sid, col_to_all_adjustments
        )

    def get_adjustments(
        self,
        zero_qtr_data,
        requested_qtr_data,
        last_per_qtr,
        dates,
        assets,
        columns,
        **kwargs,
    ):
        """Calculates both split adjustments and overwrites for all sids."""
        split_adjusted_cols_for_group = [
            self.name_map[col.name]
            for col in columns
            if self.name_map[col.name] in self._split_adjusted_column_names
        ]
        # Add all splits to the adjustment dict for this sid.
        split_adjusted_asof_idx = self.get_split_adjusted_asof_idx(dates)
        return super(SplitAdjustedEstimatesLoader, self).get_adjustments(
            zero_qtr_data,
            requested_qtr_data,
            last_per_qtr,
            dates,
            assets,
            columns,
            split_adjusted_cols_for_group=split_adjusted_cols_for_group,
            split_adjusted_asof_idx=split_adjusted_asof_idx,
        )

    def determine_end_idx_for_adjustment(
        self, adjustment_ts, dates, upper_bound, requested_quarter, sid_estimates
    ):
        """Determines the date until which the adjustment at the given date
        index should be applied for the given quarter.

        Parameters
        ----------
        adjustment_ts : pd.Timestamp
            The timestamp at which the adjustment occurs.
        dates : pd.DatetimeIndex
            The calendar dates over which the Pipeline is being computed.
        upper_bound : int
            The index of the upper bound in the calendar dates. This is the
            index until which the adjusment will be applied unless there is
            information for the requested quarter that comes in on or before
            that date.
        requested_quarter : float
            The quarter for which we are determining how the adjustment
            should be applied.
        sid_estimates : pd.DataFrame
            The DataFrame of estimates data for the sid for which we're
            applying the given adjustment.

        Returns
        -------
        end_idx : int
            The last index to which the adjustment should be applied for the
            given quarter/sid.
        """
        end_idx = upper_bound
        # Find the next newest kd that happens on or after
        # the date of this adjustment
        newest_kd_for_qtr = sid_estimates[
            (sid_estimates[NORMALIZED_QUARTERS] == requested_quarter)
            & (sid_estimates[TS_FIELD_NAME] >= adjustment_ts)
        ][TS_FIELD_NAME].min()
        if pd.notnull(newest_kd_for_qtr):
            newest_kd_idx = dates.searchsorted(newest_kd_for_qtr)
            # make_utc_aware(pd.DatetimeIndex(newest_kd_for_qtr))
            # We have fresh information that comes in
            # before the end of the overwrite and
            # presumably is already split-adjusted to the
            # current split. We should stop applying the
            # adjustment the day before this new
            # information comes in.
            if newest_kd_idx <= upper_bound:
                end_idx = newest_kd_idx - 1
        return end_idx

    def collect_pre_split_asof_date_adjustments(
        self,
        split_adjusted_asof_date_idx,
        sid_idx,
        pre_adjustments,
        requested_split_adjusted_columns,
    ):
        """Collect split adjustments that occur before the
        split-adjusted-asof-date. All those adjustments must first be
        UN-applied at the first date index and then re-applied on the
        appropriate dates in order to match point in time share pricing data.

        Parameters
        ----------
        split_adjusted_asof_date_idx : int
            The index in the calendar dates as-of which all data was
            split-adjusted.
        sid_idx : int
            The index of the sid for which adjustments should be collected in
            the adjusted array.
        pre_adjustments : tuple(list(float), list(int))
            The adjustment values, indexes in `dates`, and timestamps for
            adjustments that happened after the split-asof-date.
        requested_split_adjusted_columns : list of str
            The requested split adjusted columns.

        Returns
        -------
        col_to_split_adjustments : dict[str -> dict[int -> list of Adjustment]]
            The adjustments for this sid that occurred on or before the
            split-asof-date.
        """
        col_to_split_adjustments = {}
        if len(pre_adjustments[0]):
            adjustment_values, date_indexes = pre_adjustments
            for column_name in requested_split_adjusted_columns:
                col_to_split_adjustments[column_name] = {}
                # We need to undo all adjustments that happen before the
                # split_asof_date here by reversing the split ratio.
                col_to_split_adjustments[column_name][0] = [
                    Float64Multiply(
                        0,
                        split_adjusted_asof_date_idx,
                        sid_idx,
                        sid_idx,
                        1 / future_adjustment,
                    )
                    for future_adjustment in adjustment_values
                ]

                for adjustment, date_index in zip(adjustment_values, date_indexes):
                    adj = Float64Multiply(
                        0, split_adjusted_asof_date_idx, sid_idx, sid_idx, adjustment
                    )
                    add_new_adjustments(
                        col_to_split_adjustments, [adj], column_name, date_index
                    )

        return col_to_split_adjustments

    def collect_post_asof_split_adjustments(
        self,
        post_adjustments,
        requested_qtr_data,
        sid,
        sid_idx,
        sid_estimates,
        requested_split_adjusted_columns,
    ):
        """Collect split adjustments that occur after the
        split-adjusted-asof-date. Each adjustment needs to be applied to all
        dates on which knowledge for the requested quarter was older than the
        date of the adjustment.

        Parameters
        ----------
        post_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)
            The adjustment values, indexes in `dates`, and timestamps for
            adjustments that happened after the split-asof-date.
        requested_qtr_data : pd.DataFrame
            The requested quarter data for each calendar date per sid.
        sid : int
            The sid for which adjustments need to be collected.
        sid_idx : int
            The index of `sid` in the adjusted array.
        sid_estimates : pd.DataFrame
            The raw estimates data for this sid.
        requested_split_adjusted_columns : list of str
            The requested split adjusted columns.
        Returns
        -------
        col_to_split_adjustments : dict[str -> dict[int -> list of Adjustment]]
            The adjustments for this sid that occurred after the
            split-asof-date.
        """
        col_to_split_adjustments = {}
        if post_adjustments:
            # Get an integer index
            requested_qtr_timeline = requested_qtr_data[SHIFTED_NORMALIZED_QTRS][
                sid
            ].reset_index()
            requested_qtr_timeline = requested_qtr_timeline[
                requested_qtr_timeline[sid].notnull()
            ]

            # Split the data into range by quarter and determine which quarter
            # was being requested in each range.
            # Split integer indexes up by quarter range
            qtr_ranges_idxs = np.split(
                requested_qtr_timeline.index,
                np.where(np.diff(requested_qtr_timeline[sid]) != 0)[0] + 1,
            )
            requested_quarters_per_range = [
                requested_qtr_timeline[sid][r[0]] for r in qtr_ranges_idxs
            ]
            # Try to apply each adjustment to each quarter range.
            for i, qtr_range in enumerate(qtr_ranges_idxs):
                for adjustment, date_index, timestamp in zip(*post_adjustments):
                    # In the default case, apply through the end of the quarter
                    upper_bound = qtr_range[-1]
                    # Find the smallest KD in estimates that is on or after the
                    # date of the given adjustment. Apply the given adjustment
                    # until that KD.
                    end_idx = self.determine_end_idx_for_adjustment(
                        timestamp,
                        requested_qtr_data.index,
                        upper_bound,
                        requested_quarters_per_range[i],
                        sid_estimates,
                    )
                    # In the default case, apply adjustment on the first day of
                    #  the quarter.
                    start_idx = qtr_range[0]
                    # If the adjustment happens during this quarter, apply the
                    # adjustment on the day it happens.
                    if date_index > start_idx:
                        start_idx = date_index
                    # We only want to apply the adjustment if we have any stale
                    # data to apply it to.
                    if qtr_range[0] <= end_idx:
                        for column_name in requested_split_adjusted_columns:
                            if column_name not in col_to_split_adjustments:
                                col_to_split_adjustments[column_name] = {}
                            adj = Float64Multiply(
                                # Always apply from first day of qtr
                                qtr_range[0],
                                end_idx,
                                sid_idx,
                                sid_idx,
                                adjustment,
                            )
                            add_new_adjustments(
                                col_to_split_adjustments, [adj], column_name, start_idx
                            )

        return col_to_split_adjustments

    def retrieve_split_adjustment_data_for_sid(
        self, dates, sid, split_adjusted_asof_idx
    ):
        """

        dates : pd.DatetimeIndex
            The calendar dates.
        sid : int
            The sid for which we want to retrieve adjustments.
        split_adjusted_asof_idx : int
            The index in `dates` as-of which the data is split adjusted.

        Returns
        -------
        pre_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)
            The adjustment values and indexes in `dates` for
            adjustments that happened before the split-asof-date.
        post_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)
            The adjustment values, indexes in `dates`, and timestamps for
            adjustments that happened after the split-asof-date.
        """
        adjustments = self._split_adjustments.get_adjustments_for_sid("splits", sid)
        sorted(adjustments, key=lambda adj: adj[0])
        # Get rid of any adjustments that happen outside of our date index.
        adjustments = list(filter(lambda x: dates[0] <= x[0] <= dates[-1], adjustments))
        adjustment_values = np.array([adj[1] for adj in adjustments])
        timestamps = pd.DatetimeIndex([adj[0] for adj in adjustments])
        # We need the first date on which we would have known about each
        # adjustment.
        date_indexes = dates.searchsorted(timestamps)
        pre_adjustment_idxs = np.where(date_indexes <= split_adjusted_asof_idx)[0]
        last_adjustment_split_asof_idx = -1
        if len(pre_adjustment_idxs):
            last_adjustment_split_asof_idx = pre_adjustment_idxs.max()
        pre_adjustments = (
            adjustment_values[: last_adjustment_split_asof_idx + 1],
            date_indexes[: last_adjustment_split_asof_idx + 1],
        )
        post_adjustments = (
            adjustment_values[last_adjustment_split_asof_idx + 1 :],
            date_indexes[last_adjustment_split_asof_idx + 1 :],
            timestamps[last_adjustment_split_asof_idx + 1 :],
        )
        return pre_adjustments, post_adjustments

    def _collect_adjustments(
        self,
        requested_qtr_data,
        sid,
        sid_idx,
        sid_estimates,
        split_adjusted_asof_idx,
        pre_adjustments,
        post_adjustments,
        requested_split_adjusted_columns,
    ):
        pre_adjustments_dict = self.collect_pre_split_asof_date_adjustments(
            split_adjusted_asof_idx,
            sid_idx,
            pre_adjustments,
            requested_split_adjusted_columns,
        )

        post_adjustments_dict = self.collect_post_asof_split_adjustments(
            post_adjustments,
            requested_qtr_data,
            sid,
            sid_idx,
            sid_estimates,
            requested_split_adjusted_columns,
        )
        return pre_adjustments_dict, post_adjustments_dict

    def merge_split_adjustments_with_overwrites(
        self, pre, post, overwrites, requested_split_adjusted_columns
    ):
        """Merge split adjustments with the dict containing overwrites.

        Parameters
        ----------
        pre : dict[str -> dict[int -> list]]
            The adjustments that occur before the split-adjusted-asof-date.
        post : dict[str -> dict[int -> list]]
            The adjustments that occur after the split-adjusted-asof-date.
        overwrites : dict[str -> dict[int -> list]]
            The overwrites across all time. Adjustments will be merged into
            this dictionary.
        requested_split_adjusted_columns : list of str
            List of names of split adjusted columns that are being requested.
        """
        for column_name in requested_split_adjusted_columns:
            # We can do a merge here because the timestamps in 'pre' and
            # 'post' are guaranteed to not overlap.
            if pre:
                # Either empty or contains all columns.
                for ts in pre[column_name]:
                    add_new_adjustments(
                        overwrites, pre[column_name][ts], column_name, ts
                    )
            if post:
                # Either empty or contains all columns.
                for ts in post[column_name]:
                    add_new_adjustments(
                        overwrites, post[column_name][ts], column_name, ts
                    )


class PreviousSplitAdjustedEarningsEstimatesLoader(
    SplitAdjustedEstimatesLoader, PreviousEarningsEstimatesLoader
):
    def collect_split_adjustments(
        self,
        adjustments_for_sid,
        requested_qtr_data,
        dates,
        sid,
        sid_idx,
        sid_estimates,
        split_adjusted_asof_idx,
        pre_adjustments,
        post_adjustments,
        requested_split_adjusted_columns,
    ):
        """Collect split adjustments for previous quarters and apply them to the
        given dictionary of splits for the given sid. Since overwrites just
        replace all estimates before the new quarter with NaN, we don't need to
        worry about re-applying split adjustments.

        Parameters
        ----------
        adjustments_for_sid : dict[str -> dict[int -> list]]
            The dictionary of adjustments to which splits need to be added.
            Initially it contains only overwrites.
        requested_qtr_data : pd.DataFrame
            The requested quarter data for each calendar date per sid.
        dates : pd.DatetimeIndex
            The calendar dates for which estimates data is requested.
        sid : int
            The sid for which adjustments need to be collected.
        sid_idx : int
            The index of `sid` in the adjusted array.
        sid_estimates : pd.DataFrame
            The raw estimates data for the given sid.
        split_adjusted_asof_idx : int
            The index in `dates` as-of which the data is split adjusted.
        pre_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)
            The adjustment values and indexes in `dates` for
            adjustments that happened before the split-asof-date.
        post_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)
            The adjustment values, indexes in `dates`, and timestamps for
            adjustments that happened after the split-asof-date.
        requested_split_adjusted_columns : list of str
            List of requested split adjusted column names.
        """
        (pre_adjustments_dict, post_adjustments_dict) = self._collect_adjustments(
            requested_qtr_data,
            sid,
            sid_idx,
            sid_estimates,
            split_adjusted_asof_idx,
            pre_adjustments,
            post_adjustments,
            requested_split_adjusted_columns,
        )
        self.merge_split_adjustments_with_overwrites(
            pre_adjustments_dict,
            post_adjustments_dict,
            adjustments_for_sid,
            requested_split_adjusted_columns,
        )


class NextSplitAdjustedEarningsEstimatesLoader(
    SplitAdjustedEstimatesLoader, NextEarningsEstimatesLoader
):
    def collect_split_adjustments(
        self,
        adjustments_for_sid,
        requested_qtr_data,
        dates,
        sid,
        sid_idx,
        sid_estimates,
        split_adjusted_asof_idx,
        pre_adjustments,
        post_adjustments,
        requested_split_adjusted_columns,
    ):
        """Collect split adjustments for future quarters. Re-apply adjustments
        that would be overwritten by overwrites. Merge split adjustments with
        overwrites into the given dictionary of splits for the given sid.

        Parameters
        ----------
        adjustments_for_sid : dict[str -> dict[int -> list]]
            The dictionary of adjustments to which splits need to be added.
            Initially it contains only overwrites.
        requested_qtr_data : pd.DataFrame
            The requested quarter data for each calendar date per sid.
        dates : pd.DatetimeIndex
            The calendar dates for which estimates data is requested.
        sid : int
            The sid for which adjustments need to be collected.
        sid_idx : int
            The index of `sid` in the adjusted array.
        sid_estimates : pd.DataFrame
            The raw estimates data for the given sid.
        split_adjusted_asof_idx : int
            The index in `dates` as-of which the data is split adjusted.
        pre_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)
            The adjustment values and indexes in `dates` for
            adjustments that happened before the split-asof-date.
        post_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)
            The adjustment values, indexes in `dates`, and timestamps for
            adjustments that happened after the split-asof-date.
        requested_split_adjusted_columns : list of str
            List of requested split adjusted column names.
        """
        (pre_adjustments_dict, post_adjustments_dict) = self._collect_adjustments(
            requested_qtr_data,
            sid,
            sid_idx,
            sid_estimates,
            split_adjusted_asof_idx,
            pre_adjustments,
            post_adjustments,
            requested_split_adjusted_columns,
        )
        for column_name in requested_split_adjusted_columns:
            for overwrite_ts in adjustments_for_sid[column_name]:
                # We need to cumulatively re-apply all adjustments up to the
                # split-adjusted-asof-date. We might not have any
                # pre-adjustments, so we should check for that.
                if overwrite_ts <= split_adjusted_asof_idx and pre_adjustments_dict:
                    for split_ts in pre_adjustments_dict[column_name]:
                        # The split has to have occurred during the span of
                        # the overwrite.
                        if split_ts < overwrite_ts:
                            # Create new adjustments here so that we can
                            # re-apply all applicable adjustments to ONLY
                            # the dates being overwritten.
                            adjustments_for_sid[column_name][overwrite_ts].extend(
                                [
                                    Float64Multiply(
                                        0,
                                        overwrite_ts - 1,
                                        sid_idx,
                                        sid_idx,
                                        adjustment.value,
                                    )
                                    for adjustment in pre_adjustments_dict[column_name][
                                        split_ts
                                    ]
                                ]
                            )
                # After the split-adjusted-asof-date, we need to re-apply all
                # adjustments that occur after that date and within the
                # bounds of the overwrite. They need to be applied starting
                # from the first date and until an end date. The end date is
                # the date of the newest information we get about
                # `requested_quarter` that is >= `split_ts`, or if there is no
                # new knowledge before `overwrite_ts`, then it is the date
                # before `overwrite_ts`.
                else:
                    # Overwrites happen at the first index of a new quarter,
                    # so determine here which quarter that is.
                    requested_quarter = requested_qtr_data[
                        SHIFTED_NORMALIZED_QTRS, sid
                    ].iloc[overwrite_ts]

                    for adjustment_value, date_index, timestamp in zip(
                        *post_adjustments
                    ):
                        if split_adjusted_asof_idx < date_index < overwrite_ts:
                            # Assume the entire overwrite contains stale data
                            upper_bound = overwrite_ts - 1
                            end_idx = self.determine_end_idx_for_adjustment(
                                timestamp,
                                dates,
                                upper_bound,
                                requested_quarter,
                                sid_estimates,
                            )
                            adjustments_for_sid[column_name][overwrite_ts].append(
                                Float64Multiply(
                                    0, end_idx, sid_idx, sid_idx, adjustment_value
                                )
                            )

        self.merge_split_adjustments_with_overwrites(
            pre_adjustments_dict,
            post_adjustments_dict,
            adjustments_for_sid,
            requested_split_adjusted_columns,
        )
