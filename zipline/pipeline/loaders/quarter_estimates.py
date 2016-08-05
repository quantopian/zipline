from itertools import groupby
import numpy as np
import pandas as pd
from six import viewvalues
from zipline.pipeline.common import AD_FIELD_NAME, SID_FIELD_NAME, \
    EVENT_DATE_FIELD_NAME, FISCAL_QUARTER_FIELD_NAME, FISCAL_YEAR_FIELD_NAME
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.pipeline.loaders.frame import DataFrameLoader


def required_event_fields(columns):
    """
    Compute the set of resource columns required to serve
    ``next_value_columns`` and ``previous_value_columns``.
    """
    # These metadata columns are used to align event indexers.
    return {
        AD_FIELD_NAME,
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


def calc_forward_shift(qtr, num_shifts):
    yrs_to_shift, new_qtr = divmod(qtr + num_shifts, 4)
    if yrs_to_shift == 1 and new_qtr == 0:
        yrs_to_shift = 0
        new_qtr = 4
    return yrs_to_shift, new_qtr


def calc_backward_shift(qtr, num_shifts):
    yrs_to_shift, new_qtr = divmod(abs(num_shifts - qtr), 4)
    if yrs_to_shift == 0 and new_qtr == 0:
        yrs_to_shift = 1
        new_qtr = 4
    yrs_to_shift = -yrs_to_shift
    return yrs_to_shift, new_qtr


class QuarterEstimatesLoader(PipelineLoader):
    def __init__(self,
                 events,
                 columns):
        validate_column_specs(
            events,
            columns
        )

        self.events = events[
            events[EVENT_DATE_FIELD_NAME].notnull() and
            events[FISCAL_QUARTER_FIELD_NAME].notnull() and
            events[FISCAL_YEAR_FIELD_NAME].notnull()
        ]

        self.columns = columns

    def load_quarters(self, next_releases, num_quarters, dates_sids, gb):
        pass

    def load_adjusted_array(self, columns, dates, assets, mask):
        groups = groupby(lambda x: x.dataset.num_quarters, columns)
        out = {}
        date_values = pd.DataFrame(dates, columns=['dates'])
        date_values['key'] = 1
        self.events['key'] = 1
        merged = pd.merge(date_values, self.events, on='key')
        asset_df = pd.DataFrame(assets, columns=['sid'])
        asset_df['key'] = 1
        dates_sids = pd.merge(date_values, asset_df, on='key')
        for num_quarters in groups:
            columns = groups[num_quarters]
            # First, group by sid, fiscal year, and fiscal quarter and only
            # keep the last estimate made.
            final_releases_per_qtr = merged[merged.asof_date <=
                                            merged.dates].sort(
                ['dates', 'asof_date']
            ).groupby(
                ['dates', 'sid', 'fiscal_year', 'fiscal_quarter']
            ).last()
            gb = final_releases_per_qtr.reset_index().groupby(['dates', 'sid'])
            # Split the date-sid combinations into ones with a next release
            # and ones without
            eligible_next_releases = pd.concat([group[1] for group in gb if (
                group[1][EVENT_DATE_FIELD_NAME] >= group[1]['dates']
            ).any()])

            eligible_next_releases.sort(EVENT_DATE_FIELD_NAME)
            # For each sid, get the next release/year/quarter that we care
            # about.
            next_releases = eligible_next_releases.groupby(
                ['dates', 'sid']
            ).min()
            next_releases = next_releases.rename(
                columns={'fiscal_year': 'next_fiscal_year',
                         'fiscal_quarter': 'next_fiscal_quarter'}
            )

            result = self.load_quarters(next_releases,
                                        num_quarters,
                                        dates_sids)

            for c in columns:
                column_name = self.columns[c.name]
                # Need to pass a DataFrame that has dates as the index and
                # all sids as columns with column values being the value in
                # 'result' for column c
                loader = DataFrameLoader(
                    c,
                    result.pivot(index='dates',
                                 columns='sid',
                                 values=column_name),
                    adjustments=None
                )
                out[c] = loader.load_adjusted_array([c], dates, assets, mask)[c]
        return out


class NextQuartersEstimatesLoader(QuarterEstimatesLoader):
    def __init__(self,
                 events,
                 columns):
        super(NextQuartersEstimatesLoader).__init__(events, columns)

    def load_quarters(self, next_releases, num_quarters, dates_sids, gb):
        # `next_qtr` is already the next quarter over,
        # so we should offest `num_shifts` by 1.
        next_releases['fiscal_quarter'] = next_releases.apply(
            lambda x: calc_forward_shift(x['next_fiscal_quarter'],
                                         num_quarters - 1)[1],
            axis=1
        )
        next_releases['fiscal_year'] = next_releases.apply(
            lambda x:
            x['next_fiscal_year'] +
            calc_forward_shift(x['next_fiscal_quarter'],
                               num_quarters - 1)[0],
            axis=1
        )
        # Merge to get the rows we care about for each date
        result = dates_sids.merge(next_releases.reset_index(),
                                  on=(['dates', 'sid']),
                                  how='left')
        return result


class PreviousQuartersEstimatesLoader(QuarterEstimatesLoader):
    def __init__(self,
                 events,
                 columns):
        super(PreviousQuartersEstimatesLoader).__init__(events, columns)

    def load_quarters(self, next_releases, num_quarters, dates_sids, gb):
        next_releases['fiscal_quarter'] = next_releases.apply(
            lambda x: calc_backward_shift(x['next_fiscal_quarter'],
                                          num_quarters)[1],
            axis=1
        )
        next_releases['fiscal_year'] = next_releases.apply(
            lambda x:
            x['next_fiscal_year'] +
            calc_backward_shift(x['next_fiscal_quarter'],
                                num_quarters)[0],
            axis=1
        )
        only_previous_releases = pd.concat([group[1] for group in gb if (
                group[1][EVENT_DATE_FIELD_NAME] < group[1]['dates']
            ).all()])
        only_previous_releases.sort(EVENT_DATE_FIELD_NAME)
        # For each sid, get the latest release we knew about prior to
        # each simulation date.
        previous_releases = only_previous_releases.groupby(['dates',
                                                            'sid']).max()
        previous_releases = previous_releases.rename(columns={
            'fiscal_year': 'previous_fiscal_year',
            'fiscal_quarter': 'previous_fiscal_quarter'
        })
        previous_releases['fiscal_quarter'] = previous_releases.apply(
            lambda x: calc_backward_shift(x['previous_fiscal_quarter'],
                                          num_quarters)[1],
            axis=1
        )
        previous_releases['fiscal_year'] = previous_releases.apply(
            lambda x:
            x['previous_fiscal_year'] +
            calc_backward_shift(x['previous_fiscal_quarter'],
                                num_quarters)[0],
            axis=1
        )
        all_releases = pd.concat([next_releases, previous_releases])
        # Merge to get the rows we care about for each date
        result = dates_sids.merge(all_releases.reset_index(),
                                  on=(['dates', 'sid']), how='left')
        return result
