"""
Reference implementation for ConsensusEstimates loaders.
"""

from ..data import ConsensusEstimates
from .events import EventsLoader
from zipline.pipeline.common import (
    COUNT_FIELD_NAME,
    FISCAL_QUARTER_FIELD_NAME,
    FISCAL_YEAR_FIELD_NAME,
    HIGH_FIELD_NAME,
    LOW_FIELD_NAME,
    MEAN_FIELD_NAME,
    RELEASE_DATE_FIELD_NAME,
    STANDARD_DEVIATION_FIELD_NAME,
    ACTUAL_VALUE_FIELD_NAME)
from zipline.utils.memoize import lazyval


class ConsensusEstimatesLoader(EventsLoader):

    expected_cols = frozenset([RELEASE_DATE_FIELD_NAME,
                               STANDARD_DEVIATION_FIELD_NAME,
                               COUNT_FIELD_NAME,
                               FISCAL_QUARTER_FIELD_NAME,
                               HIGH_FIELD_NAME,
                               MEAN_FIELD_NAME,
                               FISCAL_YEAR_FIELD_NAME,
                               LOW_FIELD_NAME,
                               ACTUAL_VALUE_FIELD_NAME])

    event_date_col = RELEASE_DATE_FIELD_NAME

    def __init__(self, all_dates, events_by_sid,
                 infer_timestamps=False,
                 dataset=ConsensusEstimates):
        super(ConsensusEstimatesLoader, self).__init__(
            all_dates, events_by_sid, infer_timestamps, dataset=dataset,
        )

    @lazyval
    def next_release_date_loader(self):
        return self._next_event_date_loader(
            self.dataset.next_release_date,
        )

    @lazyval
    def previous_release_date_loader(self):
        return self._previous_event_date_loader(
            self.dataset.previous_release_date,
        )

    @lazyval
    def next_standard_deviation_loader(self):
        return self._next_event_value_loader(
            self.dataset.next_standard_deviation,
            STANDARD_DEVIATION_FIELD_NAME,
        )

    @lazyval
    def previous_standard_deviation_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_standard_deviation,
            STANDARD_DEVIATION_FIELD_NAME,
        )

    @lazyval
    def next_count_loader(self):
        return self._next_event_value_loader(
            self.dataset.next_count,
            COUNT_FIELD_NAME,
        )

    @lazyval
    def previous_count_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_count,
            COUNT_FIELD_NAME,
        )

    @lazyval
    def next_fiscal_quarter_loader(self):
        return self._next_event_value_loader(
            self.dataset.next_fiscal_quarter,
            FISCAL_QUARTER_FIELD_NAME,
        )

    @lazyval
    def previous_fiscal_quarter_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_fiscal_quarter,
            FISCAL_QUARTER_FIELD_NAME,
        )

    @lazyval
    def next_high_loader(self):
        return self._next_event_value_loader(
            self.dataset.next_high,
            HIGH_FIELD_NAME,
        )

    @lazyval
    def previous_high_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_high,
            HIGH_FIELD_NAME,
        )

    @lazyval
    def next_mean_loader(self):
        return self._next_event_value_loader(
            self.dataset.next_mean,
            MEAN_FIELD_NAME,
        )

    @lazyval
    def previous_mean_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_mean,
            MEAN_FIELD_NAME,
        )

    @lazyval
    def next_fiscal_year_loader(self):
        return self._next_event_value_loader(
            self.dataset.next_fiscal_year,
            FISCAL_YEAR_FIELD_NAME,
        )

    @lazyval
    def previous_fiscal_year_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_fiscal_year,
            FISCAL_YEAR_FIELD_NAME,
        )

    @lazyval
    def next_low_loader(self):
        return self._next_event_value_loader(
            self.dataset.next_low,
            LOW_FIELD_NAME,
        )

    @lazyval
    def previous_low_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_low,
            LOW_FIELD_NAME,
        )

    @lazyval
    def previous_actual_value_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_actual_value,
            ACTUAL_VALUE_FIELD_NAME,
        )
