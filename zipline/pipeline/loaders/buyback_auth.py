"""
Reference implementation for buyback auth loaders.
"""

from ..data import BuybackAuthorizations
from .events import EventsLoader
from zipline.pipeline.common import (
    BUYBACK_AMOUNT_FIELD_NAME,
    BUYBACK_ANNOUNCEMENT_FIELD_NAME,
    BUYBACK_TYPE_FIELD_NAME,
    BUYBACK_UNIT_FIELD_NAME
)
from zipline.utils.memoize import lazyval


class BuybackAuthorizationsLoader(EventsLoader):
    """
    Reference loader for
    :class:`zipline.pipeline.data.BuybackAuthorizations`.

    events_by_sid: dict[sid -> pd.DataFrame(knowledge date,
    event date, buyback amount, buyback unit, buyback type)]

    """
    expected_cols = frozenset([BUYBACK_ANNOUNCEMENT_FIELD_NAME,
                               BUYBACK_AMOUNT_FIELD_NAME,
                               BUYBACK_UNIT_FIELD_NAME,
                               BUYBACK_TYPE_FIELD_NAME])

    event_date_col = BUYBACK_ANNOUNCEMENT_FIELD_NAME

    def __init__(self,
                 all_dates,
                 events_by_sid,
                 infer_timestamps=False,
                 dataset=BuybackAuthorizations):
        super(BuybackAuthorizationsLoader, self).__init__(
            all_dates,
            events_by_sid,
            infer_timestamps=infer_timestamps,
            dataset=dataset,
        )

    @lazyval
    def previous_amount_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_amount,
            BUYBACK_AMOUNT_FIELD_NAME
        )

    @lazyval
    def previous_date_loader(self):
        return self._previous_event_date_loader(
            self.dataset.previous_date,
        )

    @lazyval
    def previous_unit_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_unit,
            BUYBACK_UNIT_FIELD_NAME,
        )

    @lazyval
    def previous_type_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_type,
            BUYBACK_TYPE_FIELD_NAME,
        )
