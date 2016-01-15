"""
Reference implementation for EarningsCalendar loaders.
"""

from ..data.buyback_auth import CashBuybackAuthorizations, \
    ShareBuybackAuthorizations
from events import EventsLoader
from zipline.utils.memoize import lazyval


BUYBACK_ANNOUNCEMENT_FIELD_NAME = 'buyback_dates'
SHARE_COUNT_FIELD_NAME = 'share_counts'
VALUE_FIELD_NAME = 'values'


# TODO:  split into 2 datasets - or just think about how to generalize since
# we will often have cases where we have a knowledge date and, optionally,
# a value for that event; having no value (like earnings) is a special case.
class CashBuybackAuthorizationsLoader(EventsLoader):
    """
    Reference loader for
    :class:`zipline.pipeline.data.earnings.BuybackAuthorizations`.

    Does not currently support adjustments to the dates of known buyback
    authorizations.

    events_by_sid: dict[sid -> pd.DataFrame(knowledge date,
    event date, value)]

    """

    def __init__(self,
                 all_dates,
                 events_by_sid,
                 infer_timestamps=False,
                 dataset=CashBuybackAuthorizations):
        super(CashBuybackAuthorizationsLoader, self).__init__(
            all_dates,
            events_by_sid,
            infer_timestamps=infer_timestamps,
            dataset=dataset
        )


    def get_loader(self, column):
        """dispatch to the loader for ``column``.
        """
        if column is self.dataset.previous_value:
            return self.previous_buyback_value_loader
        elif column is self.dataset.previous_announcement_date:
            return self.previous_event_date_loader
        else:
            raise ValueError("Don't know how to load column '%s'." % column)


    @lazyval
    def previous_buyback_value_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_value,
            BUYBACK_ANNOUNCEMENT_FIELD_NAME,
            VALUE_FIELD_NAME
        )

    @lazyval
    def previous_event_date_loader(self):
        return self._previous_event_date_loader(
            self.dataset.previous_announcement_date,
            BUYBACK_ANNOUNCEMENT_FIELD_NAME,
        )


class ShareBuybackAuthorizationsLoader(EventsLoader):
    """
    Reference loader for
    :class:`zipline.pipeline.data.earnings.BuybackAuthorizations`.

    Does not currently support adjustments to the dates of known buyback
    authorizations.

    events_by_sid: dict[sid -> pd.DataFrame(knowledge date,
    event date, value)]

    """

    def __init__(self,
                 all_dates,
                 events_by_sid,
                 infer_timestamps=False,
                 dataset=ShareBuybackAuthorizations):
        super(ShareBuybackAuthorizationsLoader, self).__init__(
            all_dates,
            events_by_sid,
            infer_timestamps=infer_timestamps,
            dataset=dataset
        )


    def get_loader(self, column):
        """dispatch to the loader for ``column``.
        """
        if column is self.dataset.previous_share_count:
            return self.previous_buyback_share_count_loader
        elif column is self.dataset.previous_announcement_date:
            return self.previous_event_date_loader
        else:
            raise ValueError("Don't know how to load column '%s'." % column)


    @lazyval
    def previous_buyback_share_count_loader(self):
        return self._previous_event_value_loader(
            self.dataset.previous_share_count,
            BUYBACK_ANNOUNCEMENT_FIELD_NAME,
            SHARE_COUNT_FIELD_NAME
        )

    @lazyval
    def previous_event_date_loader(self):
        return self._previous_event_date_loader(
            self.dataset.previous_announcement_date,
            BUYBACK_ANNOUNCEMENT_FIELD_NAME,
        )
