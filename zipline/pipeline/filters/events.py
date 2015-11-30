"""
Filters for handling recent or upcoming events.
"""
from zipline.pipeline.data.earnings import EarningsCalendar
from .filter import CustomFilter


class HasEarningsToday(CustomFilter):
    """
    Filter indicating whether an asset has an earnings announcement on the
    current day.
    """
    inputs = [EarningsCalendar.next_announcement]
    window_length = 1

    def compute(self, today, assets, out, next_announce_date):
        out[:] = (today.value == next_announce_date)
