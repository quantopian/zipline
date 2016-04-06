from unittest import TestCase

from pandas import (
    Timestamp,
    date_range,
)

from zipline.utils.calendars import (
    get_calendar,
    ExchangeTradingSchedule,
    normalize_date,
)


class TestExchangeTradingSchedule(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nyse_cal = get_calendar('NYSE')
        cls.nyse_exchange_schedule = ExchangeTradingSchedule(cal=cls.nyse_cal)

    def test_nyse_data_availability_time(self):
        """
        Ensure that the NYSE schedule's data availability time is the market
        open.
        """
        # This is a time on the day after Thanksgiving when the market was open
        test_dt = Timestamp('11/23/2012 11:00AM', tz='EST')
        test_date = normalize_date(test_dt)
        desired_data_time = Timestamp('11/23/2012 9:31AM', tz='EST')

        # Get the data availability time from the NYSE schedule
        data_time = self.nyse_exchange_schedule.data_availability_time(
            date=test_date
        )

        # Check the schedule answer against the hard-coded answer
        self.assertEqual(data_time, desired_data_time,
                         "Data availability time is not the market open")

    def test_nyse_execution_time(self):
        """
        Runs a series of times through both the NYSE calendar and NYSE
        schedule, ensuring that the schedule and calendar agree.
        """
        # Get all of the minutes in a 24-hour day
        start_range = Timestamp('11/23/2012 12:00AM', tz='EST')
        end_range = Timestamp('11/23/2012 11:59PM', tz='EST')
        time_range = date_range(start_range, end_range, freq='Min')

        for dt in time_range:
            cal_open = self.nyse_cal.is_open_on_minute(dt)
            sched_exec = self.nyse_exchange_schedule.is_executing_on_minute(dt)
            self.assertEqual(
                cal_open, sched_exec,
                "Mismatch between schedule: %s and calendar: %s at time %s"
                % (cal_open, sched_exec, dt)
            )
