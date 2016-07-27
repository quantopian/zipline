from datetime import time
from unittest import TestCase
import pandas as pd
from zipline.gens.sim_engine import (
    MinuteSimulationClock,
    SESSION_START,
    BEFORE_TRADING_START_BAR,
    BAR,
    SESSION_END
)

from zipline.utils.calendars import get_calendar
from zipline.utils.calendars.trading_calendar import days_at_time


class TestClock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nyse_calendar = get_calendar("NYSE")

        # july 15 is friday, so there are 3 sessions in this range (15, 18, 19)
        cls.sessions = cls.nyse_calendar.sessions_in_range(
            pd.Timestamp("2016-07-15"),
            pd.Timestamp("2016-07-19")
        )

        trading_o_and_c = cls.nyse_calendar.schedule.ix[cls.sessions]
        cls.opens = trading_o_and_c['market_open']
        cls.closes = trading_o_and_c['market_close']

    def test_bts_before_session(self):
        clock = MinuteSimulationClock(
            self.sessions,
            self.opens,
            self.closes,
            days_at_time(self.sessions, time(6, 17), "US/Eastern"),
            False
        )

        all_events = list(clock)

        def _check_session_bts_first(session_label, events, bts_dt):
            minutes = self.nyse_calendar.minutes_for_session(session_label)

            self.assertEqual(393, len(events))

            self.assertEqual(events[0], (session_label, SESSION_START))
            self.assertEqual(events[1], (bts_dt, BEFORE_TRADING_START_BAR))
            for i in range(2, 392):
                self.assertEqual(events[i], (minutes[i - 2], BAR))
            self.assertEqual(events[392], (minutes[-1], SESSION_END))

        _check_session_bts_first(
            self.sessions[0],
            all_events[0:393],
            pd.Timestamp("2016-07-15 6:17", tz='US/Eastern')
        )

        _check_session_bts_first(
            self.sessions[1],
            all_events[393:786],
            pd.Timestamp("2016-07-18 6:17", tz='US/Eastern')
        )

        _check_session_bts_first(
            self.sessions[2],
            all_events[786:],
            pd.Timestamp("2016-07-19 6:17", tz='US/Eastern')
        )

    def test_bts_during_session(self):
        self.verify_bts_during_session(
            time(11, 45), [
                pd.Timestamp("2016-07-15 11:45", tz='US/Eastern'),
                pd.Timestamp("2016-07-18 11:45", tz='US/Eastern'),
                pd.Timestamp("2016-07-19 11:45", tz='US/Eastern')
            ],
            135
        )

    def test_bts_on_first_minute(self):
        self.verify_bts_during_session(
            time(9, 30), [
                pd.Timestamp("2016-07-15 9:30", tz='US/Eastern'),
                pd.Timestamp("2016-07-18 9:30", tz='US/Eastern'),
                pd.Timestamp("2016-07-19 9:30", tz='US/Eastern')
            ],
            1
        )

    def test_bts_on_last_minute(self):
        self.verify_bts_during_session(
            time(16, 00), [
                pd.Timestamp("2016-07-15 16:00", tz='US/Eastern'),
                pd.Timestamp("2016-07-18 16:00", tz='US/Eastern'),
                pd.Timestamp("2016-07-19 16:00", tz='US/Eastern')
            ],
            390
        )

    def verify_bts_during_session(self, bts_time, bts_session_times, bts_idx):
        def _check_session_bts_during(session_label, events, bts_dt):
            minutes = self.nyse_calendar.minutes_for_session(session_label)

            self.assertEqual(393, len(events))

            self.assertEqual(events[0], (session_label, SESSION_START))

            for i in range(1, bts_idx):
                self.assertEqual(events[i], (minutes[i - 1], BAR))

            self.assertEqual(
                events[bts_idx],
                (bts_dt, BEFORE_TRADING_START_BAR)
            )

            for i in range(bts_idx + 1, 391):
                self.assertEqual(events[i], (minutes[i - 2], BAR))

            self.assertEqual(events[392], (minutes[-1], SESSION_END))

        clock = MinuteSimulationClock(
            self.sessions,
            self.opens,
            self.closes,
            days_at_time(self.sessions, bts_time, "US/Eastern"),
            False
        )

        all_events = list(clock)

        _check_session_bts_during(
            self.sessions[0],
            all_events[0:393],
            bts_session_times[0]
        )

        _check_session_bts_during(
            self.sessions[1],
            all_events[393:786],
            bts_session_times[1]
        )

        _check_session_bts_during(
            self.sessions[2],
            all_events[786:],
            bts_session_times[2]
        )

    def test_bts_after_session(self):
        clock = MinuteSimulationClock(
            self.sessions,
            self.opens,
            self.closes,
            days_at_time(self.sessions, time(19, 5), "US/Eastern"),
            False
        )

        all_events = list(clock)

        # since 19:05 Eastern is after the NYSE is closed, we don't emit
        # BEFORE_TRADING_START.  therefore, each day has SESSION_START,
        # 390 BARs, and then SESSION_END

        def _check_session_bts_after(session_label, events):
            minutes = self.nyse_calendar.minutes_for_session(session_label)

            self.assertEqual(392, len(events))
            self.assertEqual(events[0], (session_label, SESSION_START))

            for i in range(1, 391):
                self.assertEqual(events[i], (minutes[i - 1], BAR))

            self.assertEqual(events[-1], (minutes[389], SESSION_END))

        for i in range(0, 2):
            _check_session_bts_after(
                self.sessions[i],
                all_events[(i * 392): ((i + 1) * 392)]
            )
