"""
Tests for live trading.
"""
from unittest import TestCase
import pandas as pd
from datetime import time
from collections import defaultdict

from itertools import izip_longest

from mock import patch

from zipline.gens.realtimeclock import RealtimeClock
from zipline.gens.sim_engine import MinuteSimulationClock

from zipline.utils.calendars import get_calendar
from zipline.utils.calendars.trading_calendar import days_at_time


class TestRealtimeClock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nyse_calendar = get_calendar("NYSE")

        cls.sessions = cls.nyse_calendar.sessions_in_range(
            pd.Timestamp("2017-04-20"),
            pd.Timestamp("2017-04-20")
        )

        trading_o_and_c = cls.nyse_calendar.schedule.ix[cls.sessions]
        cls.opens = trading_o_and_c['market_open']
        cls.closes = trading_o_and_c['market_close']

    def setUp(self):
        self.internal_clock = None
        self.events = defaultdict(list)

    def advance_clock(self, x):
        """Mock function for sleep. Advances the internal clock by 1 min"""
        # The internal clock advance time must be 1 minute to match with MinutesSimulationClock
        self.internal_clock += pd.Timedelta('1 min')

    def get_clock(self, arg, *args, **kwargs):
        """Mock function for pandas.to_datetime which is used to query the current time in RealtimeClock"""
        assert arg == "now"
        return self.internal_clock

    def test_crosscheck_realtimeclock_with_minutesimulationclock(self):
        """Tests that RealtimeClock behaves like MinuteSimulationClock"""
        for minute_emission in (False, True):
            # MinuteSimulationClock also relies on to_datetime, shall not be created in the patch block
            msc = MinuteSimulationClock(
                self.sessions,
                self.opens,
                self.closes,
                days_at_time(self.sessions, time(8, 45), "US/Eastern"),
                minute_emission
            )
            msc_events = list(msc)

            with patch('zipline.gens.realtimeclock.pd.to_datetime') as to_dt, \
                    patch('zipline.gens.realtimeclock.sleep') as sleep:
                rtc = iter(RealtimeClock(
                    self.sessions,
                    self.opens,
                    self.closes,
                    days_at_time(self.sessions, time(8, 45), "US/Eastern"),
                    minute_emission
                ))
                self.internal_clock = pd.Timestamp("2017-04-20 00:00", tz='UTC')
                to_dt.side_effect = self.get_clock
                sleep.side_effect = self.advance_clock

                rtc_events = list(rtc)

            for rtc_event, msc_event in izip_longest(rtc_events, msc_events):
                self.assertEquals(rtc_event, msc_event)

            self.assertEquals(len(rtc_events), len(msc_events))

    def test_time_skew(self):
        """Tests that RealtimeClock's time_skew parameter behaves as expected"""
        for time_skew in (pd.Timedelta("2 hour"), pd.Timedelta("-120 sec")):
            with patch('zipline.gens.realtimeclock.pd.to_datetime') as to_dt, \
                 patch('zipline.gens.realtimeclock.sleep') as sleep:
                clock = RealtimeClock(
                    self.sessions,
                    self.opens,
                    self.closes,
                    days_at_time(self.sessions, time(11, 31), "US/Eastern"),
                    False,
                    time_skew
                )
                to_dt.side_effect = self.get_clock
                sleep.side_effect = self.advance_clock
                start_time = pd.Timestamp("2017-04-20 15:31", tz='UTC')
                self.internal_clock = start_time

                events = list(clock)

                # Event 0 is SESSION_START which always happens at 00:00.
                ts, event_type = events[1]
                self.assertEquals(ts, start_time + time_skew)

    def test_midday_start(self):
        """Tests that RealtimeClock is able to execute if started mid-day"""
        msc = MinuteSimulationClock(
            self.sessions,
            self.opens,
            self.closes,
            days_at_time(self.sessions, time(8, 45), "US/Eastern"),
            False
        )
        msc_events = list(msc)

        with patch('zipline.gens.realtimeclock.pd.to_datetime') as to_dt, \
             patch('zipline.gens.realtimeclock.sleep') as sleep:
            rtc = RealtimeClock(
                self.sessions,
                self.opens,
                self.closes,
                days_at_time(self.sessions, time(8, 45), "US/Eastern"),
                False
            )

            to_dt.side_effect = self.get_clock
            sleep.side_effect = self.advance_clock
            self.internal_clock = pd.Timestamp("2017-04-20 15:00", tz='UTC')

            rtc_events = list(rtc)

        # Count the mid-day position in the MinuteSimulationClock's events:
        # [2017-05-07 16:42:35.531282] INFO: Trade Simulation: Processing tick: 2017-04-20 00:00:00+00:00 - 1 (SESSION_START)
        # [2017-05-07 16:42:35.531622] INFO: Trade Simulation: Processing tick: 2017-04-20 12:45:00+00:00 - 4 (BEFORE_TRADING_START)
        # [2017-05-07 16:42:35.532042] INFO: Trade Simulation: Processing tick: 2017-04-20 13:31:00+00:00 - 0 (BAR)
        msc_midday_position = 2 + 90 - 1 #  2 = SESSION_START + BEFORE_TRADING_START, 90-1 minutes
        self.assertEquals(rtc_events[0], msc_events[0]) # Session start bar
        # BEFORE_TRADING_START is not fired as we're in mid-day
        self.assertEquals(rtc_events[1:], msc_events[msc_midday_position:])
