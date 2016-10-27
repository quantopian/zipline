"""
Tests for TradingCalendarDispatcher.
"""
from zipline.errors import (
    CalendarNameCollision,
    CyclicCalendarAlias,
    InvalidCalendarName,
)
from zipline.testing import ZiplineTestCase
from zipline.utils.calendars.calendar_utils import TradingCalendarDispatcher
from zipline.utils.calendars.exchange_calendar_ice import ICEExchangeCalendar


class CalendarAliasTestCase(ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(CalendarAliasTestCase, cls).init_class_fixtures()
        # Make a calendar once so that we don't spend time in every test
        # instantiating calendars.
        cls.dispatcher_kwargs = dict(
            calendars={'ICE': ICEExchangeCalendar()},
            calendar_factories={},
            aliases={
                'ICE_ALIAS': 'ICE',
                'ICE_ALIAS_ALIAS': 'ICE_ALIAS',
            },
        )

    def init_instance_fixtures(self):
        super(CalendarAliasTestCase, self).init_instance_fixtures()
        self.dispatcher = TradingCalendarDispatcher(
            # Make copies here so that tests that mutate the dispatcher dicts
            # are isolated from one another.
            **{k: v.copy() for k, v in self.dispatcher_kwargs.items()}
        )

    def test_follow_alias_chain(self):
        self.assertIs(
            self.dispatcher.get_calendar('ICE_ALIAS'),
            self.dispatcher.get_calendar('ICE'),
        )
        self.assertIs(
            self.dispatcher.get_calendar('ICE_ALIAS_ALIAS'),
            self.dispatcher.get_calendar('ICE'),
        )

    def test_add_new_aliases(self):
        with self.assertRaises(InvalidCalendarName):
            self.dispatcher.get_calendar('NOT_ICE')

        self.dispatcher.register_calendar_alias('NOT_ICE', 'ICE')

        self.assertIs(
            self.dispatcher.get_calendar('NOT_ICE'),
            self.dispatcher.get_calendar('ICE'),
        )

        self.dispatcher.register_calendar_alias(
            'ICE_ALIAS_ALIAS_ALIAS',
            'ICE_ALIAS_ALIAS'
        )
        self.assertIs(
            self.dispatcher.get_calendar('ICE_ALIAS_ALIAS_ALIAS'),
            self.dispatcher.get_calendar('ICE'),
        )

    def test_remove_aliases(self):
        self.dispatcher.deregister_calendar('ICE_ALIAS_ALIAS')
        with self.assertRaises(InvalidCalendarName):
            self.dispatcher.get_calendar('ICE_ALIAS_ALIAS')

    def test_reject_alias_that_already_exists(self):
        with self.assertRaises(CalendarNameCollision):
            self.dispatcher.register_calendar_alias('ICE', 'NOT_ICE')

        with self.assertRaises(CalendarNameCollision):
            self.dispatcher.register_calendar_alias('ICE_ALIAS', 'NOT_ICE')

    def test_allow_alias_override_with_force(self):
        self.dispatcher.register_calendar_alias('ICE', 'NOT_ICE', force=True)
        with self.assertRaises(InvalidCalendarName):
            self.dispatcher.get_calendar('ICE')

    def test_reject_cyclic_aliases(self):
        add_alias = self.dispatcher.register_calendar_alias

        add_alias('A', 'B')
        add_alias('B', 'C')

        with self.assertRaises(CyclicCalendarAlias) as e:
            add_alias('C', 'A')

        expected = "Cycle in calendar aliases: ['C' -> 'A' -> 'B' -> 'C']"
        self.assertEqual(str(e.exception), expected)
