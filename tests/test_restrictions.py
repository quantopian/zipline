import pandas as pd
from pandas.util.testing import assert_series_equal
from nose_parameterized import parameterized
from six import iteritems
from functools import partial

from zipline.finance.restrictions import (
    RESTRICTION_STATES,
    Restriction,
    HistoricalRestrictions,
    StaticRestrictions,
    NoopRestrictions,
)

from zipline.testing.fixtures import (
    WithDataPortal,
    ZiplineTestCase,
)

str_to_ts = lambda dt_str: pd.Timestamp(dt_str, tz='UTC')
FROZEN = RESTRICTION_STATES.FROZEN
ALLOWED = RESTRICTION_STATES.ALLOWED
MINUTE = pd.Timedelta(minutes=1)


class RestrictionsTestCase(WithDataPortal, ZiplineTestCase):

    ASSET_FINDER_EQUITY_SIDS = 1, 2, 3

    @classmethod
    def init_class_fixtures(cls):
        super(RestrictionsTestCase, cls).init_class_fixtures()
        cls.ASSET1 = cls.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.asset_finder.retrieve_asset(2)
        cls.ASSET3 = cls.asset_finder.retrieve_asset(3)

    def assert_is_restricted(self, rl, asset, dt):
        self.assertTrue(rl.is_restricted(asset, dt))

    def assert_not_restricted(self, rl, asset, dt):
        self.assertFalse(rl.is_restricted(asset, dt))

    def assert_vectorized_results(self, rl, expected, dt):
        assert_series_equal(
            rl.is_restricted([self.ASSET1, self.ASSET2, self.ASSET3], dt),
            pd.Series(
                index=pd.Index([self.ASSET1, self.ASSET2, self.ASSET3]),
                data=expected
            )
        )

    @parameterized.expand([
        ('_'.join([timing, ordering]),
         timing == 'intraday',
         ordering == 'ordered')
        for timing in ['intraday', 'interday']
        for ordering in ['ordered', 'unordered']
    ])
    def test_historical_restrictions(self, name, is_intraday, is_ordered):
        """
        Test historical restrictions for both interday and intraday
        restrictions, as well as restrictions defined in/not in order, for both
        single- and multi-asset queries
        """

        hour_of_day = ' 15:00' if is_intraday else ''

        if is_ordered:
            restriction_dates = {
                self.ASSET1: [
                    (str_to_ts('2011-01-04' + hour_of_day), FROZEN),
                    (str_to_ts('2011-01-05' + hour_of_day), ALLOWED),
                    (str_to_ts('2011-01-06' + hour_of_day), FROZEN),
                ],
                self.ASSET2: [
                    (str_to_ts('2011-01-05' + hour_of_day), FROZEN),
                    (str_to_ts('2011-01-06' + hour_of_day), ALLOWED),
                    (str_to_ts('2011-01-07' + hour_of_day), FROZEN),
                ],
            }
        else:
            restriction_dates = {
                self.ASSET1: [
                    (str_to_ts('2011-01-05' + hour_of_day), ALLOWED),
                    (str_to_ts('2011-01-06' + hour_of_day), FROZEN),
                    (str_to_ts('2011-01-04' + hour_of_day), FROZEN),
                ],
                self.ASSET2: [
                    (str_to_ts('2011-01-06' + hour_of_day), ALLOWED),
                    (str_to_ts('2011-01-05' + hour_of_day), FROZEN),
                    (str_to_ts('2011-01-07' + hour_of_day), FROZEN),
                ],
            }

        restrictions = sum([
            [Restriction(asset, info[0], info[1]) for info in r_history]
            for asset, r_history in iteritems(restriction_dates)
        ], [])
        rl = HistoricalRestrictions(restrictions)

        assert_not_restricted = partial(self.assert_not_restricted, rl)
        assert_is_restricted = partial(self.assert_is_restricted, rl)
        assert_vectorized_results = partial(self.assert_vectorized_results, rl)

        for asset, r_history in iteritems(restriction_dates):
            dts = sorted([info[0] for info in r_history])

            # Not restricted until on or after the freeze
            assert_not_restricted(asset, dts[0] - MINUTE)
            assert_is_restricted(asset, dts[0])
            assert_is_restricted(asset, dts[0] + MINUTE)

            # Unrestricted on or after the unfreeze
            assert_is_restricted(asset, dts[1] - MINUTE)
            assert_not_restricted(asset, dts[1])
            assert_not_restricted(asset, dts[1] + MINUTE)

            # Restricted again on or after the freeze
            assert_not_restricted(asset, dts[2] - MINUTE)
            assert_is_restricted(asset, dts[2])
            assert_is_restricted(asset, dts[2] + MINUTE)
            # Should stay restricted for the rest of time
            assert_is_restricted(asset, dts[2] + MINUTE * 1000000)

        dts = [str_to_ts(ts + hour_of_day) for ts in ['2011-01-04',
                                                      '2011-01-05',
                                                      '2011-01-06',
                                                      '2011-01-07']]

        # Expected results for [self.ASSET1, self.ASSET2, self.ASSET3],
        # ASSET3 is always False as it has no defined restrictions

        # 01/04 XX:00 ASSET1: ALLOWED --> FROZEN; ASSET2: ALLOWED
        assert_vectorized_results([False, False, False], dts[0] - MINUTE)
        assert_vectorized_results([True, False, False], dts[0])
        assert_vectorized_results([True, False, False], dts[0] + MINUTE)

        # 01/05 XX:00 ASSET1: FROZEN --> ALLOWED; ASSET2: ALLOWED --> FROZEN
        assert_vectorized_results([True, False, False], dts[1] - MINUTE)
        assert_vectorized_results([False, True, False], dts[1])
        assert_vectorized_results([False, True, False], dts[1] + MINUTE)

        # 01/06 XX:00 ASSET1: ALLOWED --> FROZEN; ASSET2: FROZEN --> ALLOWED
        assert_vectorized_results([False, True, False], dts[2] - MINUTE)
        assert_vectorized_results([True, False, False], dts[2])
        assert_vectorized_results([True, False, False], dts[2] + MINUTE)

        # 01/07 XX:00 ASSET1: FROZEN; ASSET2: ALLOWED --> FROZEN
        assert_vectorized_results([True, False, False], dts[3] - MINUTE)
        assert_vectorized_results([True, True, False], dts[3])
        assert_vectorized_results([True, True, False], dts[3] + MINUTE)
        # Should stay restricted for the rest of time
        assert_vectorized_results(
            [True, True, False],
            dts[3] + MINUTE * 10000000
        )

    def test_historical_restrictions_consecutive_states(self):
        """
        Test that defining redundant consecutive restrictions still works
        """
        rl = HistoricalRestrictions([
            Restriction(self.ASSET1, str_to_ts('2011-01-04'), ALLOWED),
            Restriction(self.ASSET1, str_to_ts('2011-01-05'), ALLOWED),
            Restriction(self.ASSET1, str_to_ts('2011-01-06'), FROZEN),
            Restriction(self.ASSET1, str_to_ts('2011-01-07'), FROZEN),
        ])

        assert_not_restricted = partial(self.assert_not_restricted, rl)
        assert_is_restricted = partial(self.assert_is_restricted, rl)

        # (implicit) ALLOWED --> ALLOWED
        assert_not_restricted(self.ASSET1, str_to_ts('2011-01-04') - MINUTE)
        assert_not_restricted(self.ASSET1, str_to_ts('2011-01-04'))
        assert_not_restricted(self.ASSET1, str_to_ts('2011-01-04') + MINUTE)

        # ALLOWED --> ALLOWED
        assert_not_restricted(self.ASSET1, str_to_ts('2011-01-05') - MINUTE)
        assert_not_restricted(self.ASSET1, str_to_ts('2011-01-05'))
        assert_not_restricted(self.ASSET1, str_to_ts('2011-01-05') + MINUTE)

        # ALLOWED --> FROZEN
        assert_not_restricted(self.ASSET1, str_to_ts('2011-01-06') - MINUTE)
        assert_is_restricted(self.ASSET1, str_to_ts('2011-01-06'))
        assert_is_restricted(self.ASSET1, str_to_ts('2011-01-06') + MINUTE)

        # FROZEN --> FROZEN
        assert_is_restricted(self.ASSET1, str_to_ts('2011-01-07') - MINUTE)
        assert_is_restricted(self.ASSET1, str_to_ts('2011-01-07'))
        assert_is_restricted(self.ASSET1, str_to_ts('2011-01-07') + MINUTE)

    def test_static_restrictions(self):
        """
        Test single- and multi-asset queries on static restrictions
        """

        restricted_a1 = self.ASSET1
        restricted_a2 = self.ASSET2
        unrestricted_a3 = self.ASSET3

        rl = StaticRestrictions([restricted_a1, restricted_a2])
        assert_not_restricted = partial(self.assert_not_restricted, rl)
        assert_is_restricted = partial(self.assert_is_restricted, rl)
        assert_vectorized_results = partial(self.assert_vectorized_results, rl)

        for dt in [str_to_ts(dt_str) for dt_str in ('2011-01-03',
                                                    '2011-01-04',
                                                    '2020-01-04')]:
            assert_is_restricted(restricted_a1, dt)
            assert_is_restricted(restricted_a2, dt)
            assert_not_restricted(unrestricted_a3, dt)

            assert_vectorized_results([True, True, False], dt)

    def test_noop_restrictions(self):
        """
        Test single- and multi-asset queries on no-op restrictions
        """

        rl = NoopRestrictions()
        assert_not_restricted = partial(self.assert_not_restricted, rl)
        assert_vectorized_results = partial(self.assert_vectorized_results, rl)

        for dt in [str_to_ts(dt_str) for dt_str in ('2011-01-03',
                                                    '2011-01-04',
                                                    '2020-01-04')]:
            assert_not_restricted(self.ASSET1, dt)
            assert_not_restricted(self.ASSET2, dt)
            assert_not_restricted(self.ASSET3, dt)
            assert_vectorized_results([False, False, False], dt)
