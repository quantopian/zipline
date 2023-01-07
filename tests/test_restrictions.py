import pandas as pd
from pandas.testing import assert_series_equal
from functools import partial

from toolz import groupby

from zipline.finance.asset_restrictions import (
    RESTRICTION_STATES,
    Restriction,
    HistoricalRestrictions,
    StaticRestrictions,
    SecurityListRestrictions,
    NoRestrictions,
    _UnionRestrictions,
)

from zipline.testing import parameter_space
from zipline.testing.fixtures import (
    WithDataPortal,
    ZiplineTestCase,
)


def str_to_ts(dt_str):
    return pd.Timestamp(dt_str, tz="UTC")


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
        cls.ALL_ASSETS = [cls.ASSET1, cls.ASSET2, cls.ASSET3]

    def assert_is_restricted(self, rl, asset, dt):
        assert rl.is_restricted(asset, dt)

    def assert_not_restricted(self, rl, asset, dt):
        assert not rl.is_restricted(asset, dt)

    def assert_all_restrictions(self, rl, expected, dt):
        self.assert_many_restrictions(rl, self.ALL_ASSETS, expected, dt)

    def assert_many_restrictions(self, rl, assets, expected, dt):
        assert_series_equal(
            rl.is_restricted(assets, dt),
            pd.Series(index=pd.Index(assets), data=expected),
        )

    @parameter_space(
        date_offset=(
            pd.Timedelta(0),
            pd.Timedelta("1 minute"),
            pd.Timedelta("15 hours 5 minutes"),
        ),
        restriction_order=(
            list(range(6)),  # Keep restrictions in order.
            [0, 2, 1, 3, 5, 4],  # Re-order within asset.
            [0, 3, 1, 4, 2, 5],  # Scramble assets, maintain per-asset order.
            [0, 5, 2, 3, 1, 4],  # Scramble assets and per-asset order.
        ),
        __fail_fast=True,
    )
    def test_historical_restrictions(self, date_offset, restriction_order):
        """Test historical restrictions for both interday and intraday
        restrictions, as well as restrictions defined in/not in order, for both
        single- and multi-asset queries
        """

        def rdate(s):
            """Convert a date string into a restriction for that date."""
            # Add date_offset to check that we handle intraday changes.
            return str_to_ts(s) + date_offset

        base_restrictions = [
            Restriction(self.ASSET1, rdate("2011-01-04"), FROZEN),
            Restriction(self.ASSET1, rdate("2011-01-05"), ALLOWED),
            Restriction(self.ASSET1, rdate("2011-01-06"), FROZEN),
            Restriction(self.ASSET2, rdate("2011-01-05"), FROZEN),
            Restriction(self.ASSET2, rdate("2011-01-06"), ALLOWED),
            Restriction(self.ASSET2, rdate("2011-01-07"), FROZEN),
        ]
        # Scramble the restrictions based on restriction_order to check that we
        # don't depend on the order in which restrictions are provided to us.
        all_restrictions = [base_restrictions[i] for i in restriction_order]

        restrictions_by_asset = groupby(lambda r: r.asset, all_restrictions)

        rl = HistoricalRestrictions(all_restrictions)
        assert_not_restricted = partial(self.assert_not_restricted, rl)
        assert_is_restricted = partial(self.assert_is_restricted, rl)
        assert_all_restrictions = partial(self.assert_all_restrictions, rl)

        # Check individual restrictions.
        for asset, r_history in restrictions_by_asset.items():
            freeze_dt, unfreeze_dt, re_freeze_dt = sorted(
                [r.effective_date for r in r_history]
            )

            # Starts implicitly unrestricted. Restricted on or after the freeze
            assert_not_restricted(asset, freeze_dt - MINUTE)
            assert_is_restricted(asset, freeze_dt)
            assert_is_restricted(asset, freeze_dt + MINUTE)

            # Unrestricted on or after the unfreeze
            assert_is_restricted(asset, unfreeze_dt - MINUTE)
            assert_not_restricted(asset, unfreeze_dt)
            assert_not_restricted(asset, unfreeze_dt + MINUTE)

            # Restricted again on or after the freeze
            assert_not_restricted(asset, re_freeze_dt - MINUTE)
            assert_is_restricted(asset, re_freeze_dt)
            assert_is_restricted(asset, re_freeze_dt + MINUTE)

            # Should stay restricted for the rest of time
            assert_is_restricted(asset, re_freeze_dt + MINUTE * 1000000)

        # Check vectorized restrictions.
        # Expected results for [self.ASSET1, self.ASSET2, self.ASSET3],
        # ASSET3 is always False as it has no defined restrictions

        # 01/04 XX:00 ASSET1: ALLOWED --> FROZEN; ASSET2: ALLOWED
        d0 = rdate("2011-01-04")
        assert_all_restrictions([False, False, False], d0 - MINUTE)
        assert_all_restrictions([True, False, False], d0)
        assert_all_restrictions([True, False, False], d0 + MINUTE)

        # 01/05 XX:00 ASSET1: FROZEN --> ALLOWED; ASSET2: ALLOWED --> FROZEN
        d1 = rdate("2011-01-05")
        assert_all_restrictions([True, False, False], d1 - MINUTE)
        assert_all_restrictions([False, True, False], d1)
        assert_all_restrictions([False, True, False], d1 + MINUTE)

        # 01/06 XX:00 ASSET1: ALLOWED --> FROZEN; ASSET2: FROZEN --> ALLOWED
        d2 = rdate("2011-01-06")
        assert_all_restrictions([False, True, False], d2 - MINUTE)
        assert_all_restrictions([True, False, False], d2)
        assert_all_restrictions([True, False, False], d2 + MINUTE)

        # 01/07 XX:00 ASSET1: FROZEN; ASSET2: ALLOWED --> FROZEN
        d3 = rdate("2011-01-07")
        assert_all_restrictions([True, False, False], d3 - MINUTE)
        assert_all_restrictions([True, True, False], d3)
        assert_all_restrictions([True, True, False], d3 + MINUTE)

        # Should stay restricted for the rest of time
        assert_all_restrictions([True, True, False], d3 + (MINUTE * 10000000))

    def test_historical_restrictions_consecutive_states(self):
        """Test that defining redundant consecutive restrictions still works"""

        rl = HistoricalRestrictions(
            [
                Restriction(self.ASSET1, str_to_ts("2011-01-04"), ALLOWED),
                Restriction(self.ASSET1, str_to_ts("2011-01-05"), ALLOWED),
                Restriction(self.ASSET1, str_to_ts("2011-01-06"), FROZEN),
                Restriction(self.ASSET1, str_to_ts("2011-01-07"), FROZEN),
            ]
        )

        assert_not_restricted = partial(self.assert_not_restricted, rl)
        assert_is_restricted = partial(self.assert_is_restricted, rl)

        # (implicit) ALLOWED --> ALLOWED
        assert_not_restricted(self.ASSET1, str_to_ts("2011-01-04") - MINUTE)
        assert_not_restricted(self.ASSET1, str_to_ts("2011-01-04"))
        assert_not_restricted(self.ASSET1, str_to_ts("2011-01-04") + MINUTE)

        # ALLOWED --> ALLOWED
        assert_not_restricted(self.ASSET1, str_to_ts("2011-01-05") - MINUTE)
        assert_not_restricted(self.ASSET1, str_to_ts("2011-01-05"))
        assert_not_restricted(self.ASSET1, str_to_ts("2011-01-05") + MINUTE)

        # ALLOWED --> FROZEN
        assert_not_restricted(self.ASSET1, str_to_ts("2011-01-06") - MINUTE)
        assert_is_restricted(self.ASSET1, str_to_ts("2011-01-06"))
        assert_is_restricted(self.ASSET1, str_to_ts("2011-01-06") + MINUTE)

        # FROZEN --> FROZEN
        assert_is_restricted(self.ASSET1, str_to_ts("2011-01-07") - MINUTE)
        assert_is_restricted(self.ASSET1, str_to_ts("2011-01-07"))
        assert_is_restricted(self.ASSET1, str_to_ts("2011-01-07") + MINUTE)

    def test_static_restrictions(self):
        """Test single- and multi-asset queries on static restrictions"""

        restricted_a1 = self.ASSET1
        restricted_a2 = self.ASSET2
        unrestricted_a3 = self.ASSET3

        rl = StaticRestrictions([restricted_a1, restricted_a2])
        assert_not_restricted = partial(self.assert_not_restricted, rl)
        assert_is_restricted = partial(self.assert_is_restricted, rl)
        assert_all_restrictions = partial(self.assert_all_restrictions, rl)

        for dt in [
            str_to_ts(dt_str)
            for dt_str in ("2011-01-03", "2011-01-04", "2011-01-04 1:01", "2020-01-04")
        ]:
            assert_is_restricted(restricted_a1, dt)
            assert_is_restricted(restricted_a2, dt)
            assert_not_restricted(unrestricted_a3, dt)

            assert_all_restrictions([True, True, False], dt)

    def test_security_list_restrictions(self):
        """Test single- and multi-asset queries on restrictions defined by
        zipline.utils.security_list.SecurityList
        """

        # A mock SecurityList object filled with fake data
        class SecurityList:
            def __init__(self, assets_by_dt):
                self.assets_by_dt = assets_by_dt

            def current_securities(self, dt):
                return self.assets_by_dt[dt]

        assets_by_dt = {
            str_to_ts("2011-01-03"): [self.ASSET1],
            str_to_ts("2011-01-04"): [self.ASSET2, self.ASSET3],
            str_to_ts("2011-01-05"): [self.ASSET1, self.ASSET2, self.ASSET3],
        }

        rl = SecurityListRestrictions(SecurityList(assets_by_dt))

        assert_not_restricted = partial(self.assert_not_restricted, rl)
        assert_is_restricted = partial(self.assert_is_restricted, rl)
        assert_all_restrictions = partial(self.assert_all_restrictions, rl)

        assert_is_restricted(self.ASSET1, str_to_ts("2011-01-03"))
        assert_not_restricted(self.ASSET2, str_to_ts("2011-01-03"))
        assert_not_restricted(self.ASSET3, str_to_ts("2011-01-03"))
        assert_all_restrictions([True, False, False], str_to_ts("2011-01-03"))

        assert_not_restricted(self.ASSET1, str_to_ts("2011-01-04"))
        assert_is_restricted(self.ASSET2, str_to_ts("2011-01-04"))
        assert_is_restricted(self.ASSET3, str_to_ts("2011-01-04"))
        assert_all_restrictions([False, True, True], str_to_ts("2011-01-04"))

        assert_is_restricted(self.ASSET1, str_to_ts("2011-01-05"))
        assert_is_restricted(self.ASSET2, str_to_ts("2011-01-05"))
        assert_is_restricted(self.ASSET3, str_to_ts("2011-01-05"))
        assert_all_restrictions([True, True, True], str_to_ts("2011-01-05"))

    def test_noop_restrictions(self):
        """Test single- and multi-asset queries on no-op restrictions"""

        rl = NoRestrictions()
        assert_not_restricted = partial(self.assert_not_restricted, rl)
        assert_all_restrictions = partial(self.assert_all_restrictions, rl)

        for dt in [
            str_to_ts(dt_str) for dt_str in ("2011-01-03", "2011-01-04", "2020-01-04")
        ]:
            assert_not_restricted(self.ASSET1, dt)
            assert_not_restricted(self.ASSET2, dt)
            assert_not_restricted(self.ASSET3, dt)
            assert_all_restrictions([False, False, False], dt)

    def test_union_restrictions(self):
        """Test that we appropriately union restrictions together, including
        eliminating redundancy (ignoring NoRestrictions) and flattening out
        the underlying sub-restrictions of _UnionRestrictions
        """

        no_restrictions_rl = NoRestrictions()

        st_restrict_asset1 = StaticRestrictions([self.ASSET1])
        st_restrict_asset2 = StaticRestrictions([self.ASSET2])
        st_restricted_assets = [self.ASSET1, self.ASSET2]

        before_frozen_dt = str_to_ts("2011-01-05")
        freeze_dt_1 = str_to_ts("2011-01-06")
        unfreeze_dt = str_to_ts("2011-01-06 16:00")
        hist_restrict_asset3_1 = HistoricalRestrictions(
            [
                Restriction(self.ASSET3, freeze_dt_1, FROZEN),
                Restriction(self.ASSET3, unfreeze_dt, ALLOWED),
            ]
        )

        freeze_dt_2 = str_to_ts("2011-01-07")
        hist_restrict_asset3_2 = HistoricalRestrictions(
            [Restriction(self.ASSET3, freeze_dt_2, FROZEN)]
        )

        # A union of a NoRestrictions with a non-trivial restriction should
        # yield the original restriction
        trivial_union_restrictions = no_restrictions_rl | st_restrict_asset1
        assert isinstance(trivial_union_restrictions, StaticRestrictions)

        # A union of two non-trivial restrictions should yield a
        # UnionRestrictions
        st_union_restrictions = st_restrict_asset1 | st_restrict_asset2
        assert isinstance(st_union_restrictions, _UnionRestrictions)

        arb_dt = str_to_ts("2011-01-04")
        self.assert_is_restricted(st_restrict_asset1, self.ASSET1, arb_dt)
        self.assert_not_restricted(st_restrict_asset1, self.ASSET2, arb_dt)
        self.assert_not_restricted(st_restrict_asset2, self.ASSET1, arb_dt)
        self.assert_is_restricted(st_restrict_asset2, self.ASSET2, arb_dt)
        self.assert_is_restricted(st_union_restrictions, self.ASSET1, arb_dt)
        self.assert_is_restricted(st_union_restrictions, self.ASSET2, arb_dt)
        self.assert_many_restrictions(
            st_restrict_asset1, st_restricted_assets, [True, False], arb_dt
        )
        self.assert_many_restrictions(
            st_restrict_asset2, st_restricted_assets, [False, True], arb_dt
        )
        self.assert_many_restrictions(
            st_union_restrictions, st_restricted_assets, [True, True], arb_dt
        )

        # A union of a 2-sub-restriction UnionRestrictions and a
        # non-trivial restrictions should yield a UnionRestrictions with
        # 3 sub restrictions. Works with UnionRestrictions on both the left
        # side or right side
        for r1, r2 in [
            (st_union_restrictions, hist_restrict_asset3_1),
            (hist_restrict_asset3_1, st_union_restrictions),
        ]:
            union_or_hist_restrictions = r1 | r2
            assert isinstance(union_or_hist_restrictions, _UnionRestrictions)
            assert len(union_or_hist_restrictions.sub_restrictions) == 3

            # Includes the two static restrictions on ASSET1 and ASSET2,
            # and the historical restriction on ASSET3 starting on freeze_dt_1
            # and ending on unfreeze_dt
            self.assert_all_restrictions(
                union_or_hist_restrictions, [True, True, False], before_frozen_dt
            )
            self.assert_all_restrictions(
                union_or_hist_restrictions, [True, True, True], freeze_dt_1
            )
            self.assert_all_restrictions(
                union_or_hist_restrictions, [True, True, False], unfreeze_dt
            )
            self.assert_all_restrictions(
                union_or_hist_restrictions, [True, True, False], freeze_dt_2
            )

        # A union of two 2-sub-restrictions UnionRestrictions should yield a
        # UnionRestrictions with 4 sub restrictions.
        hist_union_restrictions = hist_restrict_asset3_1 | hist_restrict_asset3_2
        multi_union_restrictions = st_union_restrictions | hist_union_restrictions

        assert isinstance(multi_union_restrictions, _UnionRestrictions)
        assert len(multi_union_restrictions.sub_restrictions) == 4

        # Includes the two static restrictions on ASSET1 and ASSET2, the
        # first historical restriction on ASSET3 starting on freeze_dt_1 and
        # ending on unfreeze_dt, and the second historical restriction on
        # ASSET3 starting on freeze_dt_2
        self.assert_all_restrictions(
            multi_union_restrictions, [True, True, False], before_frozen_dt
        )
        self.assert_all_restrictions(
            multi_union_restrictions, [True, True, True], freeze_dt_1
        )
        self.assert_all_restrictions(
            multi_union_restrictions, [True, True, False], unfreeze_dt
        )
        self.assert_all_restrictions(
            multi_union_restrictions, [True, True, True], freeze_dt_2
        )
