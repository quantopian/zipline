from collections import namedtuple
import datetime
from textwrap import dedent

import numpy as np
import pandas as pd
import pytz

from zipline.country import CountryCode
from zipline.pipeline import Pipeline
from zipline.pipeline.data import Column, DataSet
from zipline.pipeline.data.testing import TestingDataSet
from zipline.pipeline.domain import (
    AmbiguousDomain,
    AR_EQUITIES,
    AT_EQUITIES,
    AU_EQUITIES,
    BE_EQUITIES,
    BR_EQUITIES,
    BUILT_IN_DOMAINS,
    CA_EQUITIES,
    CH_EQUITIES,
    CL_EQUITIES,
    CN_EQUITIES,
    CO_EQUITIES,
    CZ_EQUITIES,
    DE_EQUITIES,
    DK_EQUITIES,
    EquityCalendarDomain,
    EquitySessionDomain,
    ES_EQUITIES,
    FI_EQUITIES,
    FR_EQUITIES,
    GB_EQUITIES,
    GENERIC,
    GR_EQUITIES,
    HK_EQUITIES,
    HU_EQUITIES,
    ID_EQUITIES,
    IE_EQUITIES,
    IN_EQUITIES,
    infer_domain,
    IT_EQUITIES,
    JP_EQUITIES,
    KR_EQUITIES,
    MX_EQUITIES,
    MY_EQUITIES,
    NL_EQUITIES,
    NO_EQUITIES,
    NZ_EQUITIES,
    PE_EQUITIES,
    PH_EQUITIES,
    PK_EQUITIES,
    PL_EQUITIES,
    PT_EQUITIES,
    RU_EQUITIES,
    SE_EQUITIES,
    SG_EQUITIES,
    TH_EQUITIES,
    TR_EQUITIES,
    TW_EQUITIES,
    US_EQUITIES,
    ZA_EQUITIES,
)
from zipline.pipeline.factors import CustomFactor
import zipline.testing.fixtures as zf
from zipline.testing.core import parameter_space, powerset
from zipline.testing.predicates import assert_equal, assert_messages_equal
from zipline.utils.pandas_utils import days_at_time


class Sum(CustomFactor):

    def compute(self, today, assets, out, data):
        out[:] = data.sum(axis=0)

    @classmethod
    def create(cls, column, window_length):
        return cls(inputs=[column], window_length=window_length)


class MixedGenericsTestCase(zf.WithSeededRandomPipelineEngine,
                            zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2014-01-02', tz='utc')
    END_DATE = pd.Timestamp('2014-01-31', tz='utc')
    ASSET_FINDER_EQUITY_SIDS = (1, 2, 3, 4, 5)
    ASSET_FINDER_COUNTRY_CODE = 'US'

    def test_mixed_generics(self):
        """
        Test that we can run pipelines with mixed generic/non-generic terms.

        This test is a regression test for failures encountered during
        development where having a mix of generic and non-generic columns in
        the term graph caused bugs in our extra row accounting.
        """
        USTestingDataSet = TestingDataSet.specialize(US_EQUITIES)
        base_terms = {
            'sum3_generic': Sum.create(TestingDataSet.float_col, 3),
            'sum3_special': Sum.create(USTestingDataSet.float_col, 3),
            'sum10_generic': Sum.create(TestingDataSet.float_col, 10),
            'sum10_special': Sum.create(USTestingDataSet.float_col, 10),
        }

        def run(ts):
            pipe = Pipeline(ts, domain=US_EQUITIES)
            start = self.trading_days[-5]
            end = self.trading_days[-1]
            return self.run_pipeline(pipe, start, end)

        base_result = run(base_terms)

        for subset in powerset(base_terms):
            subset_terms = {t: base_terms[t] for t in subset}
            result = run(subset_terms).sort_index(axis=1)
            expected = base_result[list(subset)].sort_index(axis=1)
            assert_equal(result, expected)


class SpecializeTestCase(zf.ZiplineTestCase):

    @parameter_space(domain=BUILT_IN_DOMAINS)
    def test_specialize(self, domain):
        class MyData(DataSet):
            col1 = Column(dtype=float)
            col2 = Column(dtype=int, missing_value=100)
            col3 = Column(dtype=object, missing_value="")

        class MyDataSubclass(MyData):
            col4 = Column(dtype=float)

        def do_checks(cls, colnames):

            specialized = cls.specialize(domain)

            # Specializations should be memoized.
            self.assertIs(specialized, cls.specialize(domain))
            self.assertIs(specialized, specialized.specialize(domain))

            # Specializations should have the same name and module
            assert_equal(specialized.__name__, cls.__name__)
            assert_equal(specialized.__module__, cls.__module__)
            self.assertIs(specialized.domain, domain)

            for attr in colnames:
                original = getattr(cls, attr)
                new = getattr(specialized, attr)

                # We should get a new column from the specialization, which
                # should be the same object that we would get from specializing
                # the original column.
                self.assertIsNot(original, new)
                self.assertIs(new, original.specialize(domain))

                # Columns should be bound to their respective datasets.
                self.assertIs(original.dataset, cls)
                self.assertIs(new.dataset, specialized)

                # The new column should have the domain of the specialization.
                assert_equal(new.domain, domain)

                # Names, dtypes, and missing_values should match.
                assert_equal(original.name, new.name)
                assert_equal(original.dtype, new.dtype)
                assert_equal(original.missing_value, new.missing_value)

        do_checks(MyData, ['col1', 'col2', 'col3'])
        do_checks(MyDataSubclass, ['col1', 'col2', 'col3', 'col4'])

    @parameter_space(domain=BUILT_IN_DOMAINS)
    def test_unspecialize(self, domain):

        class MyData(DataSet):
            col1 = Column(dtype=float)
            col2 = Column(dtype=int, missing_value=100)
            col3 = Column(dtype=object, missing_value="")

        class MyDataSubclass(MyData):
            col4 = Column(dtype=float)

        def do_checks(cls, colnames):
            specialized = cls.specialize(domain)
            unspecialized = specialized.unspecialize()
            specialized_again = unspecialized.specialize(domain)

            self.assertIs(unspecialized, cls)
            self.assertIs(specialized, specialized_again)

            for attr in colnames:
                original = getattr(cls, attr)
                new = getattr(specialized, attr)
                # Unspecializing a specialization should give back the
                # original.
                self.assertIs(new.unspecialize(), original)
                # Specializing again should give back the same as the first
                # specialization.
                self.assertIs(new.unspecialize().specialize(domain), new)

        do_checks(MyData, ['col1', 'col2', 'col3'])
        do_checks(MyDataSubclass, ['col1', 'col2', 'col3', 'col4'])

    @parameter_space(domain_param=[BE_EQUITIES, CA_EQUITIES, CH_EQUITIES])
    def test_specialized_root(self, domain_param):
        different_domain = GB_EQUITIES

        class MyData(DataSet):
            domain = domain_param
            col1 = Column(dtype=float)

        class MyDataSubclass(MyData):
            col2 = Column(dtype=float)

        def do_checks(cls, colnames):
            # DataSets with concrete domains can't be specialized to other
            # concrete domains.
            with self.assertRaises(ValueError):
                cls.specialize(different_domain)

            # Same goes for columns of the dataset.
            for name in colnames:
                col = getattr(cls, name)
                with self.assertRaises(ValueError):
                    col.specialize(different_domain)

            # We always allow unspecializing to simplify the implementation of
            # loaders and dispatchers that want to use the same loader for an
            # entire dataset family.
            generic_non_root = cls.unspecialize()

            # Allow specializing a generic non-root back to its family root.
            self.assertIs(generic_non_root.specialize(domain_param), cls)
            for name in colnames:
                # Same deal for columns.
                self.assertIs(
                    getattr(generic_non_root, name).specialize(domain_param),
                    getattr(cls, name),
                )

            # Don't allow specializing to any other domain.
            with self.assertRaises(ValueError):
                generic_non_root.specialize(different_domain)

            # Same deal for columns.
            for name in colnames:
                col = getattr(generic_non_root, name)
                with self.assertRaises(ValueError):
                    col.specialize(different_domain)

        do_checks(MyData, ['col1'])
        do_checks(MyDataSubclass, ['col1', 'col2'])


class D(DataSet):
    c1 = Column(float)
    c2 = Column(bool)
    c3 = Column(object)


class InferDomainTestCase(zf.ZiplineTestCase):

    def check(self, inputs, expected):
        result = infer_domain(inputs)
        self.assertIs(result, expected)

    def check_fails(self, inputs, expected_domains):
        with self.assertRaises(AmbiguousDomain) as e:
            infer_domain(inputs)

        err = e.exception
        self.assertEqual(err.domains, expected_domains)

        return err

    def test_all_generic(self):
        self.check([], GENERIC)
        self.check([D.c1], GENERIC)
        self.check([D.c1, D.c2], GENERIC)
        self.check([D.c1, D.c2, D.c3], GENERIC)
        self.check([D.c1.latest, D.c2.latest, D.c3.latest], GENERIC)

    @parameter_space(domain=[US_EQUITIES, GB_EQUITIES])
    def test_all_non_generic(self, domain):
        D_s = D.specialize(domain)
        self.check([D_s.c1], domain)
        self.check([D_s.c1, D_s.c2], domain)
        self.check([D_s.c1, D_s.c2, D_s.c3], domain)
        self.check([D_s.c1, D_s.c2, D_s.c3.latest], domain)

    @parameter_space(domain=[US_EQUITIES, GB_EQUITIES])
    def test_mix_generic_and_specialized(self, domain):
        D_s = D.specialize(domain)
        self.check([D.c1, D_s.c3], domain)
        self.check([D.c1, D.c2, D_s.c3], domain)
        self.check([D.c1, D_s.c2, D_s.c3], domain)

    def test_conflict(self):
        D_US = D.specialize(US_EQUITIES)
        D_CA = D.specialize(CA_EQUITIES)
        D_GB = D.specialize(GB_EQUITIES)

        # Conflict of size 2.
        self.check_fails(
            [D_US.c1, D_CA.c1],
            [CA_EQUITIES, US_EQUITIES],
        )

        # Conflict of size 3.
        self.check_fails(
            [D_US.c1, D_CA.c1, D_GB.c1],
            [CA_EQUITIES, GB_EQUITIES, US_EQUITIES],
        )

        # Make sure each domain only appears once if there are duplicates.
        self.check_fails(
            [D_US.c1, D_CA.c1, D_CA.c2],
            [CA_EQUITIES, US_EQUITIES],
        )

        # Make sure that we filter GENERIC out of the error.
        self.check_fails(
            [D_US.c1, D_CA.c1, D.c1],
            [CA_EQUITIES, US_EQUITIES],
        )

    def test_ambiguous_domain_repr(self):
        err = AmbiguousDomain([CA_EQUITIES, GB_EQUITIES, US_EQUITIES])

        result = str(err)
        expected = dedent(
            """\
            Found terms with conflicting domains:
              - EquityCalendarDomain('CA', 'XTSE')
              - EquityCalendarDomain('GB', 'XLON')
              - EquityCalendarDomain('US', 'XNYS')"""
        )
        assert_messages_equal(result, expected)


class DataQueryCutoffForSessionTestCase(zf.ZiplineTestCase):
    def test_generic(self):
        sessions = pd.date_range('2014-01-01', '2014-06-01')
        with self.assertRaises(NotImplementedError):
            GENERIC.data_query_cutoff_for_sessions(sessions)

    def _test_equity_calendar_domain(self,
                                     domain,
                                     expected_cutoff_time,
                                     expected_cutoff_date_offset=0):
        sessions = pd.DatetimeIndex(domain.calendar.all_sessions[:50])

        expected = days_at_time(
            sessions,
            expected_cutoff_time,
            domain.calendar.tz,
            expected_cutoff_date_offset,
        )
        actual = domain.data_query_cutoff_for_sessions(sessions)

        assert_equal(actual, expected, check_names=False)

    def test_built_in_equity_calendar_domain_defaults(self):
        # test the defaults
        expected_cutoff_times = {
            AR_EQUITIES: datetime.time(10, 15),
            AT_EQUITIES: datetime.time(8, 15),
            AU_EQUITIES: datetime.time(9, 15),
            BE_EQUITIES: datetime.time(8, 15),
            BR_EQUITIES: datetime.time(9, 15),
            CA_EQUITIES: datetime.time(8, 45),
            CH_EQUITIES: datetime.time(8, 15),
            CL_EQUITIES: datetime.time(8, 45),
            CN_EQUITIES: datetime.time(8, 45),
            CO_EQUITIES: datetime.time(8, 45),
            CZ_EQUITIES: datetime.time(8, 15),
            DE_EQUITIES: datetime.time(8, 15),
            DK_EQUITIES: datetime.time(8, 15),
            ES_EQUITIES: datetime.time(8, 15),
            FI_EQUITIES: datetime.time(9, 15),
            FR_EQUITIES: datetime.time(8, 15),
            GB_EQUITIES: datetime.time(7, 15),
            GR_EQUITIES: datetime.time(9, 15),
            HK_EQUITIES: datetime.time(9, 15),
            HU_EQUITIES: datetime.time(8, 15),
            ID_EQUITIES: datetime.time(8, 15),
            IE_EQUITIES: datetime.time(7, 15),
            IN_EQUITIES: datetime.time(8, 30),
            IT_EQUITIES: datetime.time(8, 15),
            JP_EQUITIES: datetime.time(8, 15),
            KR_EQUITIES: datetime.time(8, 15),
            MX_EQUITIES: datetime.time(7, 45),
            MY_EQUITIES: datetime.time(8, 15),
            NL_EQUITIES: datetime.time(8, 15),
            NO_EQUITIES: datetime.time(8, 15),
            NZ_EQUITIES: datetime.time(9, 15),
            PE_EQUITIES: datetime.time(8, 15),
            PH_EQUITIES: datetime.time(8, 45),
            PK_EQUITIES: datetime.time(8, 47),
            PL_EQUITIES: datetime.time(8, 15),
            PT_EQUITIES: datetime.time(7, 15),
            RU_EQUITIES: datetime.time(9, 15),
            SE_EQUITIES: datetime.time(8, 15),
            SG_EQUITIES: datetime.time(8, 15),
            TH_EQUITIES: datetime.time(9, 15),
            TR_EQUITIES: datetime.time(9, 15),
            TW_EQUITIES: datetime.time(8, 15),
            US_EQUITIES: datetime.time(8, 45),
            ZA_EQUITIES: datetime.time(8, 15),
        }

        # make sure we are not missing any domains in this test
        self.assertEqual(set(expected_cutoff_times), set(BUILT_IN_DOMAINS))

        for domain, expected_cutoff_time in expected_cutoff_times.items():
            self._test_equity_calendar_domain(domain, expected_cutoff_time)

    def test_equity_calendar_domain(self):
        # test non-default time
        self._test_equity_calendar_domain(
            EquityCalendarDomain(
                CountryCode.UNITED_STATES,
                'XNYS',
                data_query_offset=-np.timedelta64(2 * 60 + 30, 'm'),
            ),
            datetime.time(7, 0),
        )

        # test offset that changes the date
        self._test_equity_calendar_domain(
            EquityCalendarDomain(
                CountryCode.UNITED_STATES,
                'XNYS',
                data_query_offset=-np.timedelta64(10, 'h'),
            ),
            datetime.time(23, 30),
            expected_cutoff_date_offset=-1,
        )

        # test an offset that moves us back by more than one day
        self._test_equity_calendar_domain(
            EquityCalendarDomain(
                CountryCode.UNITED_STATES,
                'XNYS',
                data_query_offset=-np.timedelta64(24 * 6 + 10, 'h'),
            ),
            datetime.time(23, 30),
            expected_cutoff_date_offset=-7,
        )

    @parameter_space(domain=BUILT_IN_DOMAINS)
    def test_equity_calendar_not_aligned(self, domain):
        valid_sessions = domain.all_sessions()[:50]
        sessions = pd.date_range(valid_sessions[0], valid_sessions[-1])
        invalid_sessions = sessions[~sessions.isin(valid_sessions)]
        self.assertGreater(
            len(invalid_sessions),
            1,
            msg='There must be at least one invalid session.',
        )

        with self.assertRaises(ValueError) as e:
            domain.data_query_cutoff_for_sessions(sessions)

        expected_msg = (
            'cannot resolve data query time for sessions that are not on the'
            ' %s calendar:\n%s'
        ) % (domain.calendar.name, invalid_sessions)
        assert_messages_equal(str(e.exception), expected_msg)

    Case = namedtuple('Case', 'time date_offset expected_timedelta')

    @parameter_space(parameters=(
        Case(
            time=datetime.time(8, 45, tzinfo=pytz.utc),
            date_offset=0,
            expected_timedelta=datetime.timedelta(hours=8, minutes=45),
        ),
        Case(
            time=datetime.time(5, 0, tzinfo=pytz.utc),
            date_offset=0,
            expected_timedelta=datetime.timedelta(hours=5),
        ),
        Case(
            time=datetime.time(8, 45, tzinfo=pytz.timezone('Asia/Tokyo')),
            date_offset=0,
            # We should get 11:45 UTC, which is 8:45 in Tokyo time,
            # because Tokyo is 9 hours ahead of UTC.
            expected_timedelta=-datetime.timedelta(minutes=15)
        ),
        Case(
            time=datetime.time(23, 30, tzinfo=pytz.utc),
            date_offset=-1,
            # 23:30 on the previous day should be equivalent to rolling back by
            # 30 minutes.
            expected_timedelta=-datetime.timedelta(minutes=30),
        ),
        Case(
            time=datetime.time(23, 30, tzinfo=pytz.timezone('US/Eastern')),
            date_offset=-1,
            # 23:30 on the previous day in US/Eastern is equivalent to rolling
            # back 24 hours (to the previous day), then rolling forward 4 or 5
            # hours depending on daylight savings, then rolling forward 23:30,
            # so the net is:
            # -24 + 5 + 23:30 = 4:30 until April 4th
            # -24 + 4 + 23:30 = 3:30 from April 4th on.
            expected_timedelta=pd.TimedeltaIndex(
                ['4 hours 30 minutes'] * 93 + ['3 hours 30 minutes'] * 60,
            )
        )
    ))
    def test_equity_session_domain(self, parameters):
        time, date_offset, expected_timedelta = parameters
        naive_sessions = pd.date_range('2000-01-01', '2000-06-01')
        utc_sessions = naive_sessions.tz_localize('UTC')

        domain = EquitySessionDomain(
            utc_sessions,
            CountryCode.UNITED_STATES,
            data_query_time=time,
            data_query_date_offset=date_offset,
        )

        # Adding and localizing the naive_sessions here because pandas 18
        # crashes when adding a tz-aware DatetimeIndex and a
        # TimedeltaIndex. :sadpanda:.
        expected = (naive_sessions + expected_timedelta).tz_localize('utc')
        actual = domain.data_query_cutoff_for_sessions(utc_sessions)

        assert_equal(expected, actual)


class RollForwardTestCase(zf.ZiplineTestCase):

    def test_roll_forward(self):
        #     January 2017
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7

        # the first three days of the year are holidays on the Tokyo exchange,
        # so the first trading day should be the fourth
        self.assertEqual(
            JP_EQUITIES.roll_forward('2017-01-01'),
            pd.Timestamp('2017-01-04', tz='UTC'),
        )

        # in US exchanges, the first trading day after 1/1 is the 3rd
        self.assertEqual(
            US_EQUITIES.roll_forward('2017-01-01'),
            pd.Timestamp('2017-01-03', tz='UTC'),
        )

        # passing a valid trading day to roll_forward should return that day
        self.assertEqual(
            JP_EQUITIES.roll_forward('2017-01-04'),
            pd.Timestamp('2017-01-04', tz='UTC'),
        )

        # passing a date before the first session should return the
        # first session
        before_first_session = \
            JP_EQUITIES.calendar.first_session - pd.Timedelta(days=20)

        self.assertEqual(
            JP_EQUITIES.roll_forward(before_first_session),
            JP_EQUITIES.calendar.first_session
        )

        # requesting a session beyond the last session raises an ValueError
        after_last_session = \
            JP_EQUITIES.calendar.last_session + pd.Timedelta(days=20)

        with self.assertRaises(ValueError) as ve:
            JP_EQUITIES.roll_forward(after_last_session)

        self.assertEqual(
            str(ve.exception),
            "Date {} was past the last session for domain "
            "EquityCalendarDomain('JP', 'XTKS'). The last session for "
            "this domain is {}.".format(
                after_last_session.date(),
                JP_EQUITIES.calendar.last_session.date(),
            )
        )

        # test that a roll_forward works with an EquitySessionDomain,
        # not just calendar domains
        sessions = pd.DatetimeIndex(
            ['2000-01-01',
             '2000-02-01',
             '2000-04-01',
             '2000-06-01'],
            tz='UTC'
        )

        session_domain = EquitySessionDomain(
            sessions, CountryCode.UNITED_STATES
        )

        self.assertEqual(
            session_domain.roll_forward('2000-02-01'),
            pd.Timestamp('2000-02-01', tz='UTC'),
        )

        self.assertEqual(
            session_domain.roll_forward('2000-02-02'),
            pd.Timestamp('2000-04-01', tz='UTC'),
        )


class ReprTestCase(zf.ZiplineTestCase):
    def test_generic_domain_repr(self):
        self.assertEqual(repr(GENERIC), "GENERIC")
