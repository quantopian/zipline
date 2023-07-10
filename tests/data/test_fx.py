import numpy as np
import pandas as pd
import itertools

from zipline.data.fx import DEFAULT_FX_RATE
from zipline.testing.predicates import assert_equal
import zipline.testing.fixtures as zp_fixtures
import pytest


class _FXReaderTestCase(zp_fixtures.WithFXRates, zp_fixtures.ZiplineTestCase):
    """Base class for testing FXRateReader implementations.

    To test a new FXRateReader implementation, subclass from this base class
    and implement the ``reader`` property, returning an FXRateReader that uses
    the data stored in ``cls.fx_rates``.
    """

    __test__ = False

    FX_RATES_START_DATE = pd.Timestamp("2014-01-01")
    FX_RATES_END_DATE = pd.Timestamp("2014-01-31")

    # Calendar to which exchange rates data is aligned.
    FX_RATES_CALENDAR = "24/5"

    # Currencies between which exchange rates can be calculated.
    FX_RATES_CURRENCIES = ["USD", "CAD", "GBP", "EUR"]

    # Fields for which exchange rate data is present.
    FX_RATES_RATE_NAMES = ["london_mid", "tokyo_mid"]

    # Field to be used on a lookup of `'default'`.
    FX_RATES_DEFAULT_RATE = "london_mid"

    # Used by WithFXRates.
    @classmethod
    def make_fx_rates(cls, fields, currencies, sessions):
        ndays = len(sessions)

        # Give each currency a timeseries of "true" values, and compute fx
        # rates as ratios between true values.
        reference = pd.DataFrame(
            {
                "USD": np.linspace(1.0, 2.0, num=ndays),
                "CAD": np.linspace(2.0, 3.0, num=ndays),
                "GBP": np.linspace(3.0, 4.0, num=ndays),
                "EUR": np.linspace(4.0, 5.0, num=ndays),
            },
            index=sessions,
            columns=currencies,
        )

        cls.tokyo_mid_rates = cls.make_fx_rates_from_reference(reference)
        # Make london_mid different by adding +1 to reference values.
        cls.london_mid_rates = cls.make_fx_rates_from_reference(reference + 1)

        # This will be set as cls.fx_rates by WithFXRates.
        return {
            "london_mid": cls.london_mid_rates,
            "tokyo_mid": cls.tokyo_mid_rates,
        }

    @property
    def reader(self):
        raise NotImplementedError("Must be implemented by test suite.")

    def test_scalar_lookup(self):
        reader = self.reader

        rates = self.FX_RATES_RATE_NAMES
        quotes = self.FX_RATES_CURRENCIES
        bases = self.FX_RATES_CURRENCIES + [None]
        dates = pd.date_range(
            self.FX_RATES_START_DATE - pd.Timedelta("1 day"),
            self.FX_RATES_END_DATE + pd.Timedelta("1 day"),
        )
        cases = itertools.product(rates, quotes, bases, dates)

        for rate, quote, base, dt in cases:
            dts = pd.DatetimeIndex([dt])
            bases = np.array([base], dtype=object)

            result = reader.get_rates(rate, quote, bases, dts)
            assert_equal(result.shape, (1, 1))

            result_scalar = result[0, 0]
            if dt >= self.FX_RATES_START_DATE and quote == base:
                assert_equal(result_scalar, 1.0)

            expected = self.get_expected_fx_rate_scalar(rate, quote, base, dt)
            assert_equal(result_scalar, expected)

            col_result = reader.get_rates_columnar(rate, quote, bases, dts)
            assert_equal(col_result, result.ravel())

            alt_result_scalar = reader.get_rate_scalar(rate, quote, base, dt)
            assert_equal(result_scalar, alt_result_scalar)

    def test_2d_lookup(self):
        rand = np.random.RandomState(42)

        dates = pd.date_range(
            self.FX_RATES_START_DATE - pd.Timedelta("2 days"),
            self.FX_RATES_END_DATE + pd.Timedelta("2 days"),
        )
        rates = self.FX_RATES_RATE_NAMES + [DEFAULT_FX_RATE]
        possible_quotes = self.FX_RATES_CURRENCIES
        possible_bases = self.FX_RATES_CURRENCIES + [None]

        # For every combination of rate name and quote currency...
        for rate, quote in itertools.product(rates, possible_quotes):
            # Choose N random distinct days...
            for ndays in 1, 2, 7, 20:
                dts_raw = rand.choice(dates, ndays, replace=False)
                dts = pd.DatetimeIndex(
                    dts_raw,
                ).sort_values()

                # Choose M random possibly-non-distinct currencies...
                for nbases in 1, 2, 10, 200:
                    bases = rand.choice(possible_bases, nbases, replace=True).astype(
                        object
                    )

                # ...And check that we get the expected result when querying
                # for those dates/currencies.
                result = self.reader.get_rates(rate, quote, bases, dts)
                expected = self.get_expected_fx_rates(rate, quote, bases, dts)

                assert_equal(result, expected)

    def test_columnar_lookup(self):
        rand = np.random.RandomState(42)

        dates = pd.date_range(
            self.FX_RATES_START_DATE - pd.Timedelta("2 days"),
            self.FX_RATES_END_DATE + pd.Timedelta("2 days"),
        )
        rates = self.FX_RATES_RATE_NAMES + [DEFAULT_FX_RATE]
        possible_quotes = self.FX_RATES_CURRENCIES
        possible_bases = self.FX_RATES_CURRENCIES + [None]
        reader = self.reader

        # For every combination of rate name and quote currency...
        for rate, quote in itertools.product(rates, possible_quotes):
            for N in 1, 2, 10, 200:
                # Choose N (date, base) pairs randomly with replacement.
                dts_raw = rand.choice(dates, N, replace=True)
                dts = pd.DatetimeIndex(dts_raw)
                bases = rand.choice(possible_bases, N, replace=True).astype(object)

                # ... And check that we get the expected result when querying
                # for those dates/currencies.
                result = reader.get_rates_columnar(rate, quote, bases, dts)
                expected = self.get_expected_fx_rates_columnar(
                    rate,
                    quote,
                    bases,
                    dts,
                )

                assert_equal(result, expected)

    def test_load_everything(self):
        # Sanity check for the randomized tests above: check that we get
        # exactly the rates we set up in make_fx_rates if we query for their
        # indices.
        for currency in self.FX_RATES_CURRENCIES:
            tokyo_rates = self.tokyo_mid_rates[currency]
            tokyo_result = self.reader.get_rates(
                "tokyo_mid",
                currency,
                tokyo_rates.columns,
                tokyo_rates.index,
            )
            assert_equal(tokyo_result, tokyo_rates.values)

            london_rates = self.london_mid_rates[currency]
            london_result = self.reader.get_rates(
                "london_mid",
                currency,
                london_rates.columns,
                london_rates.index,
            )
            default_result = self.reader.get_rates(
                DEFAULT_FX_RATE,
                currency,
                london_rates.columns,
                london_rates.index,
            )
            assert_equal(london_result, default_result)
            assert_equal(london_result, london_rates.values)

    def test_read_before_start_date(self):
        # Reads from before the start of our data should emit NaN. We do this
        # because, for some Pipeline loaders, it's hard to put a lower bound on
        # input asof dates, so we end up making queries for asof_dates that
        # might be before the start of FX data. When that happens, we want to
        # emit NaN, but we don't want to fail.
        for bad_date in (
            self.FX_RATES_START_DATE - pd.Timedelta("1 day"),
            self.FX_RATES_START_DATE - pd.Timedelta("1000 days"),
        ):
            for rate in self.FX_RATES_RATE_NAMES:
                quote = "USD"
                bases = np.array(["CAD"], dtype=object)
                dts = pd.DatetimeIndex([bad_date])

                result = self.reader.get_rates(rate, quote, bases, dts)
                assert_equal(result.shape, (1, 1))
                assert_equal(np.nan, result[0, 0])

    def test_read_after_end_date(self):
        # Reads from **after** the end of our data, on the other hand, should
        # fail. We can always upper bound the relevant asofs that we're
        # interested in, and having fx rates forward-fill past the end of data
        # is confusing and takes a while to debug.
        for bad_date in (
            self.FX_RATES_END_DATE + pd.Timedelta("1 day"),
            self.FX_RATES_END_DATE + pd.Timedelta("1000 days"),
        ):
            for rate in self.FX_RATES_RATE_NAMES:
                quote = "USD"
                bases = np.array(["CAD"], dtype=object)
                dts = pd.DatetimeIndex([bad_date])

                result = self.reader.get_rates(rate, quote, bases, dts)
                assert_equal(result.shape, (1, 1))
                expected = self.get_expected_fx_rate_scalar(
                    rate,
                    quote,
                    "CAD",
                    self.FX_RATES_END_DATE,
                )
                assert_equal(expected, result[0, 0])

    def test_read_unknown_base(self):
        for rate in self.FX_RATES_RATE_NAMES:
            quote = "USD"
            for unknown_base in "XXX", None:
                bases = np.array([unknown_base], dtype=object)
                dts = pd.DatetimeIndex([self.FX_RATES_START_DATE])
                result = self.reader.get_rates(rate, quote, bases, dts)[0, 0]
                assert_equal(result, np.nan)


class InMemoryFXReaderTestCase(_FXReaderTestCase):
    __test__ = True

    @property
    def reader(self):
        return self.in_memory_fx_rate_reader


class HDF5FXReaderTestCase(zp_fixtures.WithTmpDir, _FXReaderTestCase):
    __test__ = True

    @classmethod
    def init_class_fixtures(cls):
        super(HDF5FXReaderTestCase, cls).init_class_fixtures()
        path = cls.tmpdir.getpath("fx_rates.h5")
        cls.h5_fx_reader = cls.write_h5_fx_rates(path)

    @property
    def reader(self):
        return self.h5_fx_reader


class FastGetLocTestCase(zp_fixtures.ZiplineTestCase):
    def test_fast_get_loc_ffilled(self):
        dts = pd.to_datetime(
            [
                "2014-01-02",
                "2014-01-03",
                # Skip 2014-01-04
                "2014-01-05",
                "2014-01-06",
            ]
        )

        for dt in pd.date_range("2014-01-02", "2014-01-08"):
            result = zp_fixtures.fast_get_loc_ffilled(dts.values, dt.asm8)
            expected = dts.get_indexer([dt], method="ffill")[0]
            assert_equal(result, expected)

        with pytest.raises(KeyError):
            dts.get_loc(pd.Timestamp("2014-01-01"))

        with pytest.raises(KeyError):
            zp_fixtures.fast_get_loc_ffilled(dts, pd.Timestamp("2014-01-01"))
