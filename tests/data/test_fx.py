import itertools

import pandas as pd
import numpy as np

from zipline.data.fx import DEFAULT_FX_RATE

from zipline.testing.predicates import assert_equal
import zipline.testing.fixtures as zp_fixtures


class _FXReaderTestCase(zp_fixtures.WithFXRates,
                        zp_fixtures.ZiplineTestCase):
    """
    Base class for testing FXRateReader implementations.

    To test a new FXRateReader implementation, subclass from this base class
    and implement the ``reader`` property, returning an FXRateReader that uses
    the data stored in ``cls.fx_rates``.
    """
    FX_RATES_START_DATE = pd.Timestamp('2014-01-01', tz='UTC')
    FX_RATES_END_DATE = pd.Timestamp('2014-01-31', tz='UTC')

    # Calendar to which exchange rates data is aligned.
    FX_RATES_CALENDAR = '24/5'

    # Currencies between which exchange rates can be calculated.
    FX_RATES_CURRENCIES = ["USD", "CAD", "GBP", "EUR"]

    # Fields for which exchange rate data is present.
    FX_RATES_RATE_NAMES = ["london_mid", "tokyo_mid"]

    # Field to be used on a lookup of `'default'`.
    FX_RATES_DEFAULT_RATE = 'london_mid'

    # Used by WithFXRates.
    @classmethod
    def make_fx_rates(cls, fields, currencies, sessions):
        ndays = len(sessions)

        # Give each currency a timeseries of "true" values, and compute fx
        # rates as ratios between true values.
        reference = pd.DataFrame({
            'USD': np.linspace(1.0, 2.0, num=ndays),
            'CAD': np.linspace(2.0, 3.0, num=ndays),
            'GBP': np.linspace(3.0, 4.0, num=ndays),
            'EUR': np.linspace(4.0, 5.0, num=ndays),
        }, index=sessions, columns=currencies)

        cls.tokyo_mid_rates = cls.make_fx_rates_from_reference(reference)
        # Make london_mid different by adding +1 to reference values.
        cls.london_mid_rates = cls.make_fx_rates_from_reference(reference + 1)

        # This will be set as cls.fx_rates by WithFXRates.
        return {
            'london_mid': cls.london_mid_rates,
            'tokyo_mid': cls.tokyo_mid_rates,
        }

    @classmethod
    def get_expected_rate_scalar(cls, rate, quote, base, dt):
        """Get the expected FX rate for the given scalar coordinates.
        """
        if rate == DEFAULT_FX_RATE:
            rate = cls.FX_RATES_DEFAULT_RATE

        col = cls.fx_rates[rate][quote][base]
        # PERF: We call this function a lot in this suite, and get_loc is
        # surprisingly expensive, so optimizing it has a meaningful impact on
        # overall suite performance. See test_fast_get_loc_ffilled_for
        # assurance that this behaves the same as get_loc.
        ix = fast_get_loc_ffilled(col.index.values, dt.asm8)
        return col.values[ix]

    @classmethod
    def get_expected_rates(cls, rate, quote, bases, dts):
        """Get an array of expected FX rates for the given indices.
        """
        out = np.empty((len(dts), len(bases)), dtype='float64')

        for i, dt in enumerate(dts):
            for j, base in enumerate(bases):
                out[i, j] = cls.get_expected_rate_scalar(rate, quote, base, dt)

        return out

    @property
    def reader(self):
        raise NotImplementedError("Must be implemented by test suite.")

    def test_scalar_lookup(self):
        reader = self.reader

        rates = self.FX_RATES_RATE_NAMES
        currencies = self.FX_RATES_CURRENCIES
        dates = pd.date_range(self.FX_RATES_START_DATE, self.FX_RATES_END_DATE)

        cases = itertools.product(rates, currencies, currencies, dates)

        for rate, quote, base, dt in cases:
            dts = pd.DatetimeIndex([dt], tz='UTC')
            bases = np.array([base])

            result = reader.get_rates(rate, quote, bases, dts)
            assert_equal(result.shape, (1, 1))

            result_scalar = result[0, 0]
            if quote == base:
                assert_equal(result_scalar, 1.0)

            expected = self.get_expected_rate_scalar(rate, quote, base, dt)
            assert_equal(result_scalar, expected)

    def test_vectorized_lookup(self):
        rand = np.random.RandomState(42)

        dates = pd.date_range(self.FX_RATES_START_DATE, self.FX_RATES_END_DATE)
        rates = self.FX_RATES_RATE_NAMES + [DEFAULT_FX_RATE]
        currencies = self.FX_RATES_CURRENCIES

        # For every combination of rate name and quote currency...
        for rate, quote in itertools.product(rates, currencies):

            # Choose N random distinct days...
            for ndays in 1, 2, 7, 20:
                dts_raw = rand.choice(dates, ndays, replace=False)
                dts = pd.DatetimeIndex(dts_raw, tz='utc').sort_values()

                # Choose M random possibly-non-distinct currencies...
                for nbases in 1, 2, 10, 200:
                    bases = rand.choice(currencies, nbases, replace=True)

                # ...And check that we get the expected result when querying
                # for those dates/currencies.
                result = self.reader.get_rates(rate, quote, bases, dts)
                expected = self.get_expected_rates(rate, quote, bases, dts)

                assert_equal(result, expected)

    def test_load_everything(self):
        # Sanity check for the randomized tests above: check that we get
        # exactly the rates we set up in make_fx_rates if we query for their
        # indices.
        for currency in self.FX_RATES_CURRENCIES:
            tokyo_rates = self.tokyo_mid_rates[currency]
            tokyo_result = self.reader.get_rates(
                'tokyo_mid',
                currency,
                tokyo_rates.columns,
                tokyo_rates.index,
            )
            assert_equal(tokyo_result, tokyo_rates.values)

            london_rates = self.london_mid_rates[currency]
            london_result = self.reader.get_rates(
                'london_mid',
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
        for bad_date in (self.FX_RATES_START_DATE - pd.Timedelta('1 day'),
                         self.FX_RATES_START_DATE - pd.Timedelta('1000 days')):

            for rate in self.FX_RATES_RATE_NAMES:
                quote = 'USD'
                bases = np.array(['CAD'], dtype='S3')
                dts = pd.DatetimeIndex([bad_date])
                with self.assertRaises(ValueError):
                    self.reader.get_rates(rate, quote, bases, dts)

    def test_read_after_end_date(self):
        for bad_date in (self.FX_RATES_END_DATE + pd.Timedelta('1 day'),
                         self.FX_RATES_END_DATE + pd.Timedelta('1000 days')):

            for rate in self.FX_RATES_RATE_NAMES:
                quote = 'USD'
                bases = np.array(['CAD'], dtype='S3')
                dts = pd.DatetimeIndex([bad_date])
                with self.assertRaises(ValueError):
                    self.reader.get_rates(rate, quote, bases, dts)


class InMemoryFXReaderTestCase(_FXReaderTestCase):

    @property
    def reader(self):
        return self.in_memory_fx_rate_reader


class HDF5FXReaderTestCase(zp_fixtures.WithTmpDir,
                           _FXReaderTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(HDF5FXReaderTestCase, cls).init_class_fixtures()
        path = cls.tmpdir.getpath('fx_rates.h5')
        cls.h5_fx_reader = cls.write_h5_fx_rates(path)

    @property
    def reader(self):
        return self.h5_fx_reader


def fast_get_loc_ffilled(dts, dt):
    """
    Equivalent to dts.get_loc(dt, method='ffill'), but with reasonable
    microperformance.
    """
    ix = dts.searchsorted(dt, side='right') - 1
    if ix < 0:
        raise KeyError(dt)
    return ix


class FastGetLocTestCase(zp_fixtures.ZiplineTestCase):

    def test_fast_get_loc_ffilled(self):
        dts = pd.to_datetime([
            '2014-01-02',
            '2014-01-03',
            # Skip 2014-01-04
            '2014-01-05',
            '2014-01-06',
        ])

        for dt in pd.date_range('2014-01-02', '2014-01-08'):
            result = fast_get_loc_ffilled(dts.values, dt.asm8)
            expected = dts.get_loc(dt, method='ffill')
            assert_equal(result, expected)

        with self.assertRaises(KeyError):
            dts.get_loc(pd.Timestamp('2014-01-01'), method='ffill')

        with self.assertRaises(KeyError):
            fast_get_loc_ffilled(dts, pd.Timestamp('2014-01-01'))
