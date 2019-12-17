import itertools

import h5py
import pandas as pd
import numpy as np

from zipline.data.fx.hdf5 import HDF5FXRateReader, HDF5FXRateWriter

from zipline.testing.predicates import assert_equal
import zipline.testing.fixtures as zp_fixtures


class _FXReaderTestCase(zp_fixtures.WithFXRates,
                        zp_fixtures.ZiplineTestCase):

    FX_RATES_START_DATE = pd.Timestamp('2014-01-01', tz='UTC')
    FX_RATES_END_DATE = pd.Timestamp('2014-01-31', tz='UTC')

    # Calendar to which exchange rates data is aligned.
    FX_RATES_CALENDAR = '24/5'

    # Currencies between which exchange rates can be calculated.
    FX_RATES_CURRENCIES = ["USD", "CAD", "GBP", "EUR"]

    # Fields for which exchange rate data is present.
    FX_RATES_RATE_NAMES = ["london_mid", "tokyo_mid"]

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
    def get_expected_rate(cls, rate, quote, base, dt):
        """Get the expected FX rate for the given coordinates.
        """
        col = cls.fx_rates[rate][quote][base]
        ix = col.index.get_loc(dt, method='ffill')
        return col.iloc[ix]

    @property
    def reader(self):
        raise NotImplementedError("Must be implemented by test suite.")

    # PERF_TODO: This test takes about a second on my machine. That will
    # probably be more like 3-4 seconds on Travis. Is there a good way to make
    # this faster without losing coverage?
    def test_scalar_lookup(self):
        reader = self.reader

        rates = self.FX_RATES_RATE_NAMES
        currencies = self.FX_RATES_CURRENCIES
        dates = pd.date_range(self.FX_RATES_START_DATE, self.FX_RATES_END_DATE)

        cases = itertools.product(rates, currencies, currencies, dates)

        for rate, quote, base, dt in cases:
            dts = pd.DatetimeIndex([dt])
            bases = np.array([base], dtype='S3')

            result = reader.get_rates(rate, quote, bases, dts)
            expected_scalar = self.get_expected_rate(rate, quote, base, dt)
            if quote == base:
                self.assertEqual(expected_scalar, 1.0)

            expected = pd.DataFrame(
                data=expected_scalar,
                index=dts,
                columns=bases,
            )

            assert_equal(result, expected)

    def test_vectorized_lookup(self):
        # TODO
        pass

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

        # Set by WithFXRates.
        sessions = cls.fx_rates_sessions

        with h5py.File(path, 'w') as h5_file:
            writer = HDF5FXRateWriter(h5_file)
            fx_data = ((rate, quote, quote_frame.values)
                       for rate, rate_dict in cls.fx_rates.items()
                       for quote, quote_frame in rate_dict.items())

            writer.write(
                dts=sessions.values,
                currencies=np.array(cls.FX_RATES_CURRENCIES, dtype='S3'),
                data=fx_data,
            )

        h5_file = cls.enter_class_context(h5py.File(path, 'r'))
        cls.h5_fx_reader = HDF5FXRateReader(h5_file)

    @property
    def reader(self):
        return self.h5_fx_reader
