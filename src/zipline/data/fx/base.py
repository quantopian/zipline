from interface import default, Interface

import numpy as np
import pandas as pd

from zipline.utils.date_utils import make_utc_aware
from zipline.utils.sentinel import sentinel
from zipline.lib._factorize import factorize_strings

DEFAULT_FX_RATE = sentinel("DEFAULT_FX_RATE")


class FXRateReader(Interface):
    """
    Interface for reading foreign exchange (fx) rates.

    An FX rate reader contains one or more distinct "rates", each of which
    corresponds to a collection of mappings from (quote, base, dt) ->
    float. The value produced for a given (quote, base, dt) triple is the
    exchange rate to use when converting from ``base`` to ``quote`` on ``dt``.

    The specific set of rates contained in a particular reader is
    user-defined. We infer no particular semantics from their names, other than
    that they are distinct rates. Examples of possible rate names might be
    things like "bid", "mid", and "ask", or "london_close", "tokyo_close",
    "nyse_close".

    Implementations of :class:`FXRateReader` must provide at least one method::

        def get_rates(self, rate, quote, bases, dts):

    which takes a rate, a quote currency, an array of base currencies, and an
    array of dts, and produces a (len(dts), len(base))-shape array containing a
    conversion rates for all pairs in the cartesian product of bases and dts.

    Given a definition of :meth:`get_rates`, this interface automatically
    generates two additional methods::

        def get_rates_scalar(self, rate, quote, base, dt):

    and::

        def get_rates_columnar(self, rate, quote, bases, dts):

    :meth:`get_rates_scalar` takes scalar-valued ``base`` and ``dt`` values,
    and returns a scalar float value for the requested fx rate.

    :meth:`get_rates_columnar` takes parallel arrays of ``bases`` and ``dts``
    and returns a same-length array of fx rates by performing a lookup on the
    (base, dt) pairs drawn from zipping together ``bases``, and ``dts``. In
    other words, its behavior is equivalent to::

        def get_rates_columnnar(self, rate, quote, bases, dts):
            out = []
            for base, dt in zip(bases, dts):
                out.append(self.get_rate_scalar(rate, quote, base, dt))
            return np.array(out)
    """

    def get_rates(self, rate, quote, bases, dts):
        """
        Load a 2D array of fx rates.

        Parameters
        ----------
        rate : str
            Name of the rate to load.
        quote : str
            Currency code of the currency to convert into.
        bases : np.array[object]
            Array of codes of the currencies to convert from. The same currency
            may appear multiple times.
        dts : pd.DatetimeIndex
            Datetimes for which to load rates. Must be sorted in ascending
            order and localized to UTC.

        Returns
        -------
        rates : np.array
            Array of shape ``(len(dts), len(bases))`` containing foreign
            exchange rates mapping currencies from ``bases`` to ``quote``.

            The row at index i corresponds to the dt in dts[i].
            The column at index j corresponds to the base currency in bases[j].
        """

    @default
    def get_rate_scalar(self, rate, quote, base, dt):
        """
        Load a scalar FX rate value.

        Parameters
        ----------
        rate : str
            Name of the rate to load.
        quote : str
            Currency code of the currency to convert into.
        base : str
            Currency code of the currency to convert from.
        dt : np.datetime64 or pd.Timestamp
            Datetime on which to load rate.

        Returns
        -------
        rate : np.float64
            Exchange rate from base -> quote on dt.
        """
        rates_2d = self.get_rates(
            rate,
            quote,
            bases=np.array([base], dtype=object),
            dts=make_utc_aware(pd.DatetimeIndex([dt])),
        )
        return rates_2d[0, 0]

    @default
    def get_rates_columnar(self, rate, quote, bases, dts):
        """
        Load a 1D array of FX rates.

        Parameters
        ----------
        rate : str
            Name of the rate to load.
        quote : str
            Currency code of the currency to convert into.
        bases : np.array[object]
            Array of codes of the currencies to convert from. The same currency
            may appear multiple times.
        dts : np.DatetimeIndex
            Datetimes for which to load rates. The same value may appear
            multiple times. Datetimes do not need to be sorted.
        """
        if len(bases) != len(dts):
            raise ValueError(
                "len(bases) ({}) != len(dts) ({})".format(len(bases), len(dts))
            )

        bases_ix, unique_bases, _ = factorize_strings(
            bases,
            missing_value=None,
            # Only dts need to be sorted, not bases.
            sort=False,
        )
        # NOTE: np.unique returns unique_dts in sorted order, which is required
        # for calling get_rates.
        unique_dts, dts_ix = np.unique(dts.values, return_inverse=True)
        rates_2d = self.get_rates(
            rate, quote, unique_bases, pd.DatetimeIndex(unique_dts, tz="utc")
        )
        return rates_2d[dts_ix, bases_ix]
