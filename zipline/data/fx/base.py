from interface import default, Interface

import numpy as np
import pandas as pd

from zipline.utils.sentinel import sentinel

DEFAULT_FX_RATE = sentinel('DEFAULT_FX_RATE')


class FXRateReader(Interface):

    def get_rates(self, rate, quote, bases, dts):
        """
        Get rates to convert ``bases`` into ``quote``.

        Parameters
        ----------
        rate : str
            Rate type to load. Readers intended for use with the Pipeline API
            should support at least ``zipline.data.fx.DEFAULT_FX_RATE``, which
            will be used by default for Pipeline API terms that don't specify a
            specific rate.
        quote : str
            Currency code of the currency to convert into.
        bases : np.array[object]
            Array of codes of the currencies to convert from. A single currency
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
        """Scalar version of ``get_rates``.

        Parameters
        ----------
        rate : str
            Rate type to load. Readers intended for use with the Pipeline API
            should support at least ``zipline.data.fx.DEFAULT_FX_RATE``, which
            will be used by default for Pipeline API terms that don't specify a
            specific rate.
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
        rates_array = self.get_rates(
            rate,
            quote,
            bases=np.array([base], dtype=object),
            dts=pd.DatetimeIndex([dt], tz='UTC'),
        )
        return rates_array[0, 0]
