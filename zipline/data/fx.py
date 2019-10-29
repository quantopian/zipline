"""
"""
from itertools import product

from interface import implements, Interface
import numpy as np
import pandas as pd


class ExchangeRateReader(Interface):

    def get_rates(self, field, quote, bases, dates):
        """Get rates to convert ``bases`` into ``quote``.

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """


class InMemoryExchangeRateReader(implements(ExchangeRateReader)):
    """
    Parameters
    ----------
    data : dict
        Nested map from field -> quote -> pd.DataFrame
    """

    def __init__(self, data):
        self._data = data

    def get_rates(self, field, quote, bases, dates):
        return self._rates[field][quote][bases].reindex(dates, how='ffill')


# TODO: This could just be a smart constructor. Is there a reason to actually
# want a separate type?
class TestingExchangeRateReader(InMemoryExchangeRateReader):
    """Helper for generating fake exchange rates from pseudo-random data.
    """

    def __init__(self, fields, currencies, dates):
        rng = np.random.RandomState(42)

        # Assign each currency a "true value" timeseries.
        true_values = {}
        for field, currency in sorted(product(fields, currencies)):
            true_values[currency] = self.random_linear(len(dates), rng)

        true_values_df = pd.DataFrame(
            true_values,
            index=dates,
            columns=sorted(currencies),
        )

        # Define rates as the ratio between each asset's true values.
        data = {}
        for i, field in enumerate(fields):
            data[field] = {}
            for quote in currencies:
                data[field][quote] = true_values_df / true_values_df[quote]

        super(TestingExchangeRateReader, self).__init__(data)

    @staticmethod
    def random_linear(N, rng, min_=0.5, max_=1.5):
        """
        Generate sequence of linearly-increasing values, with endpoints chosen
        uniformly at random from the interval [min_, max_].
        """
        start, end = sorted(rng.uniform(min_, max_, (2,)))
        return np.linspace(start, end, N)
