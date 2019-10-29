"""Reference pipeline loader implementatino for ExchangeRates dataset family.
"""
from interface import implements
from .base import PipelineLoader


class ExchangeRatesLoader(implements(PipelineLoader)):

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        pass
