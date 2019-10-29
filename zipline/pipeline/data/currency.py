"""
DatasetFamily representing exchange rates.
"""
from zipline.international import ALL_CURRENCIES

from .dataset import DatasetFamily, Column
from ..domain import WorldCurrencies


class ExchangeRates(DatasetFamily):
    """DatasetFamily for daily exchange rates.
    """
    domain = WorldCurrencies
    mid = Column(float)

    EXTRA_DIMS = [
        ('target', ['USD', 'EUR', 'JPY']),
    ]
