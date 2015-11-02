__author__ = 'michael'


from zipline.utils.protocol_utils import Enum

# Currencies we need for basic FX conversions
CCY = Enum(
    'USD',
    'GBP',
    'CHF',
    'EUR',
    'JPY',
    'AUD'
)
