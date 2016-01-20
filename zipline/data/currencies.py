__author__ = 'michael'


from zipline.utils.enum import enum

# Currencies we need for basic FX conversions
CCY = enum(
    'USD',
    'GBP',
    'CHF',
    'EUR',
    'JPY',
    'AUD'
)
