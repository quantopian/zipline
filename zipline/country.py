"""Canonical definitions for countries and currencies.

We use ISO-3166 alpha2 codes for representing countries.
We use ISO-4217 codes for representing currencies.
"""
from functools import partial, total_ordering

import numpy as np

from iso3166 import countries_by_alpha2
from iso4217 import Currency as ISO4217Currency


_ALL_COUNTRIES = {}


def strs_to_sids(strs, category_num):
    """TODO: Improve this.
    """
    out = np.full(len(strs), category_num << 50, dtype='i8')
    casted_buffer = np.ndarray(
        shape=out.shape,
        dtype='S6',
        buffer=out,
        strides=out.strides,
    )
    casted_buffer[:] = np.array(strs, dtype='S6')
    return out


def str_to_sid(str_, category_num):
    return strs_to_sids([str_], category_num)[0]


def sids_to_strs(sids, category_num):
    casted_buffer = np.ndarray(
        shape=sids.shape,
        dtype='i2',
        buffer=sids,
        strides=sids.strides,
        offset=6,
    )
    bad_metadata = np.flatnonzero(casted_buffer != 0b00000100)
    if len(bad_metadata):
        raise ValueError(
            'the following sids were not encoded from category number:'
            '{!r}'.format(sids[bad_metadata])
        )


alpha2_to_sid = partial(str_to_sid, category_num=2)


@total_ordering
class Country(object):
    """A country identifier, as defined by ISO-3166.

    Parameters
    ----------
    code : str
        ISO-3166 alpha2 code for the country.
    """
    def __new__(cls, code):
        try:
            return _ALL_COUNTRIES[code]
        except KeyError:
            iso_country = countries_by_alpha2[code]
            obj = _ALL_COUNTRIES[code] = super(Country, cls).__new__(cls)
            obj._country = iso_country
            obj._sid = alpha2_to_sid(iso_country.alpha2)
            return obj

    @property
    def code(self):
        return self._country.alpha2

    @property
    def sid(self):
        return self._sid

    @property
    def name(self):
        return self._country.apolitical_name

    def __str__(self):
        return "Country<{!r}, {!r}>".format(self.code, self.name)

    def __repr__(self):
        return "Country({!r})".format(self.code)

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.code == other.code

    def __lt__(self, other):
        return self.code < other.code


ARGENTINA = Country('AR')
AUSTRALIA = Country('AU')
AUSTRIA = Country('AT')
BELGIUM = Country('BE')
BRAZIL = Country('BR')
CANADA = Country('CA')
CHILE = Country('CL')
CHINA = Country('CN')
COLOMBIA = Country('CO')
CZECH_REPUBLIC = Country('CZ')
DENMARK = Country('DK')
FINLAND = Country('FI')
FRANCE = Country('FR')
GERMANY = Country('DE')
GREECE = Country('GR')
HONG_KONG = Country('HK')
HUNGARY = Country('HU')
INDIA = Country('IN')
INDONESIA = Country('ID')
IRELAND = Country('IE')
ISRAEL = Country('IL')
ITALY = Country('IT')
JAPAN = Country('JP')
MALAYSIA = Country('MY')
MEXICO = Country('MX')
NETHERLANDS = Country('NL')
NEW_ZEALAND = Country('NZ')
NORWAY = Country('NO')
PAKISTAN = Country('PK')
PERU = Country('PE')
PHILIPPINES = Country('PH')
POLAND = Country('PL')
PORTUGAL = Country('PT')
RUSSIA = Country('RU')
SINGAPORE = Country('SG')
SOUTH_AFRICA = Country('ZA')
SOUTH_KOREA = Country('KR')
SPAIN = Country('ES')
SWEDEN = Country('SE')
SWITZERLAND = Country('CH')
TAIWAN = Country('TW')
THAILAND = Country('TH')
TURKEY = Country('TR')
UNITED_KINGDOM = Country('GB')
UNITED_STATES = Country('US')


_ALL_CURRENCIES = {}

iso_currency_to_sid = partial(str_to_sid, category_num=3)


@total_ordering
class Currency(object):
    """A currency identifier, as defined by ISO-4217.

    Parameters
    ----------
    code : str
        ISO-4217 code for the currency.
    """
    def __new__(cls, code):
        try:
            return _ALL_CURRENCIES[code]
        except KeyError:
            iso_currency = ISO4217Currency(code)
            obj = _ALL_CURRENCIES[code] = super(Currency, cls).__new__(cls)
            obj._currency = iso_currency
            obj._sid = iso_currency_to_sid(iso_currency.value)

    @property
    def code(self):
        self._currency.value

    @property
    def name(self):
        return self._currency.currency_name

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.code == other.code

    def __lt__(self, other):
        return self.code < other.code


# TODO: These associations can change. How do we want to handle that?
COUNTRY_TO_PRIMARY_CURRENCY = {
    ARGENTINA: Currency('ARS'),
    BRAZIL: Currency('BRL'),
    CANADA: Currency('CAD'),
    CHILE: Currency('CLP'),
    COLOMBIA: Currency('COP'),
    MEXICO: Currency('MXN'),
    PERU: Currency('PEN'),
    UNITED_STATES: Currency('USD'),

    AUSTRIA: Currency('EUR'),
    BELGIUM: Currency('EUR'),
    CZECH_REPUBLIC: Currency('CZK'),
    DENMARK: Currency('DKK'),
    FINLAND: Currency('EUR'),
    FRANCE: Currency('EUR'),
    GERMANY: Currency('EUR'),
    GREECE: Currency('EUR'),
    HUNGARY: Currency('HUF'),
    IRELAND: Currency('EUR'),
    ITALY: Currency('EUR'),
    NETHERLANDS: Currency('EUR'),
    NORWAY: Currency('NOK'),
    POLAND: Currency('PLN'),
    PORTUGAL: Currency('EUR'),
    RUSSIA: Currency('RUB'),
    SOUTH_AFRICA: Currency('ZAR'),
    SPAIN: Currency('EUR'),
    SWEDEN: Currency('SEK'),
    SWITZERLAND: Currency('CHF'),
    TURKEY: Currency('TRY'),
    UNITED_KINGDOM: Currency('GBP'),

    AUSTRALIA: Currency('AUD'),
    CHINA: Currency('CNY'),
    HONG_KONG: Currency('HKD'),
    INDIA: Currency('INR'),
    INDONESIA: Currency('IDR'),
    JAPAN: Currency('JPY'),
    MALAYSIA: Currency('MYR'),
    NEW_ZEALAND: Currency('NZD'),
    PAKISTAN: Currency('PKR'),
    PHILIPPINES: Currency('PHP'),
    SINGAPORE: Currency('SGD'),
    SOUTH_KOREA: Currency('KRW'),
    TAIWAN: Currency('TWD'),
    THAILAND: Currency('THB'),
}
