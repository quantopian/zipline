"""Canonical definitions of country code constants.
"""
from iso3166 import countries_by_name


def code(name):
    return countries_by_name[name].alpha2


class CountryCode(object):
    """A simple namespace of iso3166 alpha2 country codes.
    """
    AUSTRALIA = code('AUSTRALIA')
    AUSTRIA = code('AUSTRIA')
    BELGIUM = code('BELGIUM')
    CANADA = code('CANADA')
    DENMARK = code('DENMARK')
    FINLAND = code('FINLAND')
    FRANCE = code('FRANCE')
    GERMANY = code('GERMANY')
    HONG_KONG = code('HONG KONG')
    IRELAND = code('IRELAND')
    ISRAEL = code('ISRAEL')
    ITALY = code('ITALY')
    JAPAN = code('JAPAN')
    NETHERLANDS = code('NETHERLANDS')
    NEW_ZEALAND = code('NEW ZEALAND')
    NORWAY = code('NORWAY')
    PORTUGAL = code('PORTUGAL')
    SINGAPORE = code('SINGAPORE')
    SPAIN = code('SPAIN')
    SWEDEN = code('SWEDEN')
    SWITZERLAND = code('SWITZERLAND')
    UNITED_KINGDOM = code(
        'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND'
    )
    UNITED_STATES = code('UNITED STATES OF AMERICA')
