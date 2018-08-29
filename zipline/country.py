"""Canonical definitions of country code constants.
"""
from iso3166 import countries_by_name


def code(name):
    return countries_by_name[name].alpha2


class CountryCode(object):
    """A simple namespace of iso3166 alpha2 country codes.
    """
    CANADA = code('CANADA')
    GERMANY = code('GERMANY')
    UNITED_KINGDOM = code(
        'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND'
    )
    UNITED_STATES = code('UNITED STATES OF AMERICA')
