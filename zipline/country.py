"""Canonical definitions of country code constants.
"""
from iso3166 import countries_by_name


class CountryCode(object):
    """A simple namespace of iso3166 alpha2 country codes.
    """
    CANADA = countries_by_name['CANADA'].alpha2
    GERMANY = countries_by_name['GERMANY'].alpha2
    UNITED_KINGDOM = countries_by_name[
        'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND'
    ].alpha2
    UNITED_STATES = countries_by_name['UNITED STATES OF AMERICA'].alpha2
