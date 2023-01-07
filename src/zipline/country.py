"""Canonical definitions of country code constants.
"""
from iso3166 import countries_by_name


def code(name):
    return countries_by_name[name].alpha2


class CountryCode:
    """A simple namespace of iso3166 alpha2 country codes."""

    ARGENTINA = code("ARGENTINA")
    AUSTRALIA = code("AUSTRALIA")
    AUSTRIA = code("AUSTRIA")
    BELGIUM = code("BELGIUM")
    BRAZIL = code("BRAZIL")
    CANADA = code("CANADA")
    CHILE = code("CHILE")
    CHINA = code("CHINA")
    COLOMBIA = code("COLOMBIA")
    CZECH_REPUBLIC = code("CZECHIA")
    DENMARK = code("DENMARK")
    FINLAND = code("FINLAND")
    FRANCE = code("FRANCE")
    GERMANY = code("GERMANY")
    GREECE = code("GREECE")
    HONG_KONG = code("HONG KONG")
    HUNGARY = code("HUNGARY")
    INDIA = code("INDIA")
    INDONESIA = code("INDONESIA")
    IRELAND = code("IRELAND")
    ISRAEL = code("ISRAEL")
    ITALY = code("ITALY")
    JAPAN = code("JAPAN")
    MALAYSIA = code("MALAYSIA")
    MEXICO = code("MEXICO")
    NETHERLANDS = code("NETHERLANDS")
    NEW_ZEALAND = code("NEW ZEALAND")
    NORWAY = code("NORWAY")
    PAKISTAN = code("PAKISTAN")
    PERU = code("PERU")
    PHILIPPINES = code("PHILIPPINES")
    POLAND = code("POLAND")
    PORTUGAL = code("PORTUGAL")
    RUSSIA = code("RUSSIAN FEDERATION")
    SINGAPORE = code("SINGAPORE")
    SOUTH_AFRICA = code("SOUTH AFRICA")
    SOUTH_KOREA = code("KOREA, REPUBLIC OF")
    SPAIN = code("SPAIN")
    SWEDEN = code("SWEDEN")
    SWITZERLAND = code("SWITZERLAND")
    TAIWAN = code("TAIWAN, PROVINCE OF CHINA")
    THAILAND = code("THAILAND")
    TURKEY = code("TÜRKIYE")
    UNITED_KINGDOM = code("UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND")
    UNITED_STATES = code("UNITED STATES OF AMERICA")
