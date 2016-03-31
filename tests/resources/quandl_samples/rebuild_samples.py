"""
Script for rebuilding the samples for the Quandl tests.
"""
from __future__ import print_function

import pandas as pd
import requests


from zipline.data.quandl import format_wiki_url
from zipline.utils.test_utils import test_resource_path, write_compressed


def zipfile_path(symbol):
    return test_resource_path('quandl_samples', symbol + '.csv.gz')


def main():
    start_date = pd.Timestamp('2014')
    end_date = pd.Timestamp('2015')
    symbols = ['AAPL', 'MSFT', 'BRK_A', 'ZEN']
    for sym in symbols:
        url = format_wiki_url(
            api_key=None,
            symbol=sym,
            start_date=start_date,
            end_date=end_date,
        )
        print("Fetching from %s" % url)
        response = requests.get(url)
        response.raise_for_status()

        path = zipfile_path(sym)
        print("Writing compressed data to %s" % path)
        write_compressed(path, response.content)


if __name__ == '__main__':
    main()
