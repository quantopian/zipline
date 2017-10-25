"""
Script for rebuilding the samples for the Quandl tests.
"""
from __future__ import print_function

import requests
from zipfile import ZipFile
from six.moves.urllib.parse import urlencode
from zipline.testing import test_resource_path, write_compressed

QUANDL_DATA_URL = (
    'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.csv?'
)


def format_table_query(api_key,
                       start_date,
                       end_date,
                       symbols):
    query_params = [
        ('api_key', api_key),
        ('date.gte', start_date),
        ('date.lte', end_date),
        ('ticker', ','.join(symbols)),
    ]
    return (
        QUANDL_DATA_URL + urlencode(query_params)
    )


def zipfile_path(file_name, file_ext):
    return test_resource_path('quandl_samples', file_name + file_ext)


def main():
    api_key = 'TJ8g_Jpo_NUjRMMnqfkW'
    start_date = '2014-1-1'
    end_date = '2015-1-1'
    symbols = 'AAPL', 'BRK_A', 'MSFT', 'ZEN'

    print('Downloading equity data')
    url = format_table_query(
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols
    )
    print('Fetching from %s' % url)
    response = requests.get(url)
    response.raise_for_status()

    path = zipfile_path('QUANDL_SAMPLE_TABLE', '.csv')
    print('Writing compressed data to %s' % path)
    with open(path, 'w+') as data_table:
        data_table.write(response.content)

    archive_path = zipfile_path('QUANDL_ARCHIVE', '.zip')

    with ZipFile(archive_path, 'w') as zip_file:
        zip_file.write('QUANDL_SAMPLE_TABLE.csv')
    print('Writing mock metadata')
    cols = (
        'file.link',
        'file.status',
        'file.data_snapshot_time',
        'datatable.last_refreshed_time\n'
    )
    row = (
        'https://file_url.mock.quandl',
        'fresh',
        '2017-10-17 23:48:25 UTC',
        '2017-10-17 23:48:15 UTC\n'
    )
    metadata = ','.join(cols) + ','.join(row)
    path = zipfile_path('metadata', '.csv.gz')
    print('Writing compressed data to %s' % path)
    write_compressed(path, metadata)


if __name__ == '__main__':
    main()
