"""
Script for rebuilding the samples for the Quandl tests.
"""
import os
from os.path import (
    dirname,
    join,
    realpath,
)
import requests
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from urllib.parse import urlencode
from zipline.testing import write_compressed
from zipline.data.bundles.quandl import QUANDL_DATA_URL

TEST_RESOURCE_PATH = join(
    dirname(dirname(dirname(realpath(__file__)))), "resources"  # zipline_repo/tests
)


def format_table_query(api_key, start_date, end_date, symbols):
    query_params = [
        ("api_key", api_key),
        ("date.gte", start_date),
        ("date.lte", end_date),
        ("ticker", ",".join(symbols)),
    ]
    return QUANDL_DATA_URL + urlencode(query_params)


def zipfile_path(file_name):
    return join(TEST_RESOURCE_PATH, "quandl_samples", file_name)


def main():
    api_key = os.environ.get("QUANDL_API_KEY")
    start_date = "2014-1-1"
    end_date = "2015-1-1"
    symbols = "AAPL", "BRK_A", "MSFT", "ZEN"

    url = format_table_query(
        api_key=api_key, start_date=start_date, end_date=end_date, symbols=symbols
    )
    print("Fetching equity data from %s" % url)
    response = requests.get(url)
    response.raise_for_status()

    archive_path = zipfile_path("QUANDL_ARCHIVE.zip")
    print("Writing compressed table to %s" % archive_path)
    with ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr(
            "QUANDL_SAMPLE_TABLE.csv",
            BytesIO(response.content).getvalue(),
            ZIP_DEFLATED,
        )
    print("Writing mock metadata")
    cols = (
        "file.link",
        "file.status",
        "file.data_snapshot_time",
        "datatable.last_refreshed_time\n",
    )
    row = (
        "https://file_url.mock.quandl",
        "fresh",
        "2017-10-17 23:48:25 UTC",
        "2017-10-17 23:48:15 UTC\n",
    )
    metadata = ",".join(cols) + ",".join(row)
    path = zipfile_path("metadata.csv.gz")
    print("Writing compressed metadata to %s" % path)
    write_compressed(path, metadata)


if __name__ == "__main__":
    main()
