#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import getopt
import traceback
import numpy as np
import pandas as pd
import datetime
import logging
import tables
import gzip
import glob
import os
import random
import csv
import time
from six import print_

FORMAT = "%(asctime)-15s -8s %(message)s"

logging.basicConfig(format=FORMAT, level=logging.INFO)


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


OHLCTableDescription = {'sid': tables.StringCol(14, pos=2),
                        'dt': tables.Int64Col(pos=1),
                        'open': tables.Float64Col(dflt=np.NaN, pos=3),
                        'high': tables.Float64Col(dflt=np.NaN, pos=4),
                        'low': tables.Float64Col(dflt=np.NaN, pos=5),
                        'close': tables.Float64Col(dflt=np.NaN, pos=6),
                        "volume": tables.Int64Col(dflt=0, pos=7)}


def process_line(line):
    dt = np.datetime64(line["dt"]).astype(np.int64)
    sid = line["sid"]
    open_p = float(line["open"])
    high_p = float(line["high"])
    low_p = float(line["low"])
    close_p = float(line["close"])
    volume = int(line["volume"])
    return (dt, sid, open_p, high_p, low_p, close_p, volume)


def parse_csv(csv_reader):
    previous_date = None
    data = []
    dtype = [('dt', 'int64'), ('sid', '|S14'), ('open', float),
             ('high', float), ('low', float), ('close', float),
             ('volume', int)]
    for line in csv_reader:
        row = process_line(line)
        current_date = line["dt"][:10].replace("-", "")
        if previous_date and previous_date != current_date:
            rows = np.array(data, dtype=dtype).view(np.recarray)
            yield current_date, rows
            data = []
        data.append(row)
        previous_date = current_date


def merge_all_files_into_pytables(file_dir, file_out):
    """
    process each file into pytables
    """
    start = None
    start = datetime.datetime.now()
    out_h5 = tables.openFile(file_out,
                             mode="w",
                             title="bars",
                             filters=tables.Filters(complevel=9,
                                                    complib='zlib'))
    table = None
    for file_in in glob.glob(file_dir + "/*.gz"):
        gzip_file = gzip.open(file_in)
        expected_header = ["dt", "sid", "open", "high", "low", "close",
                           "volume"]
        csv_reader = csv.DictReader(gzip_file)
        header = csv_reader.fieldnames
        if header != expected_header:
            logging.warn("expected header %s\n" % (expected_header))
            logging.warn("header_found %s" % (header))
            return

        for current_date, rows in parse_csv(csv_reader):
            table = out_h5.createTable("/TD", "date_" + current_date,
                                       OHLCTableDescription,
                                       expectedrows=len(rows),
                                       createparents=True)
            table.append(rows)
            table.flush()
        if table is not None:
            table.flush()
    end = datetime.datetime.now()
    diff = (end - start).seconds
    logging.debug("finished  it took %d." % (diff))


def create_fake_csv(file_in):
    fields = ["dt", "sid", "open", "high", "low", "close", "volume"]
    gzip_file = gzip.open(file_in, "w")
    dict_writer = csv.DictWriter(gzip_file, fieldnames=fields)
    current_dt = datetime.date.today() - datetime.timedelta(days=2)
    current_dt = pd.Timestamp(current_dt).replace(hour=9)
    current_dt = current_dt.replace(minute=30)
    end_time = pd.Timestamp(datetime.date.today())
    end_time = end_time.replace(hour=16)
    last_price = 10.0
    while current_dt < end_time:
        row = {}
        row["dt"] = current_dt
        row["sid"] = "test"
        last_price += random.randint(-20, 100) / 10000.0
        row["close"] = last_price
        row["open"] = last_price - 0.01
        row["low"] = last_price - 0.02
        row["high"] = last_price + 0.02
        row["volume"] = random.randint(10, 1000) * 10
        dict_writer.writerow(row)
        current_dt += datetime.timedelta(minutes=1)
        if current_dt.hour > 16:
            current_dt += datetime.timedelta(days=1)
            current_dt = current_dt.replace(hour=9)
            current_dt = current_dt.replace(minute=30)
    gzip_file.close()


def main(argv=None):
    """
    This script cleans minute bars into pytables file
    data_source_tables_gen.py
    [--tz_in] sets time zone of data only reasonably fast way to use
    time.tzset()
    [--dir_in] iterates through directory provided of csv files in gzip form
    in form:
    dt, sid, open, high, low, close, volume
    2012-01-01T12:30:30,1234HT,1, 2,3,4.0
    [--fake_csv] creates a fake sample csv to iterate through
    [--file_out] determines output file
    """
    if argv is None:
        argv = sys.argv
    try:
        dir_in = None
        file_out = "./all.h5"
        fake_csv = None
        try:
            opts, args = getopt.getopt(argv[1:], "hdft",
                                       ["help",
                                        "dir_in=",
                                        "debug",
                                        "tz_in=",
                                        "fake_csv=",
                                        "file_out="])
        except getopt.error as msg:
            raise Usage(msg)
        for opt, value in opts:
            if opt in ("--help", "-h"):
                print_(main.__doc__)
            if opt in ("-d", "--debug"):
                logging.basicConfig(format=FORMAT,
                                    level=logging.DEBUG)
            if opt in ("-d", "--dir_in"):
                dir_in = value
            if opt in ("-o", "--file_out"):
                file_out = value
            if opt in ("--fake_csv"):
                fake_csv = value
            if opt in ("--tz_in"):
                os.environ['TZ'] = value
                time.tzset()
        try:
            if dir_in:
                merge_all_files_into_pytables(dir_in, file_out)
            if fake_csv:
                create_fake_csv(fake_csv)
        except Exception:
            error = "An unhandled error occured in the"
            error += "data_source_tables_gen.py script."
            error += "\n\nTraceback:\n"
            error += '-' * 70 + "\n"
            error += "".join(traceback.format_tb(sys.exc_info()[2]))
            error += repr(sys.exc_info()[1]) + "\n"
            error += str(sys.exc_info()[1]) + "\n"
            error += '-' * 70 + "\n"
            print_(error)
    except Usage as err:
        print_(err.msg)
        print_("for help use --help")
        return 2

if __name__ == "__main__":
    sys.exit(main())
