"""
Custom Bundle for Loading the Zacks Equity Prices dataset from Quandl.
https://www.quandl.com/databases/ZEP
This is a premium dataset, and requires valid permissions to download.

Created by Peter Harrington on 3/1/17.
"""

import pandas as pd
import gc  # garbage collection

START_DATE = '2009-01-02'
METADATA_HEADERS = ['start_date', 'end_date', 'auto_close_date',
                    'symbol', 'exchange', 'asset_name']
UNWANTED_EXCHANGES = set(["OTC", "OTCBB", "INDX"])


def from_zacks_dump(file_name, dvdend_file=None, start=None, end=None):
    """
    Load data from a Zacks dump from Quandl, the data dump is assumed to be in
    a .csv file located at: file_name.

    https://www.quandl.com/databases/ZEP
    https://www.quandl.com/databases/ZEP/documentation/api
    Data goes back to 1984 (with only close prices), close and volume starting
    in 1990.
    Full OHLCV starts on 1999-04-05, an example line
    ITL,INTC,INTEL CORP,,NSDQ,USD,1999-04-14,30.68,30.75,28.125,28.5,10450200.0

    Tickers come from one of the following exchanges:
    'NYSE', 'NSDQ', 'OTC', 'ARCA', 'INDX', 'TSXV', 'OTCBB', 'TSX'
    This can take a long time on the order of hours.  Using the defaults on a
    modern laptop will take 1-2 hours.

    To use this make your ~/.zipline/extension.py look similar this:

    from zipline.data.bundles import register
    from zipline.data.bundles.zacks_quandl import from_zacks_dump

    register("ZacksQuandl",
         from_zacks_dump("/path/to/your/Zacks/dump/ZACKS_P_f53fjdkj993857.csv"),)

    """

    def ingest(environ,
               asset_db_writer,
               minute_bar_writer,  # unused
               daily_bar_writer,
               adjustment_writer,
               calendar,
               cache,
               show_progress,
               output_dir,
               # pass these as defaults to make them 'nonlocal' in py2
               start=start,
               end=end):

        print("starting ingesting data from: {}".format(file_name))

        # read in the whole dump (will require ~7GB of RAM)
        df = pd.read_csv(file_name, index_col='date',
                         parse_dates=['date'], na_values=['NA'])
        # drop unused columns
        df = df.drop(['comp_name_2', 'currency_code'], axis=1)
        # using inplace=true here instead of functional
        # approach saves peak memory usage

        df = df.ix[START_DATE:]  # drop dates before START_DATE
        gc.collect()  # force garbage collection to free up memory

        # drop row with NaNs or the loader will turn all columns to NaNs
        df = df.dropna()

        uv = df.m_ticker.unique()  # get unique m_tickers (Zacks primary key)

        # counter of valid securites, this will be our primary key
        sec_counter = 0
        data_list = []  # list to send to daily_bar_writer
        metadata_list = []  # list to send to asset_db_writer (metadata)

        # iterate over all the unique securities and pack data, and metadata
        # for writing
        for tkr in uv:
            df_tkr = df[df['m_ticker'] == tkr]

            row0 = df_tkr.ix[0]  # get metadata from row
            if row0["exchange"] in UNWANTED_EXCHANGES:  # skip OTC securities
                continue

            print(" preparing {} / {} ".format(row0["ticker"],
                                               row0["exchange"]))

            # update metadata; 'start_date', 'end_date', 'auto_close_date',
            # 'symbol', 'exchange', 'asset_name'
            metadata_list.append((df_tkr.index[0],
                                  df_tkr.index[-1],
                                  df_tkr.index[-1] + pd.Timedelta(days=1),
                                  row0["ticker"],
                                  row0["exchange"],
                                  row0["comp_name"]
                                  )
                                 )

            # drop metadata columns
            df_tkr = df_tkr.drop(['m_ticker', 'ticker',
                                  'comp_name', 'exchange'], axis=1)

            # pack data to be written by daily_bar_writer
            data_list.append((sec_counter, df_tkr))
            sec_counter += 1

        print("writing data for {} securities".format(len(metadata_list)))
        daily_bar_writer.write(data_list, show_progress=False)

        # write metadata
        asset_db_writer.write(equities=pd.DataFrame(metadata_list,
                                                    columns=METADATA_HEADERS))
        print("a total of {} securities were loaded into this bundle".format(
            sec_counter))
        # read in Dividend History
        """
        m_ticker,ticker,comp_name,comp_name_2,exchange,currency_code,div_ex_date,div_amt,per_end_date
        Z86Z,0425B,PCA INTL,PCA INTL,,USD,1997-06-09,0.07,1997-07-31

        div_ex_date is the date you are entitled to the dividend
        """
        if dvdend_file is None:
            adjustment_writer.write()
        else:
            dfd = pd.read_csv(dvdend_file, index_col='div_ex_date',
                              parse_dates=['div_ex_date', 'per_end_date'],
                              na_values=['NA'])

            dfd = dfd.ix[START_DATE:]  # drop old data
            # format dfd to have sid
            adjustment_writer.write(dividends=dfd)

    return ingest
