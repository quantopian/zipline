import pandas as pd
import os

# ASSUMPTIONS
# This ingest function makes some assumptions.
# The user's home folder should have have a folder named "futures".
# Inside that folder there must be one file named meta.csv.
# The file meta.csv must be a csv with the following columns: symbol,
# root_symbol, notice_date, expiration_date, tick_size, multiplier, and
# auto_close_date.
# The "futures" folder must in addition contain a bunch of .csv files
# containing actual price data. The data files must contain the following
# columns: time, open_price, high, low, close_price, volume.
# The symbol column in the meta.csv file will be used to generate the
# names of the data files (e.g. symbol CLG16 will be expected to be in file
# CLG16.csv).
# The entire meta.csv will be read as a dataframe, and the index will be
# used as sid.
# The exchanges for both the symbols and the root_symbols will be set
# to "NYSE"

DATA_FOLDER = os.path.join(os.path.expanduser('~'), 'futures')


def futurescsv(symbols, start=None, end=None):

    def ingest(environ, asset_db_writer, minute_bar_writer, daily_bar_writer,
               adjustment_writer, calendar, start_session, end_session,
               cache, show_progress, output_dir, start=start, end=end):

        # Load metadata from meta.csv
        metadata = pd.read_csv(os.path.join(DATA_FOLDER, 'meta.csv'))
        for d in ('notice_date', 'expiration_date', 'auto_close_date'):
            metadata[d] = pd.to_datetime(metadata[d])
        metadata['start_date'] = pd.Series([pd.NaT] * len(metadata))
        metadata['end_date'] = pd.Series([pd.NaT] * len(metadata))

        data = []

        # DataFrame.iterrows() iterates over (index, row)
        for sid, row in metadata.iterrows():
            symbol = row['symbol']

            data_path = os.path.join(DATA_FOLDER, symbol+'.csv')

            df = pd.read_csv(data_path,
                             usecols=['time', 'open_price', 'high',
                                      'low', 'close_price', 'volume'],
                             index_col='time',
                             parse_dates=True,
                             )
            df.rename(columns={
                    'open_price': 'open',
                    'close_price': 'close',
                    },
                    inplace=True,
            )
            df.fillna(0, inplace=True)
            # df['volume'] = df['volume']/1000.0
            data.append((sid, df))

            metadata.loc[sid, 'start_date'] = df['close'].first_valid_index()
            metadata.loc[sid, 'end_date'] = df['close'].last_valid_index()

        metadata = metadata[['start_date', 'end_date', 'auto_close_date',
                             'symbol', 'root_symbol', 'notice_date',
                             'expiration_date', 'tick_size', 'multiplier']]
        metadata['exchange'] = 'NYSE'

        daily_bar_writer.write(data, show_progress=True)

        root_symbols = metadata.root_symbol.unique()
        root_symbols = pd.DataFrame(root_symbols, columns=['root_symbol'])
        root_symbols['root_symbol_id'] = root_symbols.index.values
        root_symbols['exchange'] = 'NYSE'

        asset_db_writer.write(futures=metadata, root_symbols=root_symbols)

        adjustment_writer.write()

    return ingest
