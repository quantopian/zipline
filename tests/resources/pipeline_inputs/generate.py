"""
Quick and dirty script to generate test case inputs.
"""
from __future__ import print_function
from os.path import (
    dirname,
    join,
)
from pandas_datareader.data import DataReader

here = join(dirname(__file__))


def main():
    symbols = ['AAPL', 'MSFT', 'BRK-A']
    # Specifically chosen to include the AAPL split on June 9, 2014.
    for symbol in symbols:
        data = DataReader(
            symbol,
            'yahoo',
            start='2014-03-01',
            end='2014-09-01',
        )
        data.rename(
            columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
            },
            inplace=True,
        )
        del data['Adj Close']

        dest = join(here, symbol + '.csv')
        print("Writing %s -> %s" % (symbol, dest))
        data.to_csv(dest, index_label='day')

if __name__ == '__main__':
    main()
