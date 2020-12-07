Going Live
=============
| Old zipline-live/zipline-live2 users know the command line tool that used to go live. e.g:
.. code-block::

    zipline run -f ~/zipline-algos/demo.py --state-file ~/zipline-algos/demo.state --realtime-bar-target ~/zipline-algos/realtime-bars/ --broker ib --broker-uri localhost:7496:1232 --bundle quantopian-quandl --data-frequency minute

| You could still do that, but you shouldn't.
| The recommended way is to run inside a python file, preferably using an IDE so you could
  debug your code with breakpoints and memory view.
| I will show you exactly how to do so, providing a template that you could just copy
  and develop your code in. It will be the exact same example I provided in the backtest section
  with a different call to ``run_algorithm`` which will connect to a broker (you could use IB or Alpaca for now)

Important notes before we start
---------------------------------
| You of course need to have everything already installed, so go to the `install`_
  part if you don't
| You need to have an ingested data bundle. You could use the `Alpaca Data Bundle`_ or
  user your own.
| You need to have some understanding on how a zipline algo is used. (`Beginner Tutorial`_)
| You need to set the ``ZIPLINE_ROOT`` env variable to point to your ingested data bundle.
| You need to store the alpaca credentials in a file called ``alpaca.yaml`` in your root directory.
  it should look like this:

  .. code-block:: yaml

    key_id: "<YOUR-KEY>"
    secret: "<YOUR-SECRET>"
    base_url: https://paper-api.alpaca.markets
  ..

Algo Template
---------------
| This next code snippet is a simple algorithm that is self contained. You could
  copy that into a python file and just execute it. It will connect to the Alpaca broker.

.. code-block:: python

    import os
    import yaml
    import pytz
    import pandas as pd
    from datetime import datetime
    import pandas_datareader.data as yahoo_reader

    from zipline.utils.calendars import get_calendar
    from zipline.api import order_target, symbol
    from zipline.data import bundles
    from zipline import run_algorithm
    from zipline.gens.brokers.alpaca_broker import ALPACABroker


    def get_benchmark(symbol=None, start=None, end=None):
        bm = yahoo_reader.DataReader(symbol,
                                     'yahoo',
                                     pd.Timestamp(start),
                                     pd.Timestamp(end))['Close']
        bm.index = bm.index.tz_localize('UTC')
        return bm.pct_change(periods=1).fillna(0)


    def initialize(context):
        pass


    def handle_data(context, data):
        order_target(context.equity, 100)


    def before_trading_start(context, data):
        context.equity = symbol("AMZN")


    if __name__ == '__main__':
        bundle_name = 'alpaca_api'
        bundle_data = bundles.load(bundle_name)

        with open("alpaca.yaml", mode='r') as f:
            o = yaml.safe_load(f)
            os.environ["APCA_API_KEY_ID"] = o["key_id"]
            os.environ["APCA_API_SECRET_KEY"] = o["secret"]
            os.environ["APCA_API_BASE_URL"] = o["base_url"]
        broker = ALPACABroker()

        # Set the trading calendar
        trading_calendar = get_calendar('NYSE')

        start = pd.Timestamp(datetime(2020, 1, 1, tzinfo=pytz.UTC))
        end = pd.Timestamp.utcnow()

        run_algorithm(start=start,
                      end=end,
                      initialize=initialize,
                      handle_data=handle_data,
                      capital_base=100000,
                      benchmark_returns=get_benchmark(symbol="SPY",
                                                      start=start.date().isoformat(),
                                                      end=end.date().isoformat()),
                      bundle='alpaca_api',
                      broker=broker,
                      state_filename="./demo.state",
                      trading_calendar=trading_calendar,
                      before_trading_start=before_trading_start,
                      data_frequency='daily'
                      )


..

.. _`install` : ../latest/install.html
.. _`Alpaca Data Bundle`: ../latest/alpaca-bundle-ingestion.html
.. _`Beginner Tutorial`: ../latest/beginner-tutorial.html