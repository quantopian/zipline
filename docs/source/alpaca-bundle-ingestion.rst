Alpaca Data Bundle
=====================

| Out of the box, I support Alpaca as a data source for data ingestion.
| If you haven't created an account, start with that: https://app.alpaca.markets/signup.
| Why? You could get price data for free using the Alpaca data API. Free data, is not a common thing.
| Any other vendor could be added and you are not obligated to use that.

How To Use It
-----------------

| I currently only support daily data but free minute data will soon follow.
| To ingest daily data bundle using the alpaca api, you need to follow these steps:
* The bundle is defined in this file: ``zipline/data/bundles/alpaca_api.py``
* There a method called ``initialize_client()``, it relies on the fact that you define your
  alpaca credentials in a file called ``alpaca.yaml`` in your root directory.
  it should look like this:

  .. code-block:: yaml

    key_id: "<YOUR-KEY>"
    secret: "<YOUR-SECRET>"
    base_url: https://paper-api.alpaca.markets
  ..
* you need to define your zipline root in an environment variable (This is where the
  ingested data will be stored). should be something like this:

  .. code-block:: yaml

    ZIPLINE_ROOT=~/.zipline
  ..
  | It means you could basically put it anywhere you want as long as you always use
  that as your zipline root.

  | It also means that different bundles could have different locations.

* By defauilt the bundle ingests 30 days backwards, but you can change that under the
  ``__main__`` section.
| The ingestion process for daily data using Alpaca is extremely fast due to the Alpaca
  API allowing to query 200 equities in one api call.

Notes
))))))))

* You are ready to backtest or paper trade using the pipeline functionality.
* You should repeat this process daily since every day you will have new price data.
* This data doesn't include Fundamental data, only price data so we'll need to handle it separately.
