Zipline Trader Beginner Tutorial
===================================


zipline-trader is an open-source algorithmic trading simulator written in
Python.

The source can be found here: `source-code`_

This tutorial assumes that you have zipline-trader correctly installed.

Create Data Bundle
---------------------------
| Out of the box, I support Alpaca as a data source for data ingestion.
| If you haven't created an account, start with that: https://app.alpaca.markets/signup.

Alpaca Data Bundle
))))))))))))))))))))))))))))))
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
| It also means you could basically put it anywhere you want as long as you always use
  that as your zipline root
| It also means that different bundles could have different locations.

* By defauilt the bundle ingests 30 days backwards, but you can change that under the
  ``__main__`` section.
| The ingestion process for daily data using Alpaca is extremely fast due to the Alpaca
  API allowing to query 200 equities in one api call.

Research & Backtesting
--------------------------
| The same capabilities existing in zipline, exist in zipline-trader.
| You could use python code to backtest/optimize your algorithms, alternatively you
  could use Jupyter Notebooks to research your ideas.

Backtesting
)))))))))))))

Performing a backtest is very similar to how it is done with zipline or in Quantopian.
Unlike the recommended way by zipline using the command line interface. e.g:
.. code-block:: bash

   $ zipline run -f zipline_repo/zipline/examples/dual_moving_average.py --start 2015-1-1 --end 2018-1-1 --bundle quantopian-quandl -o out.pickle --capital-base 10000
..

| I do not recommend using it, and will not document how to. (IMO) It is not very pythonic and
  not very powerful. Developers need the ability to debug their code. I will show you how to
  create a python file that could be executed with the python binary or in an IDE allowing
  you to use breakpoints to debug your code.


-----------

The original zipline tutorial:

.. include:: beginner-tutorial.rst


.. _`source-code` : https://github.com/shlomikushchi/zipline-trader