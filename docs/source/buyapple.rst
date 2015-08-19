Quickstart
^^^^^^^^^^

In this example, we introduce the basic concepts required for building a
trading algorithm with Zipline.


Lessions
""""""""
* Loading Input Data
* Defining Your Algorithm Logic
* Running Your Algorithm


Algorithm
"""""""""

.. code-block:: Python

   >>> from zipline import TradingAlgorithm
   >>> from zipline.api import order, symbol
   >>> from zipline.data import load_from_yahoo

   >>> def my_initialize(context):
   ...     pass

   >>> def my_handle_data(context, data):
           """Order one share of AAPL every day."""
   ...     order(symbol("AAPL"), 1)

   >>> algo = TradingAlgorithm(initialize=my_initialize, handle_data=my_handle_data)
   >>> inputs = load_bars_from_yahoo(
   ...     stocks=['AAPL'],
   ...     start='2013-01-01',
   ...     end='2014-01-01',
   ... )
   >>> algo.run(inputs)

