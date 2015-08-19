Overview
========

Using Built-In Filters and Factors
----------------------------------

.. code-block:: Python

   from zipline.api import add_filter, add_factor
   from zipline.modelling.factors import VWAP

   def initalize(context):

       vwap = VWAP(window_length=4)
       add_factor(vwap, name='vwap')

       no_penny_stocks = (vwap > 1.0)
       add_filter(no_penny_stocks)

   def handle_data(context, data):
       factors = data.factors

Custom Filters and Factors
--------------------------

.. _`Directed Acyclic Graph`: https://en.wikipedia.org/wiki/Directed_acyclic_graph
.. rubric:: Footnotes
.. [#dasknote] This approach to the problem of working with large datasets is
               similar to that of many other PyData ecosystem libraries, most
               notably the Dask_ project.

.. _Dask: http://dask.pydata.org
