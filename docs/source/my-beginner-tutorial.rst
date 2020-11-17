Zipline Trader Beginner Tutorial
===================================


zipline-trader is an open-source algorithmic trading simulator written in
Python.

The source can be found here: `source-code`_

This tutorial assumes that you have zipline-trader correctly installed.

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