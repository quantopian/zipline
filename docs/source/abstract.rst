
Zipline Trader - *Documentation under construction*
==========================================================

Welcome to zipline-trader, the on-premise trading platform built on top of Quantopian's
`zipline <https://github.com/quantopian/zipline>`_.


.. image:: ../../images/zipline-live2.png
   :width: 600


Quantopian closed their services, so this project tries to be a suitable replacement.

zipline-trader is based on previous projects and work:

- `zipline <https://github.com/quantopian/zipline>`_ project.
- `zipline-live <http://www.zipline-live.io>`_ project.
- `zipline-live2 <https://github.com/shlomikushchi/zipline-live2>`_ project.

zipline-live and zipline-live2 are past iterations of this project and this is the up to date project.

After Quantopian closed their services, this project was updated to supply a suitable and
sustainable replacement for users that want to run their algorithmic trading on their own without
relying on online services that may disappear one day. It  is designed to be an extensible, drop-in replacement for
zipline with multiple brokerage support to enable on premise trading of zipline algorithms.

I recommend using python 3.6

Supported Data Sources
--------------------------
Out of the box, zipline-trader supports Alpaca as a free data source. You could use the quantopian-quandl bundle used
in old zipline versions or any other bundle you create (how to create a bundle on a later section)

Supported Brokers
------------------------
Currently 2 brokers are supported:
 * Alpaca
 * IB