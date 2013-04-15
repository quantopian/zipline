**********
Quickstart
**********

Dual-Moving Average Example
===========================

The following code implements a simple dual moving average algorithm
and tests it on data extracted from yahoo finance.

.. code-block:: python

    from zipline.algorithm import TradingAlgorithm
    from zipline.transforms import MovingAverage
    from zipline.utils.factory import load_from_yahoo

    class DualMovingAverage(TradingAlgorithm):
        """Dual Moving Average algorithm.
        """
        def initialize(self, short_window=200, long_window=400):
            # Add 2 mavg transforms, one with a long window, one
            # with a short window.
            self.add_transform(MovingAverage, 'short_mavg', ['price'],
                            market_aware=True,
                            days=short_window)

            self.add_transform(MovingAverage, 'long_mavg', ['price'],
                            market_aware=True,
                            days=long_window)

            # To keep track of whether we invested in the stock or not
            self.invested = False

            self.short_mavg = []
            self.long_mavg = []


        def handle_data(self, data):
            if (data['AAPL'].short_mavg['price'] > data['AAPL'].long_mavg['price']) and not self.invested:
                self.order('AAPL', 100)
                self.invested = True
            elif (data['AAPL'].short_mavg['price'] < data['AAPL'].long_mavg['price']) and self.invested:
                self.order('AAPL', -100)
                self.invested = False

            # Save mavgs for later analysis.
            self.short_mavg.append(data['AAPL'].short_mavg['price'])
            self.long_mavg.append(data['AAPL'].long_mavg['price'])

    data = load_from_yahoo()
    dma = DualMovingAverage()
    results = dma.run(data)

You can find other examples in the zipline/examples directory.

