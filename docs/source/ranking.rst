=====================
Combining and Ranking
=====================

In this algo, we build on the lessons of the first example. Here we show that
factors and can be combined and used together.

We also show how to use the ``rank()`` method and how to update your universe
with the results from ``data.factors``.

.. code-block:: Python

    from zipline.modelling.factor.technical import VWAP, SimpleMovingAverage
    from zipline.data.equities import USEquityPricing

    def initialize(context):

        sma_short = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30)
        add_factor(sma_short, 'sma_short')

        sma_long = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=100)
        add_factor(sma_long, 'sma_long')

        # Factors can be combined and create new factors
        sma_val = sma_short/sma_long
        add_factor(sma_val, 'sma_val')

        # You can automatically rank the resulting universe and add a factor with the ranking
        add_factor(sma_val.rank(), 'sma_rank')

        # filter out the penny stocks
        add_filter(sma_short > 0)

    def before_trading_start(context, data):
        context.my_factors = data.factors
        print "Number of securities returned = %s" % len(context.my_factors)

        # get the list of securities to short
        context.short_list = data.factors.sort(['sma_rank'], ascending=True).iloc[:200]

        # and the list of securities to long
        context.long_list = data.factors.sort(['sma_rank'], ascending=True).iloc[-200:]

        # update your universe with the SIDs of long and short securities
        update_universe(context.long_list.index.union(context.short_list.index))

    def handle_data(context, data):

        # you can then use the information to adjust your portfolio
        # for this example we will just print some information
        # about our long and short lists
        print "SHORT LIST"
        print len(context.short_list)
        log.info("\n" + str(context.short_list.sort(['sma_rank'], ascending=True).head()))
        print "LONG LIST"
        print len(context.long_list)
        log.info("\n" + str(context.long_list.sort(['sma_rank'], ascending=False).head()))
