=======
Masking
=======

In this example, we use the same ``CustomFactor`` construction we used in
Lesson 3, but we introduce the concept of "masking" an operation on a factor.

Masking is an operation that allows you to compute a ``Factor`` in a manner
that "ignores" certain values.  A common use-case for this for ranking values
by a metric **after** filtering out values you don't want to consider.

We demonstrate this idea by building and adding a filter representing the top
50 stocks by market cap each day. We perform ``rank()`` operations on another
Factor, with and without our filter applied as a mask.

When ranking naively, the ranks are correctly ordered, but the absolute values
are scattered between 1 and ~8000, because our rank considered all assets
independently of whether or not we're going to filter them out in our final
results.  When ranking with our filter applied as a mask, all the values are
between 1 and 50, as expected.

.. code-block:: Python

    from zipline.modelling.factor import CustomFactor
    from zipline.modelling.factor.technical import VWAP, SimpleMovingAverage
    from zipline.data.equities import USEquityPricing


    class MarketCap(CustomFactor):

        inputs = [USEquityPricing.close, Fundamentals.valuation.shares_outstanding]
        window_length = 1

        def compute(self, today, assets, out, close, shares):
            out[:] = close[-1] * shares[-1]


    def initialize(context):
        add_factor(USEquityPricing.close.latest, 'close')

        # Note that we don't call add_factor on these Factors.
        # We don't need to store intermediate values if we're not going to use them
        sma_short = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30)
        sma_long = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=100)

        sma_quotient = sma_short / sma_long
        add_factor(sma_quotient, 'quotient')

        mkt_cap = MarketCap()
        add_factor(mkt_cap, 'mkt_cap')

        # Create and apply a filter representing the top 50 equities by MarketCap every day.
        mkt_cap_top_50 = mkt_cap.top(50)
        add_filter(mkt_cap_top_50)

        # Naively computing rank will not take into account
        # the fact that we only care about the top 50 by mkt_cap...
        naive_rank = sma_quotient.rank()
        add_factor(naive_rank, 'naive')

        # ...but we can tell the engine to ignore the values we're
        # filtering out by passing mask=market_cap_top_50 to rank.
        # top, bottom, and percentile_between also accept `mask` as arguments.
        masked_rank = sma_quotient.rank(mask=mkt_cap_top_50)
        add_factor(masked_rank, 'masked')


    def before_trading_start(context, data):
        context.my_factors = data.factors
        print "Number of securities returned = %s" % len(context.my_factors)
        print "Factors:\n" + str(context.my_factors.sort('quotient').head())
        print "Max Naive Rank: %f" % context.my_factors['naive'].max()
        print "Max Masked Rank: %f" % context.my_factors['masked'].max()


    def handle_data(context, data):
        # Quantopian's backtester barfs unless you call sid() somewhere at least once.
        order(sid(24), 1)
