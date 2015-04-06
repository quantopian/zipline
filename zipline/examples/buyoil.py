from zipline.api import (
    order_target, record, symbol, history, add_history, order_target_percent
)

def initialize(context):
    # Register 2 histories that track daily prices,
    # one with a 100 window and one with a 300 day window
    add_history(100, '1d', 'price')
    add_history(300, '1d', 'price')

    context.i = 0

    context.clk = symbol('CLK15')
    context.clj = symbol('CLJ15')
    context.aapl = symbol('AAPL')


def handle_data(context, data):
    # Skip first 50 days to get full windows
    context.i += 1

    # Compute averages
    # history() has to be called with the same params
    # from above and returns a pandas dataframe.

    # Save values for later inspection
    record(CLK15=data[context.clk].price,
           CLJ15=data[context.clj].price,
           AAPL=data[context.aapl].price)

    if context.i > 20:
        order_target_percent(context.clk, -.33)
    if context.i > 30:
        order_target_percent(context.clj, -.33)
    if context.i > 40:
        order_target_percent(context.aapl, .33)
