from zipline.api import order_target, record, symbol, history, add_history


def initialize(context):
    # Register 2 histories that track daily prices,
    # one with a 100 window and one with a 300 day window
    add_history(100, '1d', 'price')
    add_history(300, '1d', 'price')

    context.i = 0


def handle_data(context, data):
    # Skip first 50 days to get full windows
    context.i += 1
    if context.i < 50:
        return

    # Compute averages
    # history() has to be called with the same params
    # from above and returns a pandas dataframe.

    clk = symbol('CLK15')
    clj = symbol('CLJ15')

    # order_target(clk, 100)
    order_target(clj, -100)

    # Save values for later inspection
    record(CLK15=data[clk].price,
           CLJ15=data[clj].price)
