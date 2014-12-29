import sys
import logbook
import numpy as np

zipline_logging = logbook.NestedSetup([
    logbook.NullHandler(level=logbook.DEBUG, bubble=True),
    logbook.StreamHandler(sys.stdout, level=logbook.INFO),
    logbook.StreamHandler(sys.stderr, level=logbook.ERROR),
])
zipline_logging.push_application()

from zipline.api import (
    symbols,
    schedule_function,
    date_rules,
    time_rules,
    add_history,
    history,
    order_target_percent,
    record,
    get_open_orders,
    get_datetime
)


def initialize(context, eps=1, window_length=30):
    context.stocks = symbols('CERN', 'DLTR', 'ROST', 'MSFT', 'SBUX')
    context.i = 0
    context.m = len(context.stocks)
    context.b_t = np.ones(context.m) / context.m
    context.eps = eps
    context.window_length = window_length
    context.init = False
    context.previous_datetime = None
    add_history(window_length, '1d', 'price')
    add_history(window_length, '1d', 'volume')
    schedule_function(daily, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open())


def handle_data(context, data):
    pass


def daily(context, data):
    context.i += 1
    if context.i < context.window_length:
        return

    cash = context.portfolio.cash
    record(cash=cash)

    if not context.init:
        # Equal weighting portfolio
        for stock, percent in zip(context.stocks, context.b_t):
            order_target_percent(stock, percent)
        context.init = True

    # skip tic if any orders are open or any stocks did not trade
    for stock in context.stocks:
        if bool(get_open_orders(stock)) or \
           data[stock].datetime < get_datetime():
            return

    # compute current portfolio allocations
    for i, stock in enumerate(context.stocks):
        context.b_t[i] = context.portfolio.positions[
            stock].amount * data[stock].price

    # Bring portfolio vector to unit length
    context.b_t = context.b_t / np.sum(context.b_t)

    # Compute new portfolio weights according to OLMAR algo.
    size = context.window_length - 2
    b_norm = np.zeros((context.m, size))
    x_tilde = np.zeros((context.m, size))
    for k in range(size):
        b_norm[:, k], x_tilde[:, k] = olmar(context, k + 3)

    s = np.zeros(size)
    b_norm_opt = np.zeros(context.m)
    s_sum = 0
    for k in range(size):
        s[k] = np.dot(b_norm[:, k], x_tilde[:, k])
        b_norm[:, k] = np.multiply(s[k], b_norm[:, k])
        b_norm_opt += b_norm[:, k]
        s_sum += s[k]

    b_norm_opt = np.divide(b_norm_opt, s_sum)

    print(b_norm_opt)

    # Rebalance Portfolio
    for stock, percent in zip(context.stocks, b_norm_opt):
        order_target_percent(stock, percent)


def olmar(context, window):
    """Logic of the olmar algorithm.

    :Returns: b_norm : vector for new portfolio
    """

    # get history -- prices and volums of the last 5 days (at close)
    p = history(context.window_length, '1d', 'price')
    v = history(context.window_length, '1d', 'volume')

    prices = p.ix[context.window_length - window:-1]
    volumes = v.ix[context.window_length - window:-1]

    # find relative moving volume weighted average price for each secuirty
    x_tilde = np.zeros(context.m)
    for i, stock in enumerate(context.stocks):
        vwa_price = np.dot(
            prices[stock], volumes[stock]) / np.sum(volumes[stock])
        x_tilde[i] = vwa_price / prices[stock].ix[-1]

    ###########################
    # Inside of OLMAR (algo 2)
    x_bar = x_tilde.mean()

    # Calculate terms for lambda (lam)
    dot_prod = np.dot(context.b_t, x_tilde)
    num = context.eps - dot_prod
    denom = (np.linalg.norm((x_tilde - x_bar))) ** 2

    # test for divide-by-zero case
    if denom == 0.0:
        lam = 0  # no portolio update
    else:
        lam = max(0, num / denom)

    b = context.b_t + lam * (x_tilde - x_bar)

    b_norm = simplex_projection(b)

    return b_norm, x_tilde


def simplex_projection(v, b=1):
    """Projection vectors to the simplex domain

Implemented according to the paper: Efficient projections onto the
l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
Optimization Problem: min_{w}\| w - v \|_{2}^{2}
s.t. sum_{i=1}^{m}=z, w_{i}\geq 0

Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
Output: Projection vector w

:Example:
>>> proj = simplex_projection([.4 ,.3, -.4, .5])
>>> print proj
array([ 0.33333333, 0.23333333, 0. , 0.43333333])
>>> print proj.sum()
1.0

Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
Python-port: Copyright 2012 by Thomas Wiecki (thomas.wiecki@gmail.com).
"""

    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p + 1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho + 1)])
    w = (v - theta)
    w[w < 0] = 0
    return w
