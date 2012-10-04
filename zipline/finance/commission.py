class PerShare(object):
    """
    Calculates a commission for a transaction based on a per
    share cost.
    """

    def __init__(self, cost=0.03):
        """
        Cost parameter is the cost of a trade per-share. $0.03
        means three cents per share, which is a very conservative
        (quite high) for per share costs.
        """
        self.cost = cost

    def calculate(self, transaction):
        """
        returns a tuple of:
        (per share commission, total transaction commission)
        """
        return self.cost, abs(transaction.amount * self.cost)


class PerTrade(object):
    """
    Calculates a commission for a transaction based on a per
    trade cost.
    """

    def __init__(self, cost=5.0):
        """
        Cost parameter is the cost of a trade, regardless of
        share count. $5.00 per trade is fairly typical of
        discount brokers.
        """
        self.cost = cost

    def calculate(self, transaction):
        """
        returns a tuple of:
        (per share commission, total transaction commission)
        """
        if transaction.amount == 0:
            return 0.0, 0.0

        return abs(self.cost / transaction.amount), self.cost
