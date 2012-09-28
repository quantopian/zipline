class PerShareCommission(object):

    def __init__(self, cost=0.03):
        self.cost = cost


    def calculate(self, transaction):
        return self.cost, abs(transaction.amount * self.cost)

class PerTradeCommission(object):

    def __init__(self, cost=5.0):
        self.cost = cost

    def calculate(self, transaction):
        if transaction.amount == 0:
            return 0.0, 0.0

        return abs(self.cost / transaction.amount), self.cost
