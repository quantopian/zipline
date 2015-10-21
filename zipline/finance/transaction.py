from copy import copy
from functools import partial
import math
from zipline.protocol import DATASOURCE_TYPE
from zipline.utils.serialization_utils import VERSION_LABEL


def transact_stub(slippage, commission, event, open_orders):
    """
    This is intended to be wrapped in a partial, so that the
    slippage and commission models can be enclosed.
    """
    for order, transaction in slippage(event, open_orders):
        if transaction and transaction.amount != 0:
            direction = math.copysign(1, transaction.amount)
            per_share, total_commission = commission.calculate(transaction)
            transaction.price += per_share * direction
            transaction.commission = total_commission
        yield order, transaction


def transact_partial(slippage, commission):
    return partial(transact_stub, slippage, commission)


class Transaction(object):

    def __init__(self, sid, amount, dt, price, order_id, commission=None):
        self.sid = sid
        self.amount = amount
        self.dt = dt
        self.price = price
        self.order_id = order_id
        self.commission = commission
        self.type = DATASOURCE_TYPE.TRANSACTION

    def __getitem__(self, name):
        return self.__dict__[name]

    def to_dict(self):
        py = copy(self.__dict__)
        del py['type']
        return py

    def __getstate__(self):

        state_dict = copy(self.__dict__)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("Transaction saved state is too old.")

        self.__dict__.update(state)


def create_transaction(sid, dt, order, price, amount):

    # floor the amount to protect against non-whole number orders
    # TODO: Investigate whether we can add a robust check in blotter
    # and/or tradesimulation, as well.
    amount_magnitude = int(abs(amount))

    if amount_magnitude < 1:
        raise Exception("Transaction magnitude must be at least 1.")

    transaction = Transaction(
        sid=sid,
        amount=int(amount),
        dt=dt,
        price=price,
        order_id=order.id
    )

    return transaction
