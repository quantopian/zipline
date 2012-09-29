import pytz
import math

import zipline.protocol as zp


def simulate_method_factory(slippage, commission):

    def simulate(open_orders, events):
        transaction = slippage.simulate(open_orders, events)
        if transaction:
            per_share, total_commission = commission.calculate(transaction)
            transaction.price = transaction.price + per_share
            transaction.commission = total_commission
        return transaction

    return simulate

def create_transaction(sid, amount, price, dt):

    txn = {'sid'            : sid,
                'amount'        : int(amount),
                'dt'            : dt,
                'price'         : price,
          }

    transaction = zp.ndict(txn)
    return transaction


class VolumeShareSlippage(object):

    def __init__(self,
            volume_limit=.25,
            price_impact=0.1):

        self.volume_limit = volume_limit
        self.price_impact = price_impact

    def simulate(self, event, open_orders):

        if(event.volume == 0):
            #there are zero volume events bc some stocks trade
            #less frequently than once per minute.
            return None

        if event.sid in open_orders:
            orders = open_orders[event.sid]
            orders = sorted(orders, key=lambda o: o.dt)
        else:
            return None

        dt = event.dt
        total_order = 0
        simulated_amount = 0
        simulated_impact = 0.0
        direction = 1.0
        for order in orders:

            if(order.dt < event.dt):

                # orders are only good on the day they are issued
                if order.dt.day < event.dt.day:
                    continue

                open_amount = order.amount - order.filled

                if(open_amount != 0):
                    direction = open_amount / math.fabs(open_amount)
                else:
                    direction = 1

                desired_order = total_order + open_amount

                volume_share = direction * (desired_order) / event.volume
                if volume_share > self.volume_limit:
                    volume_share = self.volume_limit
                simulated_amount = int(volume_share * event.volume * direction)
                simulated_impact = (volume_share)**2 * self.price_impact * direction * event.price

                order.filled += (simulated_amount - total_order)
                total_order = simulated_amount

                # we cap the volume share at configured % of a trade
                if volume_share == self.volume_limit:
                    break


        orders = [ x for x in orders if abs(x.amount - x.filled) > 0 and x.dt.day >= event.dt.day]

        open_orders[event.sid] = orders


        if simulated_amount != 0:
            return create_transaction(
                event.sid,
                simulated_amount,
                event.price + simulated_impact,
                dt.replace(tzinfo = pytz.utc),
            )

class FixedSlippage(object):

    def __init__(self, spread=0.0):
        """
        Use the fixed slippage model, which will just add/subtract a specified spread
        spread/2 will be added on buys and subtracted on sells per share
        """
        self.spread = spread

    def simulate(self, event, open_orders):
        if event.sid in open_orders:
            orders = open_orders[event.sid]
            orders = sorted(orders, key=lambda o: o.dt)
        else:
            return None

        amount = 0
        for order in orders:
            amount += order.amount

        if(amount == 0):
            return

        direction = amount / math.fabs(amount)

        txn = create_transaction(
            event.sid,
            amount,
            event.price + (self.spread/2.0 * direction),
            event.dt
        )

        open_orders[event.sid] = []

        return txn
