#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import copy

from zipline.assets import Asset
from zipline.protocol import DATASOURCE_TYPE
from zipline.utils.input_validation import expect_types


class Transaction:
    @expect_types(asset=Asset)
    def __init__(self, asset, amount, dt, price, order_id):
        self.asset = asset
        self.amount = amount
        self.dt = dt
        self.price = price
        self.order_id = order_id
        self.type = DATASOURCE_TYPE.TRANSACTION

    def __getitem__(self, name):
        return self.__dict__[name]

    def __repr__(self):
        template = "{cls}(asset={asset}, dt={dt}," " amount={amount}, price={price})"

        return template.format(
            cls=type(self).__name__,
            asset=self.asset,
            dt=self.dt,
            amount=self.amount,
            price=self.price,
        )

    def to_dict(self):
        py = copy(self.__dict__)
        del py["type"]
        del py["asset"]

        # Adding 'sid' for backwards compatibility with downstrean consumers.
        py["sid"] = self.asset

        # If you think this looks dumb, that is because it is! We once stored
        # commission here, but haven't for over a year. I don't want to change
        # the perf packet structure yet.
        py["commission"] = None

        return py


def create_transaction(order, dt, price, amount):

    # floor the amount to protect against non-whole number orders
    # TODO: Investigate whether we can add a robust check in blotter
    # and/or tradesimulation, as well.
    amount_magnitude = int(abs(amount))

    if amount_magnitude < 1:
        raise Exception("Transaction magnitude must be at least 1.")

    transaction = Transaction(
        asset=order.asset, amount=int(amount), dt=dt, price=price, order_id=order.id
    )

    return transaction
