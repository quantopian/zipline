#
# Copyright 2017 Quantopian, Inc.
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
import pandas as pd
from zipline.assets import Equity, ExchangeInfo
from zipline.finance.transaction import Transaction


def test_transaction_repr():
    dt = pd.Timestamp("2017-01-01")
    asset = Equity(
        1,
        exchange_info=ExchangeInfo("test", "test full", "US"),
    )
    txn = Transaction(asset, amount=100, dt=dt, price=10, order_id=0)
    expected = (
        "Transaction(asset=Equity(1), dt=2017-01-01 00:00:00," " amount=100, price=10)"
    )
    assert repr(txn) == expected
