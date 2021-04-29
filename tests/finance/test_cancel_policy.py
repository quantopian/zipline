#
# Copyright 2016 Quantopian, Inc.
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

import pytest
from zipline.finance.cancel_policy import NeverCancel, EODCancel
from zipline.gens.sim_engine import BAR, SESSION_END


TEST_INPUT = [SESSION_END, BAR]


def test_eod_cancel():
    cancel_policy = EODCancel()
    assert cancel_policy.should_cancel(SESSION_END)
    assert not cancel_policy.should_cancel(BAR)


@pytest.mark.parametrize("test_input", TEST_INPUT)
def test_never_cancel(test_input):
    cancel_policy = NeverCancel()
    assert not cancel_policy.should_cancel(test_input)
