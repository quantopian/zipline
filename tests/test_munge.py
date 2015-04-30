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
import random

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase
from zipline.utils.munge import bfill, ffill


class MungeTests(TestCase):
    def test_bfill(self):
        # test ndim=1
        N = 100
        s = pd.Series(np.random.randn(N))
        mask = random.sample(range(N), 10)
        s.iloc[mask] = np.nan

        correct = s.bfill().values
        test = bfill(s.values)
        assert_almost_equal(correct, test)

        # test ndim=2
        df = pd.DataFrame(np.random.randn(N, N))
        df.iloc[mask] = np.nan
        correct = df.bfill().values
        test = bfill(df.values)
        assert_almost_equal(correct, test)

    def test_ffill(self):
        # test ndim=1
        N = 100
        s = pd.Series(np.random.randn(N))
        mask = random.sample(range(N), 10)
        s.iloc[mask] = np.nan

        correct = s.ffill().values
        test = ffill(s.values)
        assert_almost_equal(correct, test)

        # test ndim=2
        df = pd.DataFrame(np.random.randn(N, N))
        df.iloc[mask] = np.nan
        correct = df.ffill().values
        test = ffill(df.values)
        assert_almost_equal(correct, test)
