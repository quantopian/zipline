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
from pandas.core.common import mask_missing
try:
    from pandas.core.common import backfill_2d, pad_2d
except ImportError:
    # In 0.17, pad_2d and backfill_2d werw moved from pandas.core.common to
    # pandas.core.missing
    from pandas.core.missing import backfill_2d, pad_2d


def _interpolate(values, method, axis=None):
    if values.ndim == 1:
        axis = 0
    elif values.ndim == 2:
        axis = 1
    else:
        raise Exception("Cannot interpolate array with more than 2 dims")

    values = values.copy()
    values = interpolate_2d(values, method, axis=axis)
    return values


def interpolate_2d(values, method='pad', axis=0, limit=None, fill_value=None):
    """
    Copied from the 0.15.2. This did not exist in 0.12.0.

    Differences:
        - Don't depend on pad_2d and backfill_2d to return values
        - Removed dtype kwarg. 0.12.0 did not have this option.
    """
    transf = (lambda x: x) if axis == 0 else (lambda x: x.T)

    # reshape a 1 dim if needed
    ndim = values.ndim
    if values.ndim == 1:
        if axis != 0:  # pragma: no cover
            raise AssertionError("cannot interpolate on a ndim == 1 with "
                                 "axis != 0")
        values = values.reshape(tuple((1,) + values.shape))

    if fill_value is None:
        mask = None
    else:  # todo create faster fill func without masking
        mask = mask_missing(transf(values), fill_value)

    # Note: pad_2d and backfill_2d work inplace in 0.12.0 and 0.15.2
    # in 0.15.2 they also return a reference to values
    if method == 'pad':
        pad_2d(transf(values), limit=limit, mask=mask)
    else:
        backfill_2d(transf(values), limit=limit, mask=mask)

    # reshape back
    if ndim == 1:
        values = values[0]

    return values


def ffill(values, axis=None):
    return _interpolate(values, 'pad', axis=axis)


def bfill(values, axis=None):
    return _interpolate(values, 'bfill', axis=axis)
