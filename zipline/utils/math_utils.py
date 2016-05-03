#
# Copyright 2013 Quantopian, Inc.
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
import math

from numpy import isnan


def tolerant_equals(a, b, atol=10e-7, rtol=10e-7, equal_nan=False):
    """Check if a and b are equal with some tolerance.

    Parameters
    ----------
    a, b : float
        The floats to check for equality.
    atol : float, optional
        The absolute tolerance.
    rtol : float, optional
        The relative tolerance.
    equal_nan : bool, optional
        Should NaN compare equal?

    See Also
    --------
    numpy.isclose

    Notes
    -----
    This function is just a scalar version of numpy.isclose for performance.
    See the docstring of ``isclose`` for more information about ``atol`` and
    ``rtol``.
    """
    if equal_nan and isnan(a) and isnan(b):
        return True
    return math.fabs(a - b) <= (atol + rtol * math.fabs(b))


try:
    # fast versions
    import bottleneck as bn
    nanmean = bn.nanmean
    nanstd = bn.nanstd
    nansum = bn.nansum
    nanmax = bn.nanmax
    nanmin = bn.nanmin
    nanargmax = bn.nanargmax
    nanargmin = bn.nanargmin
except ImportError:
    # slower numpy
    import numpy as np
    nanmean = np.nanmean
    nanstd = np.nanstd
    nansum = np.nansum
    nanmax = np.nanmax
    nanmin = np.nanmin
    nanargmax = np.nanargmax
    nanargmin = np.nanargmin


def round_if_near_integer(a, epsilon=1e-4):
    """
    Round a to the nearest integer if that integer is within an epsilon
    of a.
    """
    if abs(a - round(a)) <= epsilon:
        return round(a)
    else:
        return a
