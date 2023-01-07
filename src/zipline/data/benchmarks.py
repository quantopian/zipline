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
import logging

import pandas as pd

log = logging.getLogger(__name__)


def get_benchmark_returns_from_file(filelike):
    """Get a Series of benchmark returns from a file

    Parameters
    ----------
    filelike : str or file-like object
        Path to the benchmark file.
        expected csv file format:
        date,return
        2020-01-02 00:00:00+00:00,0.01
        2020-01-03 00:00:00+00:00,-0.02

    """
    log.info("Reading benchmark returns from %s", filelike)

    df = pd.read_csv(
        filelike,
        index_col=["date"],
        parse_dates=["date"],
    )
    if df.index.tz is not None:
        df = df.tz_localize(None)

    if "return" not in df.columns:
        raise ValueError(
            "The column 'return' not found in the "
            "benchmark file \n"
            "Expected benchmark file format :\n"
            "date, return\n"
            "2020-01-02 00:00:00+00:00,0.01\n"
            "2020-01-03 00:00:00+00:00,-0.02\n"
        )

    return df["return"].sort_index()
