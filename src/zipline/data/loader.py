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
import os

import pandas as pd


def load_prices_from_csv(filepath, identifier_col, tz="UTC"):
    data = pd.read_csv(filepath, index_col=identifier_col)
    data.index = pd.DatetimeIndex(data.index, tz=tz)
    data.sort_index(inplace=True)
    return data


def load_prices_from_csv_folder(folderpath, identifier_col, tz="UTC"):
    data = None
    for file in os.listdir(folderpath):
        if ".csv" not in file:
            continue
        raw = load_prices_from_csv(os.path.join(folderpath, file), identifier_col, tz)
        if data is None:
            data = raw
        else:
            data = pd.concat([data, raw], axis=1)
    return data
