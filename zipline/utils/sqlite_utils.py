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

import sqlite3

import sqlalchemy as sa
from six.moves import range

from .input_validation import coerce_string

SQLITE_MAX_VARIABLE_NUMBER = 998


def group_into_chunks(items, chunk_size=SQLITE_MAX_VARIABLE_NUMBER):
    items = list(items)
    return [items[x:x+chunk_size]
            for x in range(0, len(items), chunk_size)]


coerce_string_to_conn = coerce_string(sqlite3.connect)
coerce_string_to_eng = coerce_string(
    lambda s: sa.create_engine('sqlite:///' + s)
)
