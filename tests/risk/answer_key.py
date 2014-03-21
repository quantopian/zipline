#
# Copyright 2014 Quantopian, Inc.
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
import datetime
import hashlib
import os

import numpy as np
import pandas as pd
import pytz
import xlrd
import requests

from six.moves import map


def col_letter_to_index(col_letter):
    # Only supports single letter,
    # but answer key doesn't need multi-letter, yet.
    index = 0
    for i, char in enumerate(reversed(col_letter)):
        index += ((ord(char) - 65) + 1) * pow(26, i)
    return index

DIR = os.path.dirname(os.path.realpath(__file__))

ANSWER_KEY_CHECKSUMS_PATH = os.path.join(DIR, 'risk-answer-key-checksums')
ANSWER_KEY_CHECKSUMS = open(ANSWER_KEY_CHECKSUMS_PATH, 'r').read().splitlines()

ANSWER_KEY_FILENAME = 'risk-answer-key.xlsx'

ANSWER_KEY_PATH = os.path.join(DIR, ANSWER_KEY_FILENAME)

ANSWER_KEY_BUCKET_NAME = 'zipline-test_data'

ANSWER_KEY_DL_TEMPLATE = """
https://s3.amazonaws.com/zipline-test-data/risk/{md5}/risk-answer-key.xlsx
""".strip()

LATEST_ANSWER_KEY_URL = ANSWER_KEY_DL_TEMPLATE.format(
    md5=ANSWER_KEY_CHECKSUMS[-1])


def answer_key_signature():
    with open(ANSWER_KEY_PATH, 'rb') as f:
        md5 = hashlib.md5()
        buf = f.read(1024)
        md5.update(buf)
        while buf != b"":
            buf = f.read(1024)
            md5.update(buf)
    return md5.hexdigest()


def ensure_latest_answer_key():
    """
    Get the latest answer key from a publically available location.

    Logic for determining what and when to download is as such:

    - If there is no local spreadsheet file, then get the lastest answer key,
    as defined by the last row in the checksum file.
    - If there is a local spreadsheet file:
    -- If the spreadsheet's checksum is in the checksum file:
    --- If the spreadsheet's checksum does not match the latest, then grab the
    the latest checksum and replace the local checksum file.
    --- If the spreadsheet's checksum matches the latest, then skip download,
    and use the local spreadsheet as a cached copy.
    -- If the spreadsheet's checksum is not in the checksum file, then leave
    the local file alone, assuming that the local xls's md5 is not in the list
    due to local modifications during development.

    It is possible that md5's could collide, if that is ever case, we should
    then find an alternative naming scheme.

    The spreadsheet answer sheet is not kept in SCM, as every edit would
    increase the repo size by the file size, since it is treated as a binary.
    """

    answer_key_dl_checksum = None

    local_answer_key_exists = os.path.exists(ANSWER_KEY_PATH)
    if local_answer_key_exists:
        local_hash = answer_key_signature()

        if local_hash in ANSWER_KEY_CHECKSUMS:
            # Assume previously downloaded version.
            # Check for latest.
            if local_hash != ANSWER_KEY_CHECKSUMS[-1]:
                # More recent checksum, download
                answer_key_dl_checksum = ANSWER_KEY_CHECKSUMS[-1]
            else:
                # Assume local copy that is being developed on
                answer_key_dl_checksum = None
    else:
        answer_key_dl_checksum = ANSWER_KEY_CHECKSUMS[-1]

    if answer_key_dl_checksum:
        res = requests.get(
            ANSWER_KEY_DL_TEMPLATE.format(md5=answer_key_dl_checksum))
        with open(ANSWER_KEY_PATH, 'wb') as f:
            f.write(res.content)

# Get latest answer key on load.
ensure_latest_answer_key()


class DataIndex(object):
    """
    Coordinates for the spreadsheet, using the values as seen in the notebook.
    The python-excel libraries use 0 index, while the spreadsheet in a GUI
    uses a 1 index.
    """
    def __init__(self, sheet_name, col, row_start, row_end,
                 value_type='float'):
        self.sheet_name = sheet_name
        self.col = col
        self.row_start = row_start
        self.row_end = row_end
        self.value_type = value_type

    @property
    def col_index(self):
        return col_letter_to_index(self.col) - 1

    @property
    def row_start_index(self):
        return self.row_start - 1

    @property
    def row_end_index(self):
        return self.row_end - 1

    def __str__(self):
        return "'{sheet_name}'!{col}{row_start}:{col}{row_end}".format(
            sheet_name=self.sheet_name,
            col=self.col,
            row_start=self.row_start,
            row_end=self.row_end
        )


class AnswerKey(object):

    INDEXES = {
        'RETURNS': DataIndex('Sim Period', 'D', 4, 255),

        'BENCHMARK': {
            'Dates': DataIndex('s_p', 'A', 4, 254, value_type='date'),
            'Returns': DataIndex('s_p', 'H', 4, 254)
        },

        # Below matches the inconsistent capitalization in spreadsheet
        'BENCHMARK_PERIOD_RETURNS': {
            'Monthly': DataIndex('s_p', 'R', 8, 19),
            '3-Month': DataIndex('s_p', 'S', 10, 19),
            '6-month': DataIndex('s_p', 'T', 13, 19),
            'year': DataIndex('s_p', 'U', 19, 19),
        },

        'BENCHMARK_PERIOD_VOLATILITY': {
            'Monthly': DataIndex('s_p', 'V', 8, 19),
            '3-Month': DataIndex('s_p', 'W', 10, 19),
            '6-month': DataIndex('s_p', 'X', 13, 19),
            'year': DataIndex('s_p', 'Y', 19, 19),
        },

        'ALGORITHM_PERIOD_RETURNS': {
            'Monthly': DataIndex('Sim Period', 'Z', 23, 34),
            '3-Month': DataIndex('Sim Period', 'AA', 25, 34),
            '6-month': DataIndex('Sim Period', 'AB', 28, 34),
            'year': DataIndex('Sim Period', 'AC', 34, 34),
        },

        'ALGORITHM_PERIOD_VOLATILITY': {
            'Monthly': DataIndex('Sim Period', 'AH', 23, 34),
            '3-Month': DataIndex('Sim Period', 'AI', 25, 34),
            '6-month': DataIndex('Sim Period', 'AJ', 28, 34),
            'year': DataIndex('Sim Period', 'AK', 34, 34),
        },

        'ALGORITHM_PERIOD_SHARPE': {
            'Monthly': DataIndex('Sim Period', 'AL', 23, 34),
            '3-Month': DataIndex('Sim Period', 'AM', 25, 34),
            '6-month': DataIndex('Sim Period', 'AN', 28, 34),
            'year': DataIndex('Sim Period', 'AO', 34, 34),
        },

        'ALGORITHM_PERIOD_BETA': {
            'Monthly': DataIndex('Sim Period', 'AP', 23, 34),
            '3-Month': DataIndex('Sim Period', 'AQ', 25, 34),
            '6-month': DataIndex('Sim Period', 'AR', 28, 34),
            'year': DataIndex('Sim Period', 'AS', 34, 34),
        },

        'ALGORITHM_PERIOD_ALPHA': {
            'Monthly': DataIndex('Sim Period', 'AT', 23, 34),
            '3-Month': DataIndex('Sim Period', 'AU', 25, 34),
            '6-month': DataIndex('Sim Period', 'AV', 28, 34),
            'year': DataIndex('Sim Period', 'AW', 34, 34),
        },

        'ALGORITHM_PERIOD_BENCHMARK_VARIANCE': {
            'Monthly': DataIndex('Sim Period', 'BJ', 23, 34),
            '3-Month': DataIndex('Sim Period', 'BK', 25, 34),
            '6-month': DataIndex('Sim Period', 'BL', 28, 34),
            'year': DataIndex('Sim Period', 'BM', 34, 34),
        },

        'ALGORITHM_PERIOD_COVARIANCE': {
            'Monthly': DataIndex('Sim Period', 'BF', 23, 34),
            '3-Month': DataIndex('Sim Period', 'BG', 25, 34),
            '6-month': DataIndex('Sim Period', 'BH', 28, 34),
            'year': DataIndex('Sim Period', 'BI', 34, 34),
        },

        'ALGORITHM_PERIOD_DOWNSIDE_RISK': {
            'Monthly': DataIndex('Sim Period', 'BN', 23, 34),
            '3-Month': DataIndex('Sim Period', 'BO', 25, 34),
            '6-month': DataIndex('Sim Period', 'BP', 28, 34),
            'year': DataIndex('Sim Period', 'BQ', 34, 34),
        },

        'ALGORITHM_PERIOD_SORTINO': {
            'Monthly': DataIndex('Sim Period', 'BR', 23, 34),
            '3-Month': DataIndex('Sim Period', 'BS', 25, 34),
            '6-month': DataIndex('Sim Period', 'BT', 28, 34),
            'year': DataIndex('Sim Period', 'BU', 34, 34),
        },

        'ALGORITHM_RETURN_VALUES': DataIndex(
            'Sim Cumulative', 'D', 4, 254),

        'ALGORITHM_CUMULATIVE_VOLATILITY': DataIndex(
            'Sim Cumulative', 'P', 4, 254),

        'ALGORITHM_CUMULATIVE_SHARPE': DataIndex(
            'Sim Cumulative', 'R', 4, 254),

        'CUMULATIVE_DOWNSIDE_RISK': DataIndex(
            'Sim Cumulative', 'U', 4, 254),

        'CUMULATIVE_SORTINO': DataIndex(
            'Sim Cumulative', 'V', 4, 254),

        'CUMULATIVE_INFORMATION': DataIndex(
            'Sim Cumulative', 'AA', 4, 254),

        'CUMULATIVE_BETA': DataIndex(
            'Sim Cumulative', 'AD', 4, 254),

        'CUMULATIVE_ALPHA': DataIndex(
            'Sim Cumulative', 'AE', 4, 254),

        'CUMULATIVE_MAX_DRAWDOWN': DataIndex(
            'Sim Cumulative', 'AH', 4, 254),

    }

    def __init__(self):
        self.workbook = xlrd.open_workbook(ANSWER_KEY_PATH)

        self.sheets = {}
        self.sheets['Sim Period'] = self.workbook.sheet_by_name('Sim Period')
        self.sheets['Sim Cumulative'] = self.workbook.sheet_by_name(
            'Sim Cumulative')
        self.sheets['s_p'] = self.workbook.sheet_by_name('s_p')

        for name, index in self.INDEXES.items():
            if isinstance(index, dict):
                subvalues = {}
                for subkey, subindex in index.items():
                    subvalues[subkey] = self.get_values(subindex)
                setattr(self, name, subvalues)
            else:
                setattr(self, name, self.get_values(index))

    def parse_date_value(self, value):
        return xlrd.xldate_as_tuple(value, 0)

    def parse_float_value(self, value):
        return value if value != '' else np.nan

    def get_raw_values(self, data_index):
        return self.sheets[data_index.sheet_name].col_values(
            data_index.col_index,
            data_index.row_start_index,
            data_index.row_end_index + 1)

    @property
    def value_type_to_value_func(self):
        return {
            'float': self.parse_float_value,
            'date': self.parse_date_value,
        }

    def get_values(self, data_index):
        value_parser = self.value_type_to_value_func[data_index.value_type]
        return [value for value in
                map(value_parser, self.get_raw_values(data_index))]


ANSWER_KEY = AnswerKey()

BENCHMARK_DATES = ANSWER_KEY.BENCHMARK['Dates']
BENCHMARK_RETURNS = ANSWER_KEY.BENCHMARK['Returns']
DATES = [datetime.datetime(*x, tzinfo=pytz.UTC) for x in BENCHMARK_DATES]
BENCHMARK = pd.Series(dict(zip(DATES, BENCHMARK_RETURNS)))
ALGORITHM_RETURNS = pd.Series(
    dict(zip(DATES, ANSWER_KEY.ALGORITHM_RETURN_VALUES)))
RETURNS_DATA = pd.DataFrame({'Benchmark Returns': BENCHMARK,
                             'Algorithm Returns': ALGORITHM_RETURNS})
RISK_CUMULATIVE = pd.DataFrame({
    'volatility': pd.Series(dict(zip(
        DATES, ANSWER_KEY.ALGORITHM_CUMULATIVE_VOLATILITY))),
    'sharpe': pd.Series(dict(zip(
        DATES, ANSWER_KEY.ALGORITHM_CUMULATIVE_SHARPE))),
    'downside_risk': pd.Series(dict(zip(
        DATES, ANSWER_KEY.CUMULATIVE_DOWNSIDE_RISK))),
    'sortino': pd.Series(dict(zip(
        DATES, ANSWER_KEY.CUMULATIVE_SORTINO))),
    'information': pd.Series(dict(zip(
        DATES, ANSWER_KEY.CUMULATIVE_INFORMATION))),
    'alpha': pd.Series(dict(zip(
        DATES, ANSWER_KEY.CUMULATIVE_ALPHA))),
    'beta': pd.Series(dict(zip(
        DATES, ANSWER_KEY.CUMULATIVE_BETA))),
    'max_drawdown': pd.Series(dict(zip(
        DATES, ANSWER_KEY.CUMULATIVE_MAX_DRAWDOWN))),
})
