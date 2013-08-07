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

import hashlib
import os

import numpy as np
import xlrd
import requests


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


def answer_key_signature():
    with open(ANSWER_KEY_PATH, 'r') as f:
        md5 = hashlib.md5()
        while True:
            buf = f.read(1024)
            if not buf:
                break
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
        with open(ANSWER_KEY_PATH, 'w') as f:
            f.write(res.content)

# Get latest answer key on load.
ensure_latest_answer_key()


class DataIndex(object):
    """
    Coordinates for the spreadsheet, using the values as seen in the notebook.
    The python-excel libraries use 0 index, while the spreadsheet in a GUI
    uses a 1 index.
    """
    def __init__(self, sheet_name, col, row_start, row_end):
        self.sheet_name = sheet_name
        self.col = col
        self.row_start = row_start
        self.row_end = row_end

    @property
    def col_index(self):
        return col_letter_to_index(self.col) - 1

    @property
    def row_start_index(self):
        return self.row_start - 1

    @property
    def row_end_index(self):
        return self.row_end - 1


class AnswerKey(object):

    RETURNS = DataIndex('Sim Period', 'D', 4, 255)

    # Below matches the inconsistent capitalization in spreadsheet
    BENCHMARK_PERIOD_RETURNS = {
        'Monthly': DataIndex('s_p', 'P', 8, 19),
        '3-Month': DataIndex('s_p', 'Q', 10, 19),
        '6-month': DataIndex('s_p', 'R', 13, 19),
        'year': DataIndex('s_p', 'S', 19, 19),
    }

    BENCHMARK_PERIOD_VOLATILITY = {
        'Monthly': DataIndex('s_p', 'T', 8, 19),
        '3-Month': DataIndex('s_p', 'U', 10, 19),
        '6-month': DataIndex('s_p', 'V', 13, 19),
        'year': DataIndex('s_p', 'W', 19, 19),
    }

    ALGORITHM_PERIOD_RETURNS = {
        'Monthly': DataIndex('Sim Period', 'V', 23, 34),
        '3-Month': DataIndex('Sim Period', 'W', 25, 34),
        '6-month': DataIndex('Sim Period', 'X', 28, 34),
        'year': DataIndex('Sim Period', 'Y', 34, 34),
    }

    ALGORITHM_PERIOD_VOLATILITY = {
        'Monthly': DataIndex('Sim Period', 'Z', 23, 34),
        '3-Month': DataIndex('Sim Period', 'AA', 25, 34),
        '6-month': DataIndex('Sim Period', 'AB', 28, 34),
        'year': DataIndex('Sim Period', 'AC', 34, 34),
    }

    ALGORITHM_PERIOD_SHARPE = {
        'Monthly': DataIndex('Sim Period', 'AD', 23, 34),
        '3-Month': DataIndex('Sim Period', 'AE', 25, 34),
        '6-month': DataIndex('Sim Period', 'AF', 28, 34),
        'year': DataIndex('Sim Period', 'AG', 34, 34),
    }

    ALGORITHM_PERIOD_BETA = {
        'Monthly': DataIndex('Sim Period', 'AH', 23, 34),
        '3-Month': DataIndex('Sim Period', 'AI', 25, 34),
        '6-month': DataIndex('Sim Period', 'AJ', 28, 34),
        'year': DataIndex('Sim Period', 'AK', 34, 34),
    }

    ALGORITHM_PERIOD_ALPHA = {
        'Monthly': DataIndex('Sim Period', 'AL', 23, 34),
        '3-Month': DataIndex('Sim Period', 'AM', 25, 34),
        '6-month': DataIndex('Sim Period', 'AN', 28, 34),
        'year': DataIndex('Sim Period', 'AO', 34, 34),
    }

    ALGORITHM_PERIOD_BENCHMARK_VARIANCE = {
        'Monthly': DataIndex('Sim Period', 'BB', 23, 34),
        '3-Month': DataIndex('Sim Period', 'BC', 25, 34),
        '6-month': DataIndex('Sim Period', 'BD', 28, 34),
        'year': DataIndex('Sim Period', 'BE', 34, 34),
    }

    ALGORITHM_PERIOD_COVARIANCE = {
        'Monthly': DataIndex('Sim Period', 'AX', 23, 34),
        '3-Month': DataIndex('Sim Period', 'AY', 25, 34),
        '6-month': DataIndex('Sim Period', 'AZ', 28, 34),
        'year': DataIndex('Sim Period', 'BA', 34, 34),
    }

    def __init__(self):
        self.workbook = xlrd.open_workbook(ANSWER_KEY_PATH)

        self.sheets = {}
        self.sheets['Sim Period'] = self.workbook.sheet_by_name('Sim Period')
        self.sheets['s_p'] = self.workbook.sheet_by_name('s_p')

    def get_values(self, data_index, decimal=4):
        return [np.round(x, decimal) for x in
                self.sheets[data_index.sheet_name].col_values(
                    data_index.col_index,
                    data_index.row_start_index,
                    data_index.row_end_index + 1)]
