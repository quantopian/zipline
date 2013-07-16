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

import os

import numpy as np
import xlrd


def col_letter_to_index(col_letter):
    # Only supports single letter,
    # but answer key doesn't need multi-letter, yet.
    return ord(col_letter) - 65

DIR = os.path.dirname(os.path.realpath(__file__))

ANSWER_KEY_PATH = os.path.join(DIR, 'risk-answer-key.xls')


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
        return col_letter_to_index(self.col)

    @property
    def row_start_index(self):
        return self.row_start - 1

    @property
    def row_end_index(self):
        return self.row_end - 1


class AnswerKey(object):

    RETURNS = DataIndex('Sim', 'D', 4, 255)

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

    def __init__(self):
        self.workbook = xlrd.open_workbook(ANSWER_KEY_PATH)

        self.sheets = {}
        self.sheets['Sim'] = self.workbook.sheet_by_name('Sim')
        self.sheets['s_p'] = self.workbook.sheet_by_name('s_p')

    def get_values(self, data_index, decimal=4):
        return [np.round(x, decimal) for x in
                self.sheets[data_index.sheet_name].col_values(
                    data_index.col_index,
                    data_index.row_start_index,
                    data_index.row_end_index + 1)]
