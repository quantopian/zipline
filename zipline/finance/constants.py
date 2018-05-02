#
# Copyright 2012 Quantopian, Inc.
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

TRADING_DAYS_IN_YEAR = 250
TRADING_HOURS_IN_DAY = 6.5
MINUTES_IN_HOUR = 60

ANNUALIZER = {'daily': TRADING_DAYS_IN_YEAR,
              'hourly': TRADING_DAYS_IN_YEAR * TRADING_HOURS_IN_DAY,
              'minute': TRADING_DAYS_IN_YEAR * TRADING_HOURS_IN_DAY *
              MINUTES_IN_HOUR}

# NOTE: It may be worth revisiting how the keys for this dictionary are
# specified, for instance making them ContinuousFuture objects instead of
# static strings.
FUTURE_EXCHANGE_FEES_BY_SYMBOL = {
    'AD': 1.60,  # AUD
    'AI': 0.96,  # Bloomberg Commodity Index
    'BD': 1.50,  # Big Dow
    'BO': 1.95,  # Soybean Oil
    'BP': 1.60,  # GBP
    'CD': 1.60,  # CAD
    'CL': 1.50,  # Crude Oil
    'CM': 1.03,  # Corn e-mini
    'CN': 1.95,  # Corn
    'DJ': 1.50,  # Dow Jones
    'EC': 1.60,  # Euro FX
    'ED': 1.25,  # Eurodollar
    'EE': 1.50,  # Euro FX e-mini
    'EI': 1.50,  # MSCI Emerging Markets mini
    'EL': 1.50,  # Eurodollar NYSE LIFFE
    'ER': 0.65,  # Russell2000 e-mini
    'ES': 1.18,  # SP500 e-mini
    'ET': 1.50,  # Ethanol
    'EU': 1.50,  # Eurodollar e-micro
    'FC': 2.03,  # Feeder Cattle
    'FF': 0.96,  # 3-Day Federal Funds
    'FI': 0.56,  # Deliverable Interest Rate Swap 5y
    'FS': 1.50,  # Interest Rate Swap 5y
    'FV': 0.65,  # US 5y
    'GC': 1.50,  # Gold
    'HG': 1.50,  # Copper
    'HO': 1.50,  # Heating Oil
    'HU': 1.50,  # Unleaded Gasoline
    'JE': 0.16,  # JPY e-mini
    'JY': 1.60,  # JPY
    'LB': 2.03,  # Lumber
    'LC': 2.03,  # Live Cattle
    'LH': 2.03,  # Lean Hogs
    'MB': 1.50,  # Municipal Bonds
    'MD': 1.50,  # SP400 Midcap
    'ME': 1.60,  # MXN
    'MG': 1.50,  # MSCI EAFE mini
    'MI': 1.18,  # SP400 Midcap e-mini
    'MS': 1.03,  # Soybean e-mini
    'MW': 1.03,  # Wheat e-mini
    'ND': 1.50,  # Nasdaq100
    'NG': 1.50,  # Natural Gas
    'NK': 2.15,  # Nikkei225
    'NQ': 1.18,  # Nasdaq100 e-mini
    'NZ': 1.60,  # NZD
    'OA': 1.95,  # Oats
    'PA': 1.50,  # Palladium
    'PB': 1.50,  # Pork Bellies
    'PL': 1.50,  # Platinum
    'QG': 0.50,  # Natural Gas e-mini
    'QM': 1.20,  # Crude Oil e-mini
    'RM': 1.50,  # Russell1000 e-mini
    'RR': 1.95,  # Rough Rice
    'SB': 2.10,  # Sugar
    'SF': 1.60,  # CHF
    'SM': 1.95,  # Soybean Meal
    'SP': 2.40,  # SP500
    'SV': 1.50,  # Silver
    'SY': 1.95,  # Soybean
    'TB': 1.50,  # Treasury Bills
    'TN': 0.56,  # Deliverable Interest Rate Swap 10y
    'TS': 1.50,  # Interest Rate Swap 10y
    'TU': 1.50,  # US 2y
    'TY': 0.75,  # US 10y
    'UB': 0.85,  # Ultra Tbond
    'US': 0.80,  # US 30y
    'VX': 1.50,  # VIX
    'WC': 1.95,  # Wheat
    'XB': 1.50,  # RBOB Gasoline
    'XG': 0.75,  # Gold e-mini
    'YM': 1.50,  # Dow Jones e-mini
    'YS': 0.75,  # Silver e-mini
}

# See `zipline.finance.slippage.VolatilityVolumeShare` for more information on
# how these constants are used.
DEFAULT_ETA = 0.049018143225019836
ROOT_SYMBOL_TO_ETA = {
    'AD': DEFAULT_ETA,           # AUD
    'AI': DEFAULT_ETA,           # Bloomberg Commodity Index
    'BD': 0.050346811117733474,  # Big Dow
    'BO': 0.054930995070046298,  # Soybean Oil
    'BP': 0.047841544238716338,  # GBP
    'CD': 0.051124420640250717,  # CAD
    'CL': 0.04852544628414196,   # Crude Oil
    'CM': 0.052683478163348625,  # Corn e-mini
    'CN': 0.053499718390037809,  # Corn
    'DJ': 0.02313009072076987,   # Dow Jones
    'EC': 0.04885131067661861,   # Euro FX
    'ED': 0.094184297090245755,  # Eurodollar
    'EE': 0.048713151357687556,  # Euro FX e-mini
    'EI': 0.031712708439692663,  # MSCI Emerging Markets mini
    'EL': 0.044207422018209361,  # Eurodollar NYSE LIFFE
    'ER': 0.045930567737711307,  # Russell2000 e-mini
    'ES': 0.047304418321993502,  # SP500 e-mini
    'ET': DEFAULT_ETA,           # Ethanol
    'EU': 0.049750396084029064,  # Eurodollar e-micro
    'FC': 0.058728734202178494,  # Feeder Cattle
    'FF': 0.048970591527624042,  # 3-Day Federal Funds
    'FI': 0.033477176738170772,  # Deliverable Interest Rate Swap 5y
    'FS': 0.034557788010453824,  # Interest Rate Swap 5y
    'FV': 0.046544427716056963,  # US 5y
    'GC': 0.048933313546125207,  # Gold
    'HG': 0.052238417524987799,  # Copper
    'HO': 0.045061318412156062,  # Heating Oil
    'HU': 0.017154313062463938,  # Unleaded Gasoline
    'JE': 0.013948949613401812,  # JPY e-mini
    'JY': DEFAULT_ETA,           # JPY
    'LB': 0.06146586386903994,   # Lumber
    'LC': 0.055853801862858619,  # Live Cattle
    'LH': 0.057557004630219781,  # Lean Hogs
    'MB': DEFAULT_ETA,           # Municipal Bonds
    'MD': DEFAULT_ETA,           # SP400 Midcap
    'ME': 0.030383767727818548,  # MXN
    'MG': 0.029579261656151684,  # MSCI EAFE mini
    'MI': 0.041026288873007355,  # SP400 Midcap e-mini
    'MS': DEFAULT_ETA,           # Soybean e-mini
    'MW': 0.052579919663880245,  # Wheat e-mini
    'ND': DEFAULT_ETA,           # Nasdaq100
    'NG': 0.047897809233755716,  # Natural Gas
    'NK': 0.044555435054791433,  # Nikkei225
    'NQ': 0.044772425085977945,  # Nasdaq100 e-mini
    'NZ': 0.049170418073872041,  # NZD
    'OA': 0.056973267232775522,  # Oats
    'PA': DEFAULT_ETA,           # Palladium
    'PB': DEFAULT_ETA,           # Pork Bellies
    'PL': 0.054579379665647493,  # Platinum
    'QG': DEFAULT_ETA,           # Natural Gas e-mini
    'QM': DEFAULT_ETA,           # Crude Oil e-mini
    'RM': 0.037425041244579654,  # Russell1000 e-mini
    'RR': DEFAULT_ETA,           # Rough Rice
    'SB': 0.057388160345668134,  # Sugar
    'SF': 0.047784825569615726,  # CHF
    'SM': 0.048552860559844223,  # Soybean Meal
    'SP': DEFAULT_ETA,           # SP500
    'SV': 0.052691435039931109,  # Silver
    'SY': 0.052041703657281613,  # Soybean
    'TB': DEFAULT_ETA,           # Treasury Bills
    'TN': 0.033363465365262503,  # Deliverable Interest Rate Swap 10y
    'TS': 0.032908878455069152,  # Interest Rate Swap 10y
    'TU': 0.063867646063840794,  # US 2y
    'TY': 0.050586988554700826,  # US 10y
    'UB': DEFAULT_ETA,           # Ultra Tbond
    'US': 0.047984179873590722,  # US 30y
    'VX': DEFAULT_ETA,           # VIX
    'WC': 0.052636542119329242,  # Wheat
    'XB': 0.044444916388854484,  # RBOB Gasoline
    'XG': DEFAULT_ETA,           # Gold e-mini
    'YM': DEFAULT_ETA,           # Dow Jones e-mini
    'YS': DEFAULT_ETA,           # Silver e-mini
}
