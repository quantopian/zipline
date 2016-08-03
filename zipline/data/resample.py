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

_MINUTE_TO_SESSION_OHCLV_HOW = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}


def minute_to_session(minute_frame, calendar):
    """
    Resample a DataFrame with minute data into the frame expected by a
    BcolzDailyBarWriter.

    Parameters
    ----------
    minute_frame : pd.DataFrame
        A DataFrame with the columns `open`, `high`, `low`, `close`, `volume`,
        and `dt` (minute dts)
    calendar : zipline.utils.calendars.trading_calendar.TradingCalendar
        A TradingCalendar on which session labels to resample from minute
        to session.

    Return
    ------
    session_frame : pd.DataFrame
        A DataFrame with the columns `open`, `high`, `low`, `close`, `volume`,
        and `day` (datetime-like).
    """
    # Group minutes into their respective days. Note that this will
    # create groups for all trading days in the desired range,
    # including days with no minute data.
    return minute_frame.resample(calendar.day,
                                 how=_MINUTE_TO_SESSION_OHCLV_HOW)
