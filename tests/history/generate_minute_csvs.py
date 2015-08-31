import random
import numpy as np
import pandas as pd

from zipline.finance.trading import TradingEnvironment


def generate_test_data(first_day,
                       last_day,
                       starting_open,
                       starting_volume,
                       multipliers_list,
                       path):
    """
    Utility method to generate fake minute-level CSV data.
    :param first_day: first trading day
    :param last_day: last trading day
    :param starting_open: first open value, raw value.
    :param starting_volume: first volume value, raw value.
    :param multipliers_list: ordered list of pd.Timestamp -> float, one per day
            in the range
    :param path: path to save the CSV
    :return: None
    """

    minutes = TradingEnvironment.instance().\
        minutes_for_days_in_range(first_day, last_day)

    minutes_count = len(minutes)

    o = np.zeros(minutes_count, dtype=np.uint32)
    h = np.zeros(minutes_count, dtype=np.uint32)
    l = np.zeros(minutes_count, dtype=np.uint32)
    c = np.zeros(minutes_count, dtype=np.uint32)
    v = np.zeros(minutes_count, dtype=np.uint32)
    dt = np.zeros(minutes_count, dtype=np.uint32)

    last_open = starting_open * 1000
    last_volume = starting_volume

    for idx in range(minutes_count):
        new_open = last_open + round((random.random() * 5), 2)

        o[idx] = new_open
        h[idx] = new_open + round((random.random() * 10000), 2)
        l[idx] = new_open - round((random.random() * 10000),  2)
        c[idx] = (h[idx] + l[idx]) / 2
        v[idx] = int(last_volume + (random.randrange(-10, 10) * 1e4))
        dt[idx] = minutes[idx].value / 1e9

        last_open = o[idx]
        last_volume = v[idx]

    # now deal with multipliers
    if len(multipliers_list) > 0:
        for idx, multiplier_info in enumerate(multipliers_list):
            start_idx = idx * 390
            end_idx = start_idx + 390

            # dividing by the multipler because we're going backwards
            # and generating the original data that will then be adjusted.
            o[start_idx:end_idx] /= multiplier_info[1]
            h[start_idx:end_idx] /= multiplier_info[1]
            l[start_idx:end_idx] /= multiplier_info[1]
            c[start_idx:end_idx] /= multiplier_info[1]
            v[start_idx:end_idx] *= multiplier_info[1]

    df = pd.DataFrame({
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "dt": dt
    }, columns=[
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dt"
    ], index=minutes)

    df.to_csv(path, index_label="minute")
    #return df
