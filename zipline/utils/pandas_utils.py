"""
Utilities for working with pandas objects.
"""
import pandas as pd
from distutils.version import StrictVersion

pandas_version = StrictVersion(pd.__version__)


def explode(df):
    """
    Take a DataFrame and return a triple of

    (df.index, df.columns, df.values)
    """
    return df.index, df.columns, df.values


try:
    # This branch is hit in pandas 17
    sort_values = pd.DataFrame.sort_values
except AttributeError:
    # This branch is hit in pandas 16
    sort_values = pd.DataFrame.sort

if pandas_version >= StrictVersion('0.17.1'):
    july_5th_holiday_observance = lambda dtix: dtix[dtix.year != 2013]
else:
    july_5th_holiday_observance = lambda dt: None if dt.year == 2013 else dt