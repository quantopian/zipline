"""
Utilities for working with pandas objects.
"""
import pandas as pd


def explode(df):
    """
    Take a DataFrame and return a triple of

    (df.index, df.columns, df.values)
    """
    return df.index, df.columns, df.values


try:
    # pandas 0.16 compat
    sort_values = pd.DataFrame.sort_values
except AttributeError:
    sort_values = pd.DataFrame.sort
