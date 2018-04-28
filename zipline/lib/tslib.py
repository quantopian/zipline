import pandas

if pandas.__version__ >= '0.20.0':
    from pandas._libs.tslib import iNaT, normalize_date, Timedelta, Timestamp
else:
    from pandas.tslib import iNaT, normalize_date, Timedelta, Timestamp
