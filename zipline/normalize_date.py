import datetime

def normalize_date(dt):
    '''
    Normalize datetime.datetime value to midnight. Returns datetime.date as a
    datetime.datetime at midnight
    Returns
    -------
    normalized : datetime.datetime or Timestamp
    '''
    #if PyDateTime_Check(dt):
    if isinstance(dt, datetime.datetime):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    #elif PyDate_Check(dt):
    elif isinstance(dt, datetime.date):
        return datetime(dt.year, dt.month, dt.day)
    else:
        raise TypeError('Unrecognized type: %s' % type(dt))

