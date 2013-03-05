import talib
from zipline.transforms import BatchTransform


def _data_to_dict(data, sid):
    data_dict = dict()
    for key in ['open', 'high', 'low', 'close', 'volume']:
        if key in data:
            data_dict[key] = data[key][sid]
    return data_dict


def AD(refresh_period=1):
    """Chaikin A/D Line"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.AD(data_dict)
        return transform[-1]
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=1)
    return transform


def ADOSC(fastperiod=3, slowperiod=10, refresh_period=1):
    """Chaikin A/D Oscillator"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.ADOSC(data_dict, fastperiod, slowperiod)
        return transform[-1]
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=max(fastperiod, slowperiod))
    return transform


def ADX(timeperiod=14, refresh_period=1):
    """Average Directional Movement Index"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.ADX(data_dict, timeperiod)
        return transform[-1]
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=timeperiod)
    return transform


def ADXR(timeperiod=14, refresh_period=1):
    """Average Directional Movement Index"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.ADXR(data_dict, timeperiod)
        return transform[-1]
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=timeperiod)
    return transform


def APO(fastperiod=12, slowperiod=26, matype=0, refresh_period=1):
    """Absolute Price Oscillator"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.APO(data_dict, fastperiod, slowperiod)
        return transform[-1]
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=max(fastperiod, slowperiod))
    return transform


def AROON(timeperiod=14, refresh_period=1):
    """Aroon"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.AROON(data_dict, timeperiod)
        return tuple([t[-1] for t in transform])
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=timeperiod)
    return transform


def AROONOSC(timeperiod=14, refresh_period=1):
    """Aroon Oscillator"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.AROONOSC(data_dict, timeperiod)
        return transform[-1]
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=timeperiod)
    return transform


def ATR(timeperiod=14, refresh_period=1):
    """Average True Range"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.ATR(data_dict, timeperiod)
        return transform[-1]
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=timeperiod)
    return transform


def AVGPRICE(refresh_period=1):
    """Average Price"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.AVGPRICE(data_dict)
        return transform[-1]
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=1)
    return transform


def BBANDS(timeperiod=5, nbdevup=2, nbdevdn=2, matype=0, refresh_period=1):
    """Bollinger Bands"""
    def talib_transform(data, sid):
        data_dict = _data_to_dict(data, sid)
        transform = talib.abstract.BBANDS(data_dict, timeperiod)
        return tuple([t[-1] for t in transform])
    transform = BatchTransform(func=talib_transform,
                               refresh_period=refresh_period,
                               window_length=timeperiod)
    return transform

