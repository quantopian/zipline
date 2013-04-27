import functools
import talib
from zipline.transforms import BatchTransform


#TODO multi-SID transforms: Beta, Correl

#======== Batch Transform Template

def TALibTransform(talib_fn, *args, **kwargs):

    talib_fn.__name__ = talib_fn.info['name']

    @functools.wraps(talib_fn)
    def TALIB_wrapper(data):
        data_dict = dict()
        #TODO handle missing data
        for key in ['open', 'high', 'low', 'volume']:
            if key in data:
                data_dict[key] = data[key].values[:, 0]
        if 'close' in data:
            data_dict['close'] = data['close'].values[:, 0]
        else:
            data_dict['close'] = data['price'].values[:, 0]

        # call the TAlib function
        transform = talib_fn(data_dict, *args, **kwargs)

        # return the last result, checking for multiple results
        if isinstance(transform, (list, tuple)):
            return tuple([t[-1] for t in transform])
        else:
            return transform[-1]
    return TALIB_wrapper


#========= TA-LIB WRAPPERS


def AD(sid, refresh_period=1):
    """Chaikin A/D Line"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.AD),
        refresh_period=refresh_period,
        window_length=1)
    return transform


def ADOSC(sid, fastperiod=3, slowperiod=10, refresh_period=1):
    """Chaikin A/D Oscillator"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.ADOSC, fastperiod, slowperiod),
        refresh_period=refresh_period,
        window_length=max(fastperiod, slowperiod))
    return transform


def ADX(sid, timeperiod=14, refresh_period=1):
    """Average Directional Movement Index"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.ADX, timeperiod),
        refresh_period=refresh_period,
        window_length=timeperiod)
    return transform


def ADXR(sid, timeperiod=14, refresh_period=1):
    """Average Directional Movement Index Rating"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.ADXR, timeperiod),
        refresh_period=refresh_period,
        window_length=timeperiod)
    return transform


def APO(sid, fastperiod=12, slowperiod=26, matype=0, refresh_period=1):
    """Absolute Price Oscillator"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.APO, fastperiod, slowperiod),
        refresh_period=refresh_period,
        window_length=max(fastperiod, slowperiod))
    return transform


def AROON(sid, timeperiod=14, refresh_period=1):
    """Aroon"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.AROON, timeperiod),
        refresh_period=refresh_period,
        window_length=timeperiod)
    return transform


def AROONOSC(sid, timeperiod=14, refresh_period=1):
    """Aroon Oscillator"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.AROONOSC, timeperiod),
        refresh_period=refresh_period,
        window_length=timeperiod)
    return transform


def ATR(sid, timeperiod=14, refresh_period=1):
    """Average True Range"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.ATR, timeperiod),
        refresh_period=refresh_period,
        window_length=timeperiod)
    return transform


def AVGPRICE(sid, refresh_period=1):
    """Average True Range"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.AVGPRICE),
        refresh_period=refresh_period,
        window_length=1)
    return transform


def BBANDS(sid,
           timeperiod=5,
           nbdevup=2,
           nbdevdn=2,
           matype=0,
           refresh_period=1):
    """Bollinger Bands"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(
            talib.abstract.BBANDS, timeperiod, nbdevup, nbdevdn, matype),
        refresh_period=refresh_period,
        window_length=timeperiod)
    return transform


def BOP(sid, refresh_period=1):
    """Balance of Power"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.BOP),
        refresh_period=refresh_period,
        window_length=1)
    return transform


def CCI(sid, timeperiod=14, refresh_period=1):
    """Commodity Channel Index"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.CCI, timeperiod),
        refresh_period=refresh_period,
        window_length=timeperiod)
    return transform


def CDL2CROWS(sid, refresh_period=1):
    """Two Crows"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.CDL2CROWS),
        refresh_period=refresh_period,
        window_length=1)
    return transform


def CDL3BLACKCROWS(sid, refresh_period=1):
    """Two Crows"""
    transform = BatchTransform(
        sids=sid,
        func=TALibTransform(talib.abstract.CDL2CROWS),
        refresh_period=refresh_period,
        window_length=1)
    return transform
