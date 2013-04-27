import functools
import talib
from zipline.transforms import BatchTransform


#TODO multi-SID transforms: Beta, Correl

#======== Batch Transform Template


def TALibTransform(talib_fn, *args, **kwargs):
    talib_fn.__name__ = talib_fn.info['name']
    @functools.wraps(talib_fn)
    def TALIB_wrapper(data, sid):
        data_dict = dict()
        for key in ['open', 'high', 'low', 'volume']:
            if key in data:
                data_dict[key] = data[key][sid].values
        if 'close' in data:
            data_dict['close'] = data['close'][sid].values
        else:
            data_dict['close'] = data['price'][sid].values

        # call the TAlib function
        transform = talib_fn(data_dict, *args, **kwargs)

        # return the last result, checking for multiple results
        if isinstance(transform, (list, tuple)):
            return tuple([t[-1] for t in transform])
        else:
            return transform[-1]
    return TALIB_wrapper


