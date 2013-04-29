import numpy as np
import talib
import copy
from zipline.transforms import BatchTransform


def make_transform(talib_fn):
    """
    A factory for BatchTransforms based on TALIB abstract functions.
    """
    class TALibTransform(BatchTransform):
        def __init__(self, sid, **kwargs):

            # check for BatchTransform refresh_period
            refresh_period = kwargs.pop('refresh_period', 0)
            key_map = {'high':   kwargs.pop('high', 'high'),
                       'low':    kwargs.pop('low', 'low'),
                       'open':   kwargs.pop('open', 'open'),
                       'volume': kwargs.pop('volume', 'volume'),
                       'close':  kwargs.pop('close', 'price')}

            # Make deepcopy of talib abstract function.
            # This is necessary because talib abstract functions remember
            # state, including parameters, and we need to set the parameters
            # in order to compute the lookback period that will determine the
            # BatchTransform window_length. TALIB has no way to restore default
            # parameters, so the deepcopy lets us change this function's
            # parameters without affecting other TALibTransforms of the same
            # function.
            self.talib_fn = copy.deepcopy(talib_fn)

            # set the parameters
            for param in self.talib_fn.get_parameters().keys():
                if param in kwargs:
                    self.talib_fn.set_parameters({param: kwargs[param]})

            # get the lookback
            self.lookback = self.talib_fn.lookback

            def zipline_wrapper(data):
                # get required TA-Lib input names
                if 'price' in self.talib_fn.input_names:
                    req_inputs = [self.talib_fn.input_names['price']]
                elif 'prices' in self.talib_fn.input_names:
                    req_inputs = self.talib_fn.input_names['prices']
                else:
                    req_inputs = []

                # build talib_data from zipline data
                talib_data = dict()
                for talib_key, zipline_key in key_map.iteritems():
                    # if zipline_key is found, add it to talib_data
                    if zipline_key in data:
                        talib_data[talib_key] = data[zipline_key].values[:, 0]
                    # if zipline_key is not found and not required, add zeros
                    elif talib_key not in req_inputs:
                        talib_data[talib_key] = np.zeros(data.shape[1])
                    # if zipline key is not found and required, raise error
                    else:
                        raise KeyError(
                            'Tried to set required TA-Lib data with key '
                            '\'{0}\' but no Zipline data is available under '
                            'expected key \'{1}\'.'.format(
                                talib_key, zipline_key))

                # call talib
                result = self.talib_fn(talib_data)

                # keep only the most recent result
                if isinstance(result, (list, tuple)):
                    return tuple([r[-1] for r in result])
                else:
                    return result[-1]

            super(TALibTransform, self).__init__(
                func=zipline_wrapper,
                sids=sid,
                refresh_period=refresh_period,
                window_length=max(1, self.lookback))

        def __repr__(self):
            return 'Zipline BatchTransform: {0}'.format(
                self.talib_fn.info['name'])

    # bind a class docstring to reveal parameters
    TALibTransform.__doc__ = getattr(talib, talib_fn.info['name']).__doc__

    #return class
    return TALibTransform


# add all TA-Lib functions to locals
for name in talib.abstract.__all__:
    fn = getattr(talib.abstract, name)
    if name != 'Function':
        locals()[name] = make_transform(fn)
