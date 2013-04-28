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
                    self.talib_fn.set_parameters({param : kwargs[param]})

            # get the lookback
            self.lookback = self.talib_fn.lookback

            def zipline_wrapper(data):
                # convert zipline dataframe to talib data_dict
                data_dict = dict()

                #TODO handle missing data
                for key in ['open', 'high', 'low', 'volume']:
                    if key in data:
                        data_dict[key] = data[key].values[:, 0]
                if 'close' in data:
                    data_dict['close'] = data['close'].values[:, 0]
                else:
                    data_dict['close'] = data['price'].values[:, 0]

                # call talib
                result = self.talib_fn(data_dict)

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
