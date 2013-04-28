import talib
from zipline.transforms import BatchTransform


def make_transform(talib_fn):
    """
    A factory for BatchTransforms based on TALIB abstract functions.
    """
    class TALibTransform(BatchTransform):
        def __init__(self, sid, refresh_period=0, **kwargs):

            self.talib_fn = talib_fn

            # get default talib parameters
            self.talib_parameters = talib_fn.get_parameters()

            # update new parameters from kwargs
            for k, v in kwargs.iteritems():
                if k in self.talib_parameters:
                    self.talib_parameters[k] = v

            def zipline_wrapper(data):
                # Set the parameters at each iteration in case the same
                # abstract talib function is being used in another
                # BatchTransform with different parameters.
                # FIXME -- this might not be necessary if the abstract
                # functions can be copied into separate objects.
                self.talib_fn.set_parameters(self.talib_parameters)

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
                window_length=max(1, self.talib_fn.lookback + 1))

    # bind a class docstring to reveal parameters
    TALibTransform.__doc__ = getattr(talib, talib_fn.info['name']).__doc__

    #return class
    return TALibTransform


# add all TA-Lib functions to locals
for name in talib.abstract.__all__:
    fn = getattr(talib.abstract, name)
    if name != 'Function':
        locals()[name] = make_transform(fn)
