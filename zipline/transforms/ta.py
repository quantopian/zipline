#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import talib
import copy
from zipline.transforms import BatchTransform


def make_transform(talib_fn, name):
    """
    A factory for BatchTransforms based on TALIB abstract functions.
    """
    # make class docstring
    header = '\n#---- TA-Lib docs\n\n'
    talib_docs = getattr(talib, talib_fn.info['name']).__doc__
    divider1 = '\n#---- Default mapping (TA-Lib : Zipline)\n\n'
    mappings = '\n'.join('        {0} : {1}'.format(k, v)
                         for k, v in talib_fn.input_names.items())
    divider2 = '\n\n#---- Zipline docs\n'
    help_str = (header + talib_docs + divider1 + mappings
                + divider2)

    class TALibTransform(BatchTransform):
        __doc__ = help_str + """
        TA-Lib keyword arguments must be passed at initialization. For
        example, to construct a moving average with timeperiod of 5, pass
        "timeperiod=5" during initialization.

        All abstract TA-Lib functions accept a data dictionary containing
        'open', 'high', 'low', 'close', and 'volume' keys, even if they do
        not require those keys to run. For example, talib.MA (moving
        average) is always computed using the data under the 'close'
        key. By default, Zipline constructs this data dictionary with the
        appropriate sid data, but users may overwrite this by passing
        mappings as keyword arguments. For example, to compute the moving
        average of the sid's high, provide "close = 'high'" and Zipline's
        'high' data will be used as TA-Lib's 'close' data. Similarly, if a
        user had a data column named 'Oil', they could compute its moving
        average by passing "close='Oil'".


        Example
        --------

        A moving average of a data column called 'Oil' with timeperiod 5,
        for sid 'XYZ':
            talib.transforms.ta.MA('XYZ', close='Oil', timeperiod=5)

        The user could find the default arguments and mappings by calling:
            help(zipline.transforms.ta.MA)


        Arguments
        ---------

        sid : zipline sid

        open   : string, default 'open'
        high   : string, default 'high'
        low    : string, default 'low'
        close  : string, default 'price'
        volume : string, default 'volume'

        refresh_period : int, default 0
            The refresh_period of the BatchTransform determines the number
            of iterations that pass before the BatchTransform updates its
            internal data.

        **kwargs : any arguments to be passed to the TA-Lib function.
        """

        def __init__(self,
                     sid,
                     close='price',
                     open='open',
                     high='high',
                     low='low',
                     volume='volume',
                     refresh_period=0,
                     **kwargs):

            key_map = {'high': high,
                       'low': low,
                       'open': open,
                       'volume': volume,
                       'close': close}

            # Rename window_length to timeperiod to conform with
            # external batch_transform interface.
            if 'window_length' in kwargs:
                kwargs['timeperiod'] = kwargs['window_length']

            self.call_kwargs = kwargs

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
                window_length=max(1, self.lookback + 1))

        def __repr__(self):
            return 'Zipline BatchTransform: {0}'.format(
                self.talib_fn.info['name'])

    TALibTransform.__name__ = name
    #return class
    return TALibTransform


# add all TA-Lib functions to locals
for name in talib.abstract.__all__:
    fn = getattr(talib.abstract, name)
    if name != 'Function':
        locals()[name] = make_transform(fn, name)
