#
# Copyright 2014 Quantopian, Inc.
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

from functools import wraps

import zipline.api
from zipline.utils.algo_instance import get_algo_instance, set_algo_instance


class ZiplineAPI(object):
    """
    Context manager for making an algorithm instance available to zipline API
    functions within a scoped block.
    """

    def __init__(self, algo_instance):
        self.algo_instance = algo_instance

    def __enter__(self):
        """
        Set the given algo instance, storing any previously-existing instance.
        """
        self.old_algo_instance = get_algo_instance()
        set_algo_instance(self.algo_instance)

    def __exit__(self, _type, _value, _tb):
        """
        Restore the algo instance stored in __enter__.
        """
        set_algo_instance(self.old_algo_instance)


def api_method(f):
    # Decorator that adds the decorated class method as a callable
    # function (wrapped) to zipline.api
    @wraps(f)
    def wrapped(*args, **kwargs):
        # Get the instance and call the method
        algo_instance = get_algo_instance()
        if algo_instance is None:
            raise RuntimeError(
                'zipline api method %s must be called during a simulation.'
                % f.__name__
            )
        return getattr(algo_instance, f.__name__)(*args, **kwargs)
    # Add functor to zipline.api
    setattr(zipline.api, f.__name__, wrapped)
    zipline.api.__all__.append(f.__name__)
    f.is_api_method = True
    return f


def require_not_initialized(exception):
    """
    Decorator for API methods that should only be called during or before
    TradingAlgorithm.initialize.  `exception` will be raised if the method is
    called after initialize.

    Examples
    --------
    @require_not_initialized(SomeException("Don't do that!"))
    def method(self):
        # Do stuff that should only be allowed during initialize.
    """
    def decorator(method):
        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if self.initialized:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator


def require_initialized(exception):
    """
    Decorator for API methods that should only be called after
    TradingAlgorithm.initialize.  `exception` will be raised if the method is
    called before initialize has completed.

    Examples
    --------
    @require_initialized(SomeException("Don't do that!"))
    def method(self):
        # Do stuff that should only be allowed after initialize.
    """
    def decorator(method):
        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if not self.initialized:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator


def disallowed_in_before_trading_start(exception):
    """
    Decorator for API methods that cannot be called from within
    TradingAlgorithm.before_trading_start.  `exception` will be raised if the
    method is called inside `before_trading_start`.

    Examples
    --------
    @disallowed_in_before_trading_start(SomeException("Don't do that!"))
    def method(self):
        # Do stuff that is not allowed inside before_trading_start.
    """
    def decorator(method):
        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if self._in_before_trading_start:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator
