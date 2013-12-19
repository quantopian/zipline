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

from functools import wraps
import zipline.api

import threading
context = threading.local()


def get_algo_instance():
    return getattr(context, 'algorithm', None)


def set_algo_instance(algo):
    context.algorithm = algo


def api_method(f):
    # Decorator that adds the decorated class method as a callable
    # function (wrapped) to zipline.api
    @wraps(f)
    def wrapped(*args, **kwargs):
        # Get the instance and call the method
        return getattr(get_algo_instance(), f.__name__)(*args, **kwargs)
    # Add functor to zipline.api
    setattr(zipline.api, f.__name__, wrapped)
    zipline.api.__all__.append(f.__name__)

    return f
