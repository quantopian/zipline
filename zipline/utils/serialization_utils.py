#
# Copyright 2015 Quantopian, Inc.
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

from six import BytesIO
import pickle
from functools import partial

from zipline.assets import AssetFinder
from zipline.finance.trading import TradingEnvironment

# Label for the serialization version field in the state returned by
# __getstate__.
VERSION_LABEL = '_stateversion_'


def _persistent_id(obj):
    if isinstance(obj, AssetFinder):
        return AssetFinder.PERSISTENT_TOKEN
    if isinstance(obj, TradingEnvironment):
        return TradingEnvironment.PERSISTENT_TOKEN
    return None


def _persistent_load(persid, env):
    if persid == AssetFinder.PERSISTENT_TOKEN:
        return env.asset_finder
    if persid == TradingEnvironment.PERSISTENT_TOKEN:
        return env


def dumps_with_persistent_ids(obj, protocol=None):
    """
    Performs a pickle dumps on the given object, substituting all references to
    a TradingEnvironment or AssetFinder with tokenized representations.

    All arguments are passed to pickle.Pickler and are described therein.
    """
    file = BytesIO()
    pickler = pickle.Pickler(file, protocol)
    pickler.persistent_id = _persistent_id
    pickler.dump(obj)
    return file.getvalue()


def loads_with_persistent_ids(str, env):
    """
    Performs a pickle loads on the given string, substituting the given
    TradingEnvironment in to any tokenized representations of a
    TradingEnvironment or AssetFinder.

    Parameters
    ----------
    str : String
        The string representation of the object to be unpickled.
    env : TradingEnvironment
        The TradingEnvironment to be inserted to the unpickled object.

    Returns
    -------
    obj
       An unpickled object formed from the parameter 'str'.
    """
    file = BytesIO(str)
    unpickler = pickle.Unpickler(file)
    unpickler.persistent_load = partial(_persistent_load, env=env)
    return unpickler.load()
