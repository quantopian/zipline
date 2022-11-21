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
from packaging.version import Version
import os
import numpy as np

# This is *not* a place to dump arbitrary classes/modules for convenience,
# it is a place to expose the public interfaces.
from zipline.utils.calendar_utils import get_calendar

from . import data
from . import finance
from . import gens
from . import utils
from .utils.numpy_utils import numpy_version
from .utils.pandas_utils import new_pandas
from .utils.run_algo import run_algorithm

# These need to happen after the other imports.
from .algorithm import TradingAlgorithm
from . import api
from zipline import extensions as ext
from zipline.finance.blotter import Blotter

# PERF: Fire a warning if calendars were instantiated during zipline import.
# Having calendars doesn't break anything per-se, but it makes zipline imports
# noticeably slower, which becomes particularly noticeable in the Zipline CLI.
from zipline.utils.calendar_utils import global_calendar_dispatcher

if global_calendar_dispatcher._calendars:
    import warnings

    warnings.warn(
        "Found TradingCalendar instances after zipline import.\n"
        "Zipline startup will be much slower until this is fixed!",
    )
    del warnings
del global_calendar_dispatcher

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

extension_args = ext.Namespace()


def load_ipython_extension(ipython):
    from .__main__ import zipline_magic

    ipython.register_magic_function(zipline_magic, "line_cell", "zipline")


if os.name == "nt":
    # we need to be able to write to our temp directoy on windows so we
    # create a subdir in %TMP% that has write access and use that as %TMP%
    def _():
        import atexit
        import tempfile

        tempfile.tempdir = tempdir = tempfile.mkdtemp()

        @atexit.register
        def cleanup_tempdir():
            import shutil

            shutil.rmtree(tempdir)

    _()
    del _

__all__ = [
    "Blotter",
    "TradingAlgorithm",
    "api",
    "data",
    "finance",
    "get_calendar",
    "gens",
    "run_algorithm",
    "utils",
    "extension_args",
]


def setup(
    self,
    np=np,
    numpy_version=numpy_version,
    Version=Version,
    new_pandas=new_pandas,
):
    """Lives in zipline.__init__ for doctests."""

    if numpy_version >= Version("1.14"):
        self.old_opts = np.get_printoptions()
        np.set_printoptions(legacy="1.13")
    else:
        self.old_opts = None

    if new_pandas:
        self.old_err = np.geterr()
        # old pandas has numpy compat that sets this
        np.seterr(all="ignore")
    else:
        self.old_err = None


def teardown(self, np=np):
    """Lives in zipline.__init__ for doctests."""

    if self.old_err is not None:
        np.seterr(**self.old_err)

    if self.old_opts is not None:
        np.set_printoptions(**self.old_opts)


del os
del np
del numpy_version
del Version
del new_pandas
