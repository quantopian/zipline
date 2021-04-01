#!/usr/bin/env python
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
import os
from setuptools import (
    Extension,
    find_packages,
    setup,
)

import versioneer


class LazyBuildExtCommandClass(dict):
    """
    Lazy command class that defers operations requiring Cython and numpy until
    they've actually been downloaded and installed by setup_requires.
    """

    def __contains__(self, key):
        return key == "build_ext" or super(LazyBuildExtCommandClass, self) \
            .__contains__(
            key
        )

    def __setitem__(self, key, value):
        if key == "build_ext":
            raise AssertionError("build_ext overridden!")
        super(LazyBuildExtCommandClass, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key != "build_ext":
            return super(LazyBuildExtCommandClass, self).__getitem__(key)

        from Cython.Distutils import build_ext as cython_build_ext
        import numpy

        # Cython_build_ext isn't a new-style class in Py2.
        class build_ext(cython_build_ext, object):
            """
            Custom build_ext command that lazily adds numpy's include_dir to
            extensions.
            """

            def build_extensions(self):
                """
                Lazily append numpy's include directory to Extension includes.

                This is done here rather than at module scope because setup.py
                may be run before numpy has been installed, in which case
                importing numpy and calling `numpy.get_include()` will fail.
                """
                numpy_incl = numpy.get_include()
                for ext in self.extensions:
                    ext.include_dirs.append(numpy_incl)

                super(build_ext, self).build_extensions()

        return build_ext


def window_specialization(typename):
    """Make an extension for an AdjustedArrayWindow specialization."""
    return Extension(
        "zipline.lib._{name}window".format(name=typename),
        ["zipline/lib/_{name}window.pyx".format(name=typename)],
        depends=["zipline/lib/_windowtemplate.pxi"],
    )


ext_options = dict(
    compiler_directives=dict(profile=True, language_level="3"), annotate=True
)
ext_modules = [
    Extension(name="zipline.assets._assets",
              sources=["zipline/assets/_assets.pyx"]),
    Extension(
        name="zipline.assets.continuous_futures",
        sources=["zipline/assets/continuous_futures.pyx"],
    ),
    Extension(name="zipline.lib.adjustment",
              sources=["zipline/lib/adjustment.pyx"]),
    Extension(name="zipline.lib._factorize",
              sources=["zipline/lib/_factorize.pyx"]),
    window_specialization("float64"),
    window_specialization("int64"),
    window_specialization("int64"),
    window_specialization("uint8"),
    window_specialization("label"),
    Extension(name="zipline.lib.rank",
              sources=["zipline/lib/rank.pyx"]),
    Extension(name="zipline.data._equities",
              sources=["zipline/data/_equities.pyx"]),
    Extension(
        name="zipline.data._adjustments",
        sources=["zipline/data/_adjustments.pyx"]
    ),
    Extension(name="zipline._protocol",
              sources=["zipline/_protocol.pyx"]),
    Extension(
        name="zipline.finance._finance_ext",
        sources=["zipline/finance/_finance_ext.pyx"],
    ),
    Extension(name="zipline.gens.sim_engine",
              sources=["zipline/gens/sim_engine.pyx"]),
    Extension(
        name="zipline.data._minute_bar_internal",
        sources=["zipline/data/_minute_bar_internal.pyx"],
    ),
    Extension(name="zipline.data._resample",
              sources=["zipline/data/_resample.pyx"]),
]
for ext_module in ext_modules:
    ext_module.cython_directives = dict(language_level="3")

setup(
    version=versioneer.get_version(),
    test_suite='tests',
    cmdclass=LazyBuildExtCommandClass(versioneer.get_cmdclass()),
    entry_points={
        "console_scripts": [
            "zipline = zipline.__main__:main",
        ],
    },
    packages=find_packages(include=["zipline", "zipline.*"]),
    ext_modules=ext_modules,
    package_data={
        root.replace(os.sep, "."): ["*.pyi", "*.pyx", "*.pxi", "*.pxd"]
        for root, dirnames, filenames in os.walk("zipline")
        if "__pycache__" not in root
    },
)
