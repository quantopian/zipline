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

import re
import sys
from operator import lt, gt, eq, le, ge
from os.path import (
    abspath,
    dirname,
    join,
)
from distutils.version import StrictVersion
from setuptools import setup, Extension
try:
    from Cython.Build import cythonize
    from numpy import get_include
    ext_modules = cythonize(
        [
            Extension(
                'zipline.assets._assets',
                ['zipline/assets/_assets.pyx'],
                include_dirs=[get_include()],
            ),
            Extension(
                'zipline.lib.adjusted_array',
                ['zipline/lib/adjusted_array.pyx'],
                include_dirs=[get_include()],
            ),
            Extension(
                'zipline.lib.adjustment',
                ['zipline/lib/adjustment.pyx'],
                include_dirs=[get_include()],
            ),
            Extension(
                'zipline.data.ffc.loaders._us_equity_pricing',
                ['zipline/data/ffc/loaders/_us_equity_pricing.pyx'],
                include_dirs=[get_include()],
            ),
        ]
    )
except ImportError:
    if 'build_ext' in sys.argv:
        raise
    ext_modules = []


def _filter_requirements(lines_iter):
    for line in lines_iter:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # pip install -r understands line with ;python_version<'3.0', but
        # whatever happens inside extras_requires doesn't.  Parse the line
        # manually and conditionally add it if needed.
        if ';' in line:
            requirement, version_spec = line.split(';')
            try:
                groups = re.match(
                    "(python_version)([<>=]{1,2})(')([0-9\.]+)(')(.*)",
                    version_spec,
                ).groups()
                comp = {
                    '<': lt,
                    '<=': le,
                    '=': eq,
                    '==': eq,
                    '>': gt,
                    '>=': ge,
                }[groups[1]]
                version_spec = StrictVersion(groups[3])
            except Exception as e:
                # My kingdom for a 'raise from'!
                raise ValueError(
                    "Couldn't parse requirement line; '%s'\n"
                    "Error was:\n"
                    "%r" % (line, e)
                )

            sys_version = '.'.join(list(map(str, sys.version_info[:3])))
            if comp(sys_version, version_spec):
                yield requirement
        else:
            yield line


def read_requirements(path):
    """
    Read a requirements.txt file, expressed as a path relative to Zipline root.
    """
    real_path = join(dirname(abspath(__file__)), path)
    with open(real_path) as f:
        return list(_filter_requirements(f.readlines()))


def setup_requires():
    requires = read_requirements('etc/requirements.txt')
    numpy_req = [req for req in requires if 'numpy' in req]
    assert len(numpy_req) == 1
    return numpy_req


def install_requires():
    return read_requirements('etc/requirements.txt')


def extras_requires():
    return {
        'dev': read_requirements('etc/requirements_dev.txt'),
        'talib': ['talib'],
    }


setup(
    name='zipline',
    version='0.8.0rc1',
    description='A backtester for financial algorithms.',
    author='Quantopian Inc.',
    author_email='opensource@quantopian.com',
    packages=['zipline'],
    ext_modules=ext_modules,
    scripts=['scripts/run_algo.py'],
    include_package_data=True,
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: System :: Distributed Computing',
    ],
    setup_requires=setup_requires(),
    install_requires=install_requires(),
    extras_require=extras_requires(),
    url="http://zipline.io"
)
