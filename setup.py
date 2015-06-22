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

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        'zipline.assets._assets',
        ['zipline/assets/_assets.pyx'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'zipline.data.adjusted_array',
        ['zipline/data/adjusted_array.pyx'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'zipline.data.adjustment',
        ['zipline/data/adjustment.pyx'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'zipline.data.ffc.loaders._us_equity_pricing',
        ['zipline/data/ffc/loaders/_us_equity_pricing.pyx'],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name='zipline',
    version='0.8.0rc1',
    description='A backtester for financial algorithms.',
    author='Quantopian Inc.',
    author_email='opensource@quantopian.com',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
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
    install_requires=[
        'Logbook',
        'pytz',
        'requests',
        'numpy',
        'pandas',
        'six',
        'Cython',
    ],
    extras_require={
        'talib':  ["talib"],
    },
    url="http://zipline.io"
)
