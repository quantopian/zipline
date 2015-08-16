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

from setuptools import setup, Extension, find_packages


class LazyCythonizingList(list):
    cythonized = False

    def lazy_cythonize(self):
        if self.cythonized:
            return
        self.cythonized = True
        from Cython.Build import cythonize
        import numpy as np
        self[:] = cythonize([Extension(e[0], e[1],
                                       include_dirs=[np.get_include()])
                             for e in self])

    def __iter__(self):
        self.lazy_cythonize()
        return super(LazyCythonizingList, self).__iter__()

    def __getitem__(self, num):
        self.lazy_cythonize()
        return super(LazyCythonizingList, self).__getitem__(num)

ext_modules = [
    ('zipline.assets._assets', ['zipline/assets/_assets.pyx']),
    ('zipline.lib.adjusted_array', ['zipline/lib/adjusted_array.pyx']),
    ('zipline.lib.adjustment', ['zipline/lib/adjustment.pyx']),
    ('zipline.data.ffc.loaders._us_equity_pricing',
     ['zipline/data/ffc/loaders/_us_equity_pricing.pyx']),
]

setup(
    name='zipline',
    version='0.8.0rc1',
    description='A backtester for financial algorithms.',
    author='Quantopian Inc.',
    author_email='opensource@quantopian.com',
    packages=find_packages('.', include=['zipline', 'zipline.*']),
    ext_modules=LazyCythonizingList(ext_modules),
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
    setup_requires=[
        'Cython',
        'numpy',
    ],
    install_requires=[
        'Logbook',
        'pytz',
        'requests',
        'numpy',
        'pandas',
        'six',
        'Cython',
        'contextlib2',
        'networkx',
        'scipy',
        'numexpr',
    ],
    extras_require={
        'talib':  ["talib"],
    },
    url="http://zipline.io"
)
