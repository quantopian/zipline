#!/usr/bin/env python
#
# Copyright 2012 Quantopian, Inc.
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

from setuptools import setup, find_packages

setup(name='zipline',
      version='0.5.0',
      description='A backtester for financial algorithms.',
      author='Quantopian Inc.',
      author_email='opensource@quantopian.com',
      packages=find_packages(),
      long_description=open('README.md').read(),
      license='Apache 2.0',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Office/Business :: Financial',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: System :: Distributed Computing',
      ],
      install_requires=[
          'msgpack-python',
          'iso8601',
          'Logbook',
          'blist',
          'pytz',
          'numpy',
          'pandas'
          ],
      url="https://github.com/quantopian/zipline"
)
