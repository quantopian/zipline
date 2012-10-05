#!/usr/bin/env python

from distutils.core import setup

setup(name='zipline',
      version='0.5.0',
      description='A backtester for financial algorithms.',
      author='Quantopian Inc.',
      author_email='opensource@quantopian.com',
      packages=['zipline'],
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
          ]
)
