#!/usr/bin/env python

from distutils.core import setup

setup(name='zipline',
      version='0.1',
      description='A backtester for financial algorithms.',
      author='Quantopian Inc.',
      author_email='opensource@quantopian.com',
      packages=['zipline'],
      long_description=open('README.md').read(),
      license='BSD',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: C',
          'Programming Language :: Cython',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Office/Business :: Financial',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: System :: Distributed Computing',
      ],
      install_requires=[
          'msgpack-python',
          'humanhash',
          'iso8601',
          'Logbook',
          'blist',
          'pytz',
          'numpy',
          'Cython',
          'pandas'
          ]
)
