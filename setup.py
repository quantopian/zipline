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
from __future__ import print_function
import os
import re
import sys
from operator import lt, gt, eq, le, ge
from os.path import (
    abspath,
    dirname,
    join,
)
from distutils.version import StrictVersion
from functools import partial
from setuptools import (
    Extension,
    find_packages,
    setup,
)

import versioneer


conda_build = os.path.basename(sys.argv[0]) in ('conda-build',  # unix
                                                'conda-build-script.py')  # win

if conda_build:
    # If conda-build is running this, then we're currently expanding the jinja
    # template in conda/zipline/meta.yaml, not actually installing. We don't
    # have numpy or Cython yet, but luckily we only need the names from
    # install_requires and build_requires.
    ext_modules = []
else:
    try:
        import Cython  # noqa
    except ImportError:
        raise Exception("Install Cython before zipline.")

    try:
        import numpy as np
    except ImportError:
        raise Exception("Install numpy before zipline.")

    NumpyExtension = partial(Extension, include_dirs=[np.get_include()])

    def window_specialization(typename):
        """Make an extension for an AdjustedArrayWindow specialization."""
        return NumpyExtension(
            'zipline.lib._{name}window'.format(name=typename),
            ['zipline/lib/_{name}window.pyx'.format(name=typename)],
            depends=['zipline/lib/_windowtemplate.pxi'],
        )

    ext_modules = [
        NumpyExtension('zipline.assets._assets',
                       ['zipline/assets/_assets.pyx']),
        NumpyExtension('zipline.assets.continuous_futures',
                       ['zipline/assets/continuous_futures.pyx']),
        NumpyExtension('zipline.lib.adjustment',
                       ['zipline/lib/adjustment.pyx']),
        NumpyExtension('zipline.lib._factorize',
                       ['zipline/lib/_factorize.pyx']),
        window_specialization('float64'),
        window_specialization('int64'),
        window_specialization('int64'),
        window_specialization('uint8'),
        window_specialization('label'),
        NumpyExtension('zipline.lib.rank', ['zipline/lib/rank.pyx']),
        NumpyExtension('zipline.data._equities',
                       ['zipline/data/_equities.pyx']),
        NumpyExtension('zipline.data._adjustments',
                       ['zipline/data/_adjustments.pyx']),
        NumpyExtension('zipline._protocol', ['zipline/_protocol.pyx']),
        NumpyExtension(
            'zipline.finance._finance_ext',
            ['zipline/finance/_finance_ext.pyx'],
        ),
        NumpyExtension('zipline.gens.sim_engine',
                       ['zipline/gens/sim_engine.pyx']),
        NumpyExtension(
            'zipline.data._minute_bar_internal',
            ['zipline/data/_minute_bar_internal.pyx']
        ),
        NumpyExtension(
            'zipline.data._resample',
            ['zipline/data/_resample.pyx']
        ),
        NumpyExtension(
            'zipline.pipeline.loaders.blaze._core',
            ['zipline/pipeline/loaders/blaze/_core.pyx'],
            depends=['zipline/lib/adjustment.pxd'],
        ),
    ]


STR_TO_CMP = {
    '<': lt,
    '<=': le,
    '=': eq,
    '==': eq,
    '>': gt,
    '>=': ge,
}

SYS_VERSION = '.'.join(list(map(str, sys.version_info[:3])))


def _filter_requirements(lines_iter, filter_names=None,
                         filter_sys_version=False):
    for line in lines_iter:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        match = REQ_PATTERN.match(line)
        if match is None:
            raise AssertionError(
                "Could not parse requirement: '%s'" % line)

        name = match.group('name')
        if filter_names is not None and name not in filter_names:
            continue

        if filter_sys_version and match.group('pyspec'):
            pycomp, pyspec = match.group('pycomp', 'pyspec')
            comp = STR_TO_CMP[pycomp]
            pyver_spec = StrictVersion(pyspec)
            if comp(SYS_VERSION, pyver_spec):
                # pip install -r understands lines with ;python_version<'3.0',
                # but pip install -e does not.  Filter here, removing the
                # env marker.
                yield line.split(';')[0]
            continue

        yield line


REQ_UPPER_BOUNDS = {
    'bcolz': '<1',
    'pandas': '<=0.22',
    'networkx': '<2.0',
}


def _with_bounds(req):
    try:
        req, lower = req.split('==')
    except ValueError:
        return req
    else:
        with_bounds = [req, '>=', lower]
        upper = REQ_UPPER_BOUNDS.get(req)
        if upper:
            with_bounds.extend([',', upper])
        return ''.join(with_bounds)


REQ_PATTERN = re.compile(
    r"(?P<name>[^=<>]+)(?P<comp>[<=>]{1,2})(?P<spec>[^;]+)"
    r"(?:(;\W*python_version\W*(?P<pycomp>[<=>]{1,2})\W*"
    r"(?P<pyspec>[0-9\.]+)))?"
)


def _conda_format(req):
    def _sub(m):
        name = m.group('name').lower()
        if name == 'numpy':
            return 'numpy x.x'
        if name == 'tables':
            name = 'pytables'

        formatted = '%s %s%s' % ((name,) + m.group('comp', 'spec'))
        pycomp, pyspec = m.group('pycomp', 'pyspec')
        if pyspec:
            # Compare the two-digit string versions as ints.
            selector = ' # [int(py) %s int(%s)]' % (
                pycomp, ''.join(pyspec.split('.')[:2]).ljust(2, '0')
            )
            return formatted + selector

        return formatted

    return REQ_PATTERN.sub(_sub, req, 1)


def read_requirements(path,
                      strict_bounds,
                      conda_format=False,
                      filter_names=None):
    """
    Read a requirements.txt file, expressed as a path relative to Zipline root.

    Returns requirements with the pinned versions as lower bounds
    if `strict_bounds` is falsey.
    """
    real_path = join(dirname(abspath(__file__)), path)
    with open(real_path) as f:
        reqs = _filter_requirements(f.readlines(), filter_names=filter_names,
                                    filter_sys_version=not conda_format)

        if not strict_bounds:
            reqs = map(_with_bounds, reqs)

        if conda_format:
            reqs = map(_conda_format, reqs)

        return list(reqs)


def install_requires(strict_bounds=False, conda_format=False):
    return read_requirements('etc/requirements.txt',
                             strict_bounds=strict_bounds,
                             conda_format=conda_format)


def extras_requires(conda_format=False):
    extras = {
        extra: read_requirements('etc/requirements_{0}.txt'.format(extra),
                                 strict_bounds=True,
                                 conda_format=conda_format)
        for extra in ('dev', 'talib')
    }
    extras['all'] = [req for reqs in extras.values() for req in reqs]

    return extras


def setup_requirements(requirements_path, module_names, strict_bounds,
                       conda_format=False):
    module_names = set(module_names)
    module_lines = read_requirements(requirements_path,
                                     strict_bounds=strict_bounds,
                                     conda_format=conda_format,
                                     filter_names=module_names)

    if len(set(module_lines)) != len(module_names):
        raise AssertionError(
            "Missing requirements. Looking for %s, but found %s."
            % (module_names, module_lines)
        )
    return module_lines


setup_requires = setup_requirements(
    'etc/requirements.txt',
    ('Cython', 'numpy'),
    strict_bounds=conda_build,
    conda_format=conda_build,
)

conditional_arguments = (
    {'build_requires': setup_requires} if conda_build else {}
)


setup(
    name='zipline',
    url="http://zipline.io",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A backtester for financial algorithms.',
    entry_points={
        'console_scripts': [
            'zipline = zipline.__main__:main',
        ],
    },
    author='Quantopian Inc.',
    author_email='opensource@quantopian.com',
    packages=find_packages(include=['zipline', 'zipline.*']),
    ext_modules=ext_modules,
    include_package_data=True,
    package_data={root.replace(os.sep, '.'):
                  ['*.pyi', '*.pyx', '*.pxi', '*.pxd']
                  for root, dirnames, filenames in os.walk('zipline')
                  if '__pycache__' not in root},
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: System :: Distributed Computing',
    ],
    install_requires=install_requires(conda_format=conda_build),
    extras_require=extras_requires(conda_format=conda_build),
    **conditional_arguments
)
