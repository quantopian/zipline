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
from setuptools import (
    find_packages,
    setup,
    Extension as _Extension,
)

import numpy as np
from Cython.Build import cythonize

import versioneer


def make_extension(*args, **kwargs):
    kwargs.setdefault('include_dirs', []).append(np.get_include())
    return _Extension(*args, **kwargs)


def window_specialization(typename):
    """Make an extension for an AdjustedArrayWindow specialization."""
    return make_extension(
        'zipline.lib._{name}window'.format(name=typename),
        ['zipline/lib/_{name}window.pyx'.format(name=typename)],
        depends=['zipline/lib/_windowtemplate.pxi'],
    )


def make_extensions():
    return cythonize([
        make_extension('zipline.assets._assets', ['zipline/assets/_assets.pyx']),
        make_extension(
            'zipline.assets.continuous_futures',
            ['zipline/assets/continuous_futures.pyx'],
        ),
        make_extension('zipline.lib.adjustment', ['zipline/lib/adjustment.pyx']),
        make_extension('zipline.lib._factorize', ['zipline/lib/_factorize.pyx']),
        window_specialization('float64'),
        window_specialization('int64'),
        window_specialization('int64'),
        window_specialization('uint8'),
        window_specialization('label'),
        make_extension('zipline.lib.rank', ['zipline/lib/rank.pyx']),
        make_extension('zipline.data._equities', ['zipline/data/_equities.pyx']),
        make_extension(
            'zipline.data._adjustments',
            ['zipline/data/_adjustments.pyx'],
        ),
        make_extension('zipline._protocol', ['zipline/_protocol.pyx']),
        make_extension(
            'zipline.finance._finance_ext',
            ['zipline/finance/_finance_ext.pyx'],
        ),
        make_extension('zipline.gens.sim_engine', ['zipline/gens/sim_engine.pyx']),
        make_extension(
            'zipline.data._minute_bar_internal',
            ['zipline/data/_minute_bar_internal.pyx']
        ),
        make_extension(
            'zipline.data._resample',
            ['zipline/data/_resample.pyx']
        ),
        make_extension(
            'zipline.pipeline.loaders.blaze._core',
            ['zipline/pipeline/loaders/blaze/_core.pyx'],
            depends=['zipline/lib/adjustment.pxd'],
        ),
    ])


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
            raise AssertionError("Could not parse requirement: %r" % line)

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


REQ_PATTERN = re.compile(
    r"(?P<name>[^=<>;]+)((?P<comp>[<=>]{1,2})(?P<spec>[^;]+))?"
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

        comp, spec = m.group('comp', 'spec')
        if comp and spec:
            formatted = '%s %s%s' % (name, comp, spec)
        else:
            formatted = name
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
                      conda_format=False,
                      filter_names=None):
    """
    Read a requirements file, expressed as a path relative to Zipline root.
    """
    real_path = join(dirname(abspath(__file__)), path)
    with open(real_path) as f:
        reqs = _filter_requirements(f.readlines(), filter_names=filter_names,
                                    filter_sys_version=not conda_format)

        if conda_format:
            reqs = map(_conda_format, reqs)

        return list(reqs)


def install_requires(conda_format=False):
    return read_requirements('etc/requirements.in', conda_format=conda_format)


def extras_requires(conda_format=False):
    extras = {
        extra: read_requirements('etc/requirements_{0}.in'.format(extra),
                                 conda_format=conda_format)
        for extra in ('dev', 'talib')
    }
    extras['all'] = [req for reqs in extras.values() for req in reqs]

    return extras


def setup_requirements(requirements_path, module_names,
                       conda_format=False):
    module_names = set(module_names)
    module_lines = read_requirements(requirements_path,
                                     conda_format=conda_format,
                                     filter_names=module_names)

    if len(set(module_lines)) != len(module_names):
        raise AssertionError(
            "Missing requirements. Looking for %s, but found %s."
            % (module_names, module_lines)
        )
    return module_lines


conda_build = os.path.basename(sys.argv[0]) in ('conda-build',  # unix
                                                'conda-build-script.py')  # win

if conda_build:
    conditional_arguments = {
        'build_requires': setup_requirements(
            'etc/requirements_build.in',
            ('Cython', 'numpy'),
            conda_format=conda_build,
        )
    }
else:
    conditional_arguments = {}

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
    ext_modules=make_extensions(),
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
        'Programming Language :: Python :: 3.6',
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
