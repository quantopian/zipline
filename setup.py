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
        return (
            key == 'build_ext'
            or super(LazyBuildExtCommandClass, self).__contains__(key)
        )

    def __setitem__(self, key, value):
        if key == 'build_ext':
            raise AssertionError("build_ext overridden!")
        super(LazyBuildExtCommandClass, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key != 'build_ext':
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


ext_modules = [
    Extension('zipline.assets._assets', ['zipline/assets/_assets.pyx']),
    Extension('zipline.lib.adjustment', ['zipline/lib/adjustment.pyx']),
    Extension('zipline.lib._factorize', ['zipline/lib/_factorize.pyx']),
    Extension(
        'zipline.lib._float64window', ['zipline/lib/_float64window.pyx']
    ),
    Extension('zipline.lib._int64window', ['zipline/lib/_int64window.pyx']),
    Extension('zipline.lib._uint8window', ['zipline/lib/_uint8window.pyx']),
    Extension('zipline.lib._labelwindow', ['zipline/lib/_labelwindow.pyx']),
    Extension('zipline.lib.rank', ['zipline/lib/rank.pyx']),
    Extension('zipline.data._equities', ['zipline/data/_equities.pyx']),
    Extension('zipline.data._adjustments', ['zipline/data/_adjustments.pyx']),
    Extension('zipline._protocol', ['zipline/_protocol.pyx']),
    Extension('zipline.gens.sim_engine', ['zipline/gens/sim_engine.pyx']),
    Extension(
        'zipline.data._minute_bar_internal',
        ['zipline/data/_minute_bar_internal.pyx']
    )
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
    'pandas': '<0.18',
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


REQ_PATTERN = re.compile("(?P<name>[^=<>]+)(?P<comp>[<=>]{1,2})(?P<spec>[^;]+)"
                         "(?:(;\W*python_version\W*(?P<pycomp>[<=>]{1,2})\W*"
                         "(?P<pyspec>[0-9\.]+)))?")


def _conda_format(req):
    def _sub(m):
        name = m.group('name').lower()
        if name == 'numpy':
            return 'numpy x.x'

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

conda_build = os.path.basename(sys.argv[0]) in ('conda-build',  # unix
                                                'conda-build-script.py')  # win

setup_requires = setup_requirements(
    'etc/requirements.txt',
    ('Cython', 'numpy'),
    strict_bounds=conda_build,
    conda_format=conda_build,
)

conditional_arguments = {
    'setup_requires' if not conda_build else 'build_requires': setup_requires,
}

setup(
    name='zipline',
    url="http://zipline.io",
    version=versioneer.get_version(),
    cmdclass=LazyBuildExtCommandClass(versioneer.get_cmdclass()),
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
        'Programming Language :: Python :: 3.4',
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
