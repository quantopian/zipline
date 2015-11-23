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

from distutils.version import StrictVersion
from itertools import starmap
from operator import lt, gt, eq, le, ge
from os.path import (
    abspath,
    dirname,
    join,
)
from pkg_resources import resource_filename
import re
from setuptools import (
    Extension,
    find_packages,
    setup,
)
import sys

import versioneer


class LazyCommandClass(dict):
    """
    Lazy command class that defers operations requiring Cython and numpy until
    they've actually been downloaded and installed by setup_requires.
    """
    def __contains__(self, key):
        return (
            key == 'build_ext'
            or super(LazyCommandClass, self).__contains__(key)
        )

    def __setitem__(self, key, value):
        if key == 'build_ext':
            raise AssertionError("build_ext overridden!")
        super(LazyCommandClass, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key != 'build_ext':
            return super(LazyCommandClass, self).__getitem__(key)

        from Cython.Distutils import build_ext as cython_build_ext

        class build_ext(cython_build_ext):
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
                numpy_incl = resource_filename('numpy', 'core/include')
                for ext in self.extensions:
                    ext.include_dirs.append(numpy_incl)

                # This explicitly calls the superclass method rather than the
                # usual super() invocation because distutils' build_class, of
                # which Cython's build_ext is a subclass, is an old-style class
                # in Python 2, which doesn't support `super`.
                cython_build_ext.build_extensions(self)
        return build_ext


ext_modules = list(starmap(Extension, (
    ('zipline.assets._assets', ['zipline/assets/_assets.pyx']),
    ('zipline.lib.adjustment', ['zipline/lib/adjustment.pyx']),
    ('zipline.lib._float64window', ['zipline/lib/_float64window.pyx']),
    ('zipline.lib._int64window', ['zipline/lib/_int64window.pyx']),
    ('zipline.lib._uint8window', ['zipline/lib/_uint8window.pyx']),
    ('zipline.lib.rank', ['zipline/lib/rank.pyx']),
    (
        'zipline.data._equities',
        ['zipline/data/_equities.pyx'],
    ),
    (
        'zipline.data._adjustments',
        ['zipline/data/_adjustments.pyx'],
    ),
)))

STR_TO_CMP = {
    '<': lt,
    '<=': le,
    '=': eq,
    '==': eq,
    '>': gt,
    '>=': ge,
}


def _filter_requirements(lines_iter):
    for line in lines_iter:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # pip install -r understands line with ;python_version<'3.0', but
        # whatever happens inside extras_requires doesn't.  Parse the line
        # manually and conditionally add it if needed.
        if ';' not in line:
            yield line
            continue

        requirement, version_spec = line.split(';')
        try:
            groups = re.match(
                "(python_version)([<>=]{1,2})(')([0-9\.]+)(')(.*)",
                version_spec,
            ).groups()
            comp = STR_TO_CMP[groups[1]]
            version_spec = StrictVersion(groups[3])
        except Exception as e:
            # My kingdom for a 'raise from'!
            raise AssertionError(
                "Couldn't parse requirement line; '%s'\n"
                "Error was:\n"
                "%r" % (line, e)
            )

        sys_version = '.'.join(list(map(str, sys.version_info[:3])))
        if comp(sys_version, version_spec):
            yield requirement


REQ_UPPER_BOUNDS = {
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


def read_requirements(path, strict_bounds):
    """
    Read a requirements.txt file, expressed as a path relative to Zipline root.

    Returns requirements with the pinned versions as lower bounds
    if `strict_bounds` is falsey.
    """
    real_path = join(dirname(abspath(__file__)), path)
    with open(real_path) as f:
        reqs = _filter_requirements(f.readlines())

        if strict_bounds:
            return list(reqs)
        else:
            return list(map(_with_bounds, reqs))


def install_requires(strict_bounds=False):
    return read_requirements('etc/requirements.txt',
                             strict_bounds=strict_bounds)


def extras_requires():
    dev_reqs = read_requirements('etc/requirements_dev.txt',
                                 strict_bounds=True)
    talib_reqs = ['TA-Lib==0.4.9']
    return {
        'dev': dev_reqs,
        'talib': talib_reqs,
        'all': dev_reqs + talib_reqs,
    }


def module_requirements(requirements_path, module_names):
    module_names = set(module_names)
    found = set()
    module_lines = []
    parser = re.compile("([^=<>]+)([<=>]{1,2})(.*)")
    for line in read_requirements(requirements_path, strict_bounds=False):
        match = parser.match(line)
        if match is None:
            raise AssertionError("Could not parse requirement: '%s'" % line)

        groups = match.groups()
        name = groups[0]
        if name in module_names:
            found.add(name)
            module_lines.append(line)

    if found != module_names:
        raise AssertionError(
            "No requirements found for %s." % module_names - found
        )
    return module_lines


def setup_requires():
    if not set(sys.argv) & {'install', 'develop', 'egg_info', 'bdist_wheel'}:
        return []

    required = ('Cython', 'numpy')
    return module_requirements('etc/requirements.txt', required)


setup(
    name='zipline',
    version=versioneer.get_version(),
    cmdclass=LazyCommandClass(versioneer.get_cmdclass()),
    description='A backtester for financial algorithms.',
    author='Quantopian Inc.',
    author_email='opensource@quantopian.com',
    packages=find_packages('.', include=['zipline', 'zipline.*']),
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
        'Programming Language :: Python :: 3.4',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: System :: Distributed Computing',
    ],
    install_requires=install_requires(),
    setup_requires=setup_requires(),
    extras_require=extras_requires(),
    url="http://zipline.io",
)
