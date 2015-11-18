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


class LazyCythonizingList(list):
    cythonized = False

    def lazy_cythonize(self):
        if self.cythonized:
            return
        self.cythonized = True

        from Cython.Build import cythonize
        from numpy import get_include

        self[:] = cythonize(
            [
                Extension(*ext_args, include_dirs=[get_include()])
                for ext_args in self
            ]
        )

    def __iter__(self):
        self.lazy_cythonize()
        return super(LazyCythonizingList, self).__iter__()

    def __getitem__(self, num):
        self.lazy_cythonize()
        return super(LazyCythonizingList, self).__getitem__(num)


ext_modules = LazyCythonizingList([
    ('zipline.assets._assets', ['zipline/assets/_assets.pyx']),
    ('zipline.lib.adjusted_array', ['zipline/lib/adjusted_array.pyx']),
    ('zipline.lib.adjustment', ['zipline/lib/adjustment.pyx']),
    ('zipline.lib.rank', ['zipline/lib/rank.pyx']),
    (
        'zipline.data._equities',
        ['zipline/data/_equities.pyx'],
    ),
    (
        'zipline.data._adjustments',
        ['zipline/data/_adjustments.pyx'],
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


def pre_setup():
    if not set(sys.argv) & {'install', 'develop', 'egg_info', 'bdist_wheel'}:
        return

    try:
        import pip
        if StrictVersion(pip.__version__) < StrictVersion('7.1.0'):
            raise AssertionError(
                "Zipline installation requires pip>=7.1.0, but your pip "
                "version is {version}. \n"
                "You can upgrade your pip with "
                "'pip install --upgrade pip'.".format(
                    version=pip.__version__,
                )
            )
    except ImportError:
        raise AssertionError("Zipline installation requires pip")

    required = ('Cython', 'numpy')
    for line in module_requirements('etc/requirements.txt', required):
        pip.main(['install', line])


pre_setup()


setup(
    name='zipline',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
    extras_require=extras_requires(),
    url="http://zipline.io",
)
