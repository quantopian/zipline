#
# Copyright 2013 Quantopian, Inc.
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

# This code is based on a unittest written by John Salvatier:
# https://github.com/pymc-devs/pymc/blob/pymc3/tests/test_examples.py

# Disable plotting
#
import matplotlib
matplotlib.use('Agg')

from os import path
import os
import fnmatch
import imp


def test_examples():
    os.chdir(example_dir())
    for fname in all_matching_files('.', '*.py'):
        yield check_example, fname


def all_matching_files(d, pattern):
    def addfiles(fls, dir, nfiles):
        nfiles = fnmatch.filter(nfiles, pattern)
        nfiles = [path.join(dir, f) for f in nfiles]
        fls.extend(nfiles)

    files = []
    path.walk(d, addfiles, files)
    return files


def example_dir():
    import zipline
    d = path.dirname(zipline.__file__)
    return path.join(path.abspath(d), 'examples/')


def check_example(p):
    imp.load_source('__main__', path.basename(p))
