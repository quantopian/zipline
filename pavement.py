import os
import re
import sys
import glob
import time
from distutils.dep_util import newer

#from distutils.extension import Extension
from setuptools.extension import Extension

from paver.easy import options, Bunch, task, needs, path, info
from paver.setuputils import install_distutils_tasks, \
    find_packages, find_package_data
from subprocess import call
install_distutils_tasks()

# =========
# Compilers
# =========

try:
    from Cython.Compiler.Main import compile
    from Cython.Distutils import build_ext
    have_cython = True
except ImportError:
    have_cython = False

try:
    import numpy as np
    have_numpy = True
except:
    have_numpy = False

# ===================
# Release Information
# ===================

PACKAGE  = 'zipline'
SRC_PATH = 'zipline'

MAJOR = 0
MINOR = 1
MICRO = 0
DEVELOPMENT = True

if DEVELOPMENT:
    VERSION = '%d.%d.%d dev' % (MAJOR, MINOR, MICRO)
else:
    VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# The PyPi page
DESCRIPTION = open('README.md').read()
EMAIL='dev@quantopian.com'

# ===========
# Setuputils
# ===========

def parse_requirements(file_name):
    requirements = []
    for line in open(file_name, 'r').read().split('\n'):
        if re.match(r'(\s*#)|(\s*$)', line):
            continue
        if re.match(r'\s*-e\s+', line):
            requirements.append(re.sub(r'\s*-e\s+.*#egg=(.*)$', r'\1', line))
        elif re.match(r'\s*-f\s+', line):
            pass
        else:
            requirements.append(line)
    return requirements

example = Extension(
    "zipline/speedups/example", ["zipline/speedups/example.pyx"],
     #include_dirs=[np.get_include()],
)

# ============
# Dependencies
# ============

install_requires =  (
    parse_requirements('./etc/requirements.txt') +
    parse_requirements('./etc/requirements_sci.txt')
)
tests_require = install_requires + parse_requirements('./etc/requirements_dev.txt')

# ========
# seutp.py
# ========

if have_numpy and have_cython:
    cext = [example]
else:
    cext = []

options(
    sphinx = Bunch(
        builddir="_build",
        sourcedir=""
    ),
    setup = Bunch(
          name                   = PACKAGE,
          version                = VERSION,
          packages               = find_packages(),
          package_data           = find_package_data(
              SRC_PATH,
              package = PACKAGE,
              only_in_packages = False
          ),
          long_description       = DESCRIPTION,
          install_requires       = install_requires,
          tests_require          = tests_require,
          test_suite             = 'nose.collector',
          include_package_data   = True,
          zip_safe               = False,
          classifiers            = [
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
          ext_modules            = cext,
          cmdclass               = {
              'build_ext': build_ext
          },
          entry_points           = {
              'console_scripts': [
                  'zipline = zipline.core.interpreter:main',
              ]
          },
      ),
)

# ============
# C Extensions
# ============

@task
def clean_inplace():
    """
    Remove shared objects and C files from the extension
    directory.
    """
    for fn in glob.glob(os.path.join(SRC_PATH, 'speedups', '*.c')):
        p = path(fn)
        p.remove()

    for fn in glob.glob(os.path.join(SRC_PATH, 'speedups', '*.so')):
        p = path(fn)
        p.remove()

@task
def build_cython():
    for fn in glob.glob(os.path.join(SRC_PATH, 'speedups', '*.pyx')):
        p = path(fn)

        modname = p.splitext()[0].basename()
        dest = p.splitext()[0] + '.c'

        if newer(p.abspath(), dest.abspath()):
            info('cython %s -o %s'%(p, dest.basename()))
            compile(p.abspath(), full_module_name=modname)

@task
@needs(['build_cython', 'setuptools.command.build_ext'])
def build_ext():
     pass

# ======
# Tasks
# ======

# Because I'm lazy
stuff_i_want_in_my_debug_shell = [
    ('qutil', 'zipline.util', []),
    ('zmq', 'zmq', []),
]

@task
def coverage():
    """
    Run the devsever under the coverage reporter, generate the
    coverage report.
    """
    call('nosetests zipline', shell=True)
    call('coverage html', shell=True)
    call('chromium %s/cover/index.html' % (os.path.abspath(".")), shell=True)

@task
def profile():
    """
    Runtime profiling using cProfile, use pStats to find heavy
    calls. Or use python -m pstats to get more granular
    statistics about runtimes.
    """
    try:
        call('python -m cProfile -o zipline.prof qexec/web/devserver.py --hostsettings', shell=True)
    except KeyboardInterrupt:
        pass
    import pstats
    time.sleep(1) # wait for disk io

    p = pstats.Stats('zipline.prof')
    # Print the hundred heaviest function calls
    p.sort_stats('time').print_stats(100)

@task
def lineprofile():
    """
    Line by line profiler. Find hotspots in your code using the
    @profile decorator .
    """
    path('devserver.py.lprof').remove()
    try:
        call('kernprof.py -l qexec/web/devserver.py --hostsettings', shell=True)
    except KeyboardInterrupt:
        pass
    time.sleep(1) # wait for disk io
    call('python -m line_profiler devserver.py.lprof', shell=True)

def magic_shell():
    sys.path.append(path())
    imported_objects = {}
    for mod in find_packages():
        imported_objects[mod] = __import__(mod)

    for name, mod, defs in stuff_i_want_in_my_debug_shell:
        imported_objects[name] = __import__(mod, globals(), locals(), defs)

    return  imported_objects

@task
def shell():
    """
    Run a bpython shell with all your desired modules right at
    your fingertips.
    """
    from bpython import embed
    embed(magic_shell())

@task
def ishell():
    """
    Run a ipython shell with all your desired modules right at
    your fingertips.
    """
    from IPython.frontend.terminal.embed import InteractiveShellEmbed
    shell = InteractiveShellEmbed(user_ns=magic_shell())
    #shell.extensiosn = ['line_profiler',]
    shell()

@task
def findbugs():
    """
    Google's bug prediction algorithm. Algorithmically find
    hotspots in your code where bugs are likely to occur based on
    their git history.
    """
    call('bugspot.py zipline', shell=True)

@task
def findtodos():
    """
    Grep for TODO
    """
    call('grep TODO zipline/*/*.py -C 3 ', shell=True)

@task
def findpdb():
    """
    find references to debugger
    """
    call('grep "import pdb; pdb.set_trace()" zipline/*/*.py -C 3 ', shell=True)

@task
def guppy():
    """
    Guppy heap analyzer
    """
    pass

@task
def apidocs():
    """
    Recursively autogenerate the Sphinx autodoc for the module and
    its submodules.
    """
    call('rm docs/zipline*.rst', shell=True)
    call('sphinx-apidoc -o docs/ zipline', shell=True)
