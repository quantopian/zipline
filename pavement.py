import os, sys
import time
import re
import platform

from subprocess import call

from paver.easy import *
from paver.doctools import *
from paver.setuputils import *

from paved import *
from paved.util import *
from paved.pycheck import *

#add setuputils tasks
paver.setuputils.install_distutils_tasks()

operating_system = platform.system()

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

version='dev'
install_requires = parse_requirements('./etc/requirements.txt') + parse_requirements('./etc/requirements_sci.txt')
tests_require = install_requires + parse_requirements('./etc/requirements_dev.txt')

options(
    sphinx=Bunch(
        builddir="_build",
        sourcedir=""
    ),
    setup = Bunch(name='zipline',
          version              = version,
          classifiers          = [],
          packages             = find_packages(),
          install_requires     = install_requires,
          tests_require        = tests_require,
          test_suite           = 'nose.collector',
          include_package_data = True,
          zip_safe             = False,
    ),
)

options.paved.clean.patterns.extend([
    #'*.swp', # vim related
    #'*.swo', # vim related
    'nosetests.xml',
    '.coverage',
    ',coverage',
    '*.lprof',
    '*.prof',
])

# Because I'm lazy
stuff_i_want_in_my_debug_shell = [
    ('qutil', 'zipline.util', []),
    ('zmq', 'zmq', []),
]

# ======
# Tasks
# ======

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
