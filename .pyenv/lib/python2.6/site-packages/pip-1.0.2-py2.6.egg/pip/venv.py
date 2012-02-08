"""Tools for working with virtualenv environments"""

import os
import sys
import subprocess
from pip.exceptions import BadCommand
from pip.log import logger


def restart_in_venv(venv, base, site_packages, args):
    """
    Restart this script using the interpreter in the given virtual environment
    """
    if base and not os.path.isabs(venv) and not venv.startswith('~'):
        base = os.path.expanduser(base)
        # ensure we have an abs basepath at this point:
        #    a relative one makes no sense (or does it?)
        if os.path.isabs(base):
            venv = os.path.join(base, venv)

    if venv.startswith('~'):
        venv = os.path.expanduser(venv)

    if not os.path.exists(venv):
        try:
            import virtualenv
        except ImportError:
            print('The virtual environment does not exist: %s' % venv)
            print('and virtualenv is not installed, so a new environment cannot be created')
            sys.exit(3)
        print('Creating new virtualenv environment in %s' % venv)
        virtualenv.logger = logger
        logger.indent += 2
        virtualenv.create_environment(venv, site_packages=site_packages)
    if sys.platform == 'win32':
        python = os.path.join(venv, 'Scripts', 'python.exe')
        # check for bin directory which is used in buildouts
        if not os.path.exists(python):
            python = os.path.join(venv, 'bin', 'python.exe')
    else:
        python = os.path.join(venv, 'bin', 'python')
    if not os.path.exists(python):
        python = venv
    if not os.path.exists(python):
        raise BadCommand('Cannot find virtual environment interpreter at %s' % python)
    base = os.path.dirname(os.path.dirname(python))
    file = os.path.join(os.path.dirname(__file__), 'runner.py')
    if file.endswith('.pyc'):
        file = file[:-1]
    proc = subprocess.Popen(
        [python, file] + args + [base, '___VENV_RESTART___'])
    proc.wait()
    sys.exit(proc.returncode)
