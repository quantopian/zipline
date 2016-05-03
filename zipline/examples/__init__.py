from glob import glob
from importlib import import_module
import os

for f in os.listdir(os.path.dirname(__file__)):
    if not f.endswith('.py') or f == '__init__.py':
        continue
    modname = f[:-len('.py')]
    globals()[modname] = import_module('.' + modname, package=__name__)

del f
try:
    del modname
except NameError:
    pass

del os, import_module, glob
