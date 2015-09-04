from __future__ import print_function
import sys
import doctest
from unittest import TestCase

from zipline.lib import adjustment
from zipline.modelling import (
    engine,
    expression,
)
from zipline.utils import (
    memoize,
    test_utils,
)


class DoctestTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        import pdb
        # Workaround for the issue addressed by this (unmerged) PR to pdbpp:
        # https://bitbucket.org/antocuni/pdb/pull-request/40/fix-ensure_file_can_write_unicode/diff  # noqa
        if '_pdbpp_path_hack' in pdb.__file__:
            cls._skip = True
        else:
            cls._skip = False

    def _check_docs(self, module):
        if self._skip:
            # Printing this directly to __stdout__ so that it doesn't get
            # captured by nose.
            print("Warning: Skipping doctests for %s because "
                  "pdbpp is installed." % module.__name__, file=sys.__stdout__)
            return
        try:
            doctest.testmod(module, verbose=True, raise_on_error=True)
        except doctest.UnexpectedException as e:
            raise e.exc_info[1]

    def test_adjustment_docs(self):
        self._check_docs(adjustment)

    def test_expression_docs(self):
        self._check_docs(expression)

    def test_engine_docs(self):
        self._check_docs(engine)

    def test_memoize_docs(self):
        self._check_docs(memoize)

    def test_test_utils_docs(self):
        self._check_docs(test_utils)
