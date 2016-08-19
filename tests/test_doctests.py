from __future__ import print_function
import sys
import doctest
from unittest import TestCase

from zipline import testing
from zipline.lib import adjustment, normalize
from zipline.pipeline import (
    engine,
    expression,
)
from zipline.utils import (
    cache,
    data,
    functional,
    input_validation,
    memoize,
    numpy_utils,
    preprocess,
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
        cls.flags = doctest.REPORT_CDIFF | doctest.IGNORE_EXCEPTION_DETAIL

    def _check_docs(self, module):
        if self._skip:
            # Printing this directly to __stdout__ so that it doesn't get
            # captured by nose.
            print("Warning: Skipping doctests for %s because "
                  "pdbpp is installed." % module.__name__, file=sys.__stdout__)
            return
        try:
            doctest.testmod(
                module,
                verbose=True,
                raise_on_error=True,
                optionflags=self.flags,
            )
        except doctest.UnexpectedException as e:
            raise e.exc_info[1]
        except doctest.DocTestFailure as e:
            print("Got:")
            print(e.got)
            raise

    def test_adjustment_docs(self):
        self._check_docs(adjustment)

    def test_expression_docs(self):
        self._check_docs(expression)

    def test_engine_docs(self):
        self._check_docs(engine)

    def test_memoize_docs(self):
        self._check_docs(memoize)

    def test_testing_docs(self):
        self._check_docs(testing)

    def test_preprocess_docs(self):
        self._check_docs(preprocess)

    def test_input_validation_docs(self):
        self._check_docs(input_validation)

    def test_cache_docs(self):
        self._check_docs(cache)

    def test_numpy_utils_docs(self):
        self._check_docs(numpy_utils)

    def test_data_docs(self):
        self._check_docs(data)

    def test_functional_docs(self):
        self._check_docs(functional)

    def test_normalize_docs(self):
        self._check_docs(normalize)
