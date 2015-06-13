import doctest
from unittest import TestCase

from zipline.data import adjustment
from zipline.modelling import expression


class DoctestTestCase(TestCase):

    def _check_docs(self, module):
        try:
            results = doctest.testmod(
                module,
                verbose=True,
                raise_on_error=True,
            )
            print results
        except doctest.UnexpectedException as e:
            raise e.exc_info[1]

    def test_adjustment(self):
        self._check_docs(adjustment)

    def test_modelling(self):
        self._check_docs(expression)
