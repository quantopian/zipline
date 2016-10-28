import numpy as np
import pandas as pd

from zipline.testing import parameter_space, ZiplineTestCase
import zipline.utils.input_validation as iv


class InputValidationTestCase(ZiplineTestCase):

    @parameter_space(extra_key=['e', 'z'])
    def test_check_series_nonnull(self, extra_key):
        s = pd.Series(
            {'a': 1.0, 'b': np.nan, 'c': 3, 'd': np.nan, extra_key: np.nan}
        )

        msg_template = "Error Message: {keys}"
        expected = "Error Message: ['b', 'd', {k!r}]".format(k=extra_key)

        with self.assertRaises(ValueError) as e:
            iv.check_series_notnull(s, msg_template)

        self.assertEqual(expected, str(e.exception))
