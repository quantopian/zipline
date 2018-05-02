from itertools import product
from operator import eq, ne
import warnings

import numpy as np
from toolz import take

from zipline.lib.labelarray import LabelArray
from zipline.testing import check_arrays, parameter_space, ZiplineTestCase
from zipline.testing.predicates import assert_equal
from zipline.utils.compat import unicode


def rotN(l, N):
    """
    Rotate a list of elements.

    Pulls N elements off the end of the list and appends them to the front.

    >>> rotN(['a', 'b', 'c', 'd'], 2)
    ['c', 'd', 'a', 'b']
    >>> rotN(['a', 'b', 'c', 'd'], 3)
    ['d', 'a', 'b', 'c']
    """
    assert len(l) >= N, "Can't rotate list by longer than its length."
    return l[N:] + l[:N]


def all_ufuncs():
    ufunc_type = type(np.isnan)
    return (f for f in vars(np).values() if isinstance(f, ufunc_type))


class LabelArrayTestCase(ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(LabelArrayTestCase, cls).init_class_fixtures()

        cls.rowvalues = row = ['', 'a', 'b', 'ab', 'a', '', 'b', 'ab', 'z']
        cls.strs = np.array([rotN(row, i) for i in range(3)], dtype=object)

    def test_fail_on_direct_construction(self):
        # See http://docs.scipy.org/doc/numpy-1.10.0/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray  # noqa

        with self.assertRaises(TypeError) as e:
            np.ndarray.__new__(LabelArray, (5, 5))

        self.assertEqual(
            str(e.exception),
            "Direct construction of LabelArrays is not supported."
        )

    @parameter_space(
        __fail_fast=True,
        compval=['', 'a', 'z', 'not in the array'],
        shape=[(27,), (3, 9), (3, 3, 3)],
        array_astype=(bytes, unicode, object),
        missing_value=('', 'a', 'not in the array', None),
    )
    def test_compare_to_str(self,
                            compval,
                            shape,
                            array_astype,
                            missing_value):

        strs = self.strs.reshape(shape).astype(array_astype)
        if missing_value is None:
            # As of numpy 1.9.2, object array != None returns just False
            # instead of an array, with a deprecation warning saying the
            # behavior will change in the future.  Work around that by just
            # using the ufunc.
            notmissing = np.not_equal(strs, missing_value)
        else:
            if not isinstance(missing_value, array_astype):
                missing_value = array_astype(missing_value, 'utf-8')
            notmissing = (strs != missing_value)

        arr = LabelArray(strs, missing_value=missing_value)

        if not isinstance(compval, array_astype):
            compval = array_astype(compval, 'utf-8')

        # arr.missing_value should behave like NaN.
        check_arrays(
            arr == compval,
            (strs == compval) & notmissing,
        )
        check_arrays(
            arr != compval,
            (strs != compval) & notmissing,
        )

        np_startswith = np.vectorize(lambda elem: elem.startswith(compval))
        check_arrays(
            arr.startswith(compval),
            np_startswith(strs) & notmissing,
        )

        np_endswith = np.vectorize(lambda elem: elem.endswith(compval))
        check_arrays(
            arr.endswith(compval),
            np_endswith(strs) & notmissing,
        )

        np_contains = np.vectorize(lambda elem: compval in elem)
        check_arrays(
            arr.has_substring(compval),
            np_contains(strs) & notmissing,
        )

    @parameter_space(
        __fail_fast=True,
        f=[
            lambda s: str(len(s)),
            lambda s: s[0],
            lambda s: ''.join(reversed(s)),
            lambda s: '',
        ]
    )
    def test_map(self, f):
        data = np.array(
            [['E', 'GHIJ', 'HIJKLMNOP', 'DEFGHIJ'],
             ['CDE', 'ABCDEFGHIJKLMNOPQ', 'DEFGHIJKLMNOPQRS', 'ABCDEFGHIJK'],
             ['DEFGHIJKLMNOPQR', 'DEFGHI', 'DEFGHIJ', 'FGHIJK'],
             ['EFGHIJKLM', 'EFGHIJKLMNOPQRS', 'ABCDEFGHI', 'DEFGHIJ']],
            dtype=object,
        )
        la = LabelArray(data, missing_value=None)

        numpy_transformed = np.vectorize(f)(data)
        la_transformed = la.map(f).as_string_array()

        assert_equal(numpy_transformed, la_transformed)

    @parameter_space(missing=['A', None])
    def test_map_ignores_missing_value(self, missing):
        data = np.array([missing, 'B', 'C'], dtype=object)
        la = LabelArray(data, missing_value=missing)

        def increment_char(c):
            return chr(ord(c) + 1)

        result = la.map(increment_char)
        expected = LabelArray([missing, 'C', 'D'], missing_value=missing)
        assert_equal(result.as_string_array(), expected.as_string_array())

    @parameter_space(
        __fail_fast=True,
        f=[
            lambda s: 0,
            lambda s: 0.0,
            lambda s: object(),
        ]
    )
    def test_map_requires_f_to_return_a_string_or_none(self, f):
        la = LabelArray(self.strs, missing_value=None)

        with self.assertRaises(TypeError):
            la.map(f)

    def test_map_can_only_return_none_if_missing_value_is_none(self):

        # Should work.
        la = LabelArray(self.strs, missing_value=None)
        result = la.map(lambda x: None)

        check_arrays(
            result,
            LabelArray(np.full_like(self.strs, None), missing_value=None),
        )

        la = LabelArray(self.strs, missing_value="__MISSING__")
        with self.assertRaises(TypeError):
            la.map(lambda x: None)

    @parameter_space(
        __fail_fast=True,
        missing_value=('', 'a', 'not in the array', None),
    )
    def test_compare_to_str_array(self, missing_value):
        strs = self.strs
        shape = strs.shape
        arr = LabelArray(strs, missing_value=missing_value)

        if missing_value is None:
            # As of numpy 1.9.2, object array != None returns just False
            # instead of an array, with a deprecation warning saying the
            # behavior will change in the future.  Work around that by just
            # using the ufunc.
            notmissing = np.not_equal(strs, missing_value)
        else:
            notmissing = (strs != missing_value)

        check_arrays(arr.not_missing(), notmissing)
        check_arrays(arr.is_missing(), ~notmissing)

        # The arrays are equal everywhere, but comparisons against the
        # missing_value should always produce False
        check_arrays(strs == arr, notmissing)
        check_arrays(strs != arr, np.zeros_like(strs, dtype=bool))

        def broadcastable_row(value, dtype):
            return np.full((shape[0], 1), value, dtype=strs.dtype)

        def broadcastable_col(value, dtype):
            return np.full((1, shape[1]), value, dtype=strs.dtype)

        # Test comparison between arr and a like-shap 2D array, a column
        # vector, and a row vector.
        for comparator, dtype, value in product((eq, ne),
                                                (bytes, unicode, object),
                                                set(self.rowvalues)):
            check_arrays(
                comparator(arr, np.full_like(strs, value)),
                comparator(strs, value) & notmissing,
            )
            check_arrays(
                comparator(arr, broadcastable_row(value, dtype=dtype)),
                comparator(strs, value) & notmissing,
            )
            check_arrays(
                comparator(arr, broadcastable_col(value, dtype=dtype)),
                comparator(strs, value) & notmissing,
            )

    @parameter_space(
        __fail_fast=True,
        slice_=[
            0, 1, -1,
            slice(None),
            slice(0, 0),
            slice(0, 3),
            slice(1, 4),
            slice(0),
            slice(None, 1),
            slice(0, 4, 2),
            (slice(None), 1),
            (slice(None), slice(None)),
            (slice(None), slice(1, 2)),
        ]
    )
    def test_slicing_preserves_attributes(self, slice_):
        arr = LabelArray(self.strs.reshape((9, 3)), missing_value='')
        sliced = arr[slice_]
        self.assertIsInstance(sliced, LabelArray)
        self.assertIs(sliced.categories, arr.categories)
        self.assertIs(sliced.reverse_categories, arr.reverse_categories)
        self.assertIs(sliced.missing_value, arr.missing_value)

    def test_infer_categories(self):
        """
        Test that categories are inferred in sorted order if they're not
        explicitly passed.
        """
        arr1d = LabelArray(self.strs, missing_value='')
        codes1d = arr1d.as_int_array()
        self.assertEqual(arr1d.shape, self.strs.shape)
        self.assertEqual(arr1d.shape, codes1d.shape)

        categories = arr1d.categories
        unique_rowvalues = set(self.rowvalues)

        # There should be an entry in categories for each unique row value, and
        # each integer stored in the data array should be an index into
        # categories.
        self.assertEqual(list(categories), sorted(set(self.rowvalues)))
        self.assertEqual(
            set(codes1d.ravel()),
            set(range(len(unique_rowvalues)))
        )
        for idx, value in enumerate(arr1d.categories):
            check_arrays(
                self.strs == value,
                arr1d.as_int_array() == idx,
            )

        # It should be equivalent to pass the same set of categories manually.
        arr1d_explicit_categories = LabelArray(
            self.strs,
            missing_value='',
            categories=arr1d.categories,
        )
        check_arrays(arr1d, arr1d_explicit_categories)

        for shape in (9, 3), (3, 9), (3, 3, 3):
            strs2d = self.strs.reshape(shape)
            arr2d = LabelArray(strs2d, missing_value='')
            codes2d = arr2d.as_int_array()

            self.assertEqual(arr2d.shape, shape)
            check_arrays(arr2d.categories, categories)

            for idx, value in enumerate(arr2d.categories):
                check_arrays(strs2d == value, codes2d == idx)

    def test_reject_ufuncs(self):
        """
        The internal values of a LabelArray should be opaque to numpy ufuncs.

        Test that all unfuncs fail.
        """
        l = LabelArray(self.strs, '')
        ints = np.arange(len(l))

        with warnings.catch_warnings():
            # Some ufuncs return NotImplemented, but warn that they will fail
            # in the future.  Both outcomes are fine, so ignore the warnings.
            warnings.filterwarnings(
                'ignore',
                message="unorderable dtypes.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                'ignore',
                message="elementwise comparison failed.*",
                category=FutureWarning,
            )
            for func in all_ufuncs():
                # Different ufuncs vary between returning NotImplemented and
                # raising a TypeError when provided with unknown dtypes.
                # This is a bit unfortunate, but still better than silently
                # accepting an int array.
                try:
                    if func.nin == 1:
                        ret = func(l)
                    elif func.nin == 2:
                        ret = func(l, ints)
                    else:
                        self.fail("Who added a ternary ufunc !?!")
                except TypeError:
                    pass
                else:
                    self.assertIs(ret, NotImplemented)

    @parameter_space(
        __fail_fast=True,
        val=['', 'a', 'not in the array', None],
        missing_value=['', 'a', 'not in the array', None],
    )
    def test_setitem_scalar(self, val, missing_value):
        arr = LabelArray(self.strs, missing_value=missing_value)

        if not arr.has_label(val):
            self.assertTrue(
                (val == 'not in the array')
                or (val is None and missing_value is not None)
            )
            for slicer in [(0, 0), (0, 1), 1]:
                with self.assertRaises(ValueError):
                    arr[slicer] = val
            return

        arr[0, 0] = val
        self.assertEqual(arr[0, 0], val)

        arr[0, 1] = val
        self.assertEqual(arr[0, 1], val)

        arr[1] = val
        if val == missing_value:
            self.assertTrue(arr.is_missing()[1].all())
        else:
            self.assertTrue((arr[1] == val).all())
            self.assertTrue((arr[1].as_string_array() == val).all())

        arr[:, -1] = val
        if val == missing_value:
            self.assertTrue(arr.is_missing()[:, -1].all())
        else:
            self.assertTrue((arr[:, -1] == val).all())
            self.assertTrue((arr[:, -1].as_string_array() == val).all())

        arr[:] = val
        if val == missing_value:
            self.assertTrue(arr.is_missing().all())
        else:
            self.assertFalse(arr.is_missing().any())
            self.assertTrue((arr == val).all())

    def test_setitem_array(self):
        arr = LabelArray(self.strs, missing_value=None)
        orig_arr = arr.copy()

        # Write a row.
        self.assertFalse(
            (arr[0] == arr[1]).all(),
            "This test doesn't test anything because rows 0"
            " and 1 are already equal!"
        )
        arr[0] = arr[1]
        for i in range(arr.shape[1]):
            self.assertEqual(arr[0, i], arr[1, i])

        # Write a column.
        self.assertFalse(
            (arr[:, 0] == arr[:, 1]).all(),
            "This test doesn't test anything because columns 0"
            " and 1 are already equal!"
        )
        arr[:, 0] = arr[:, 1]
        for i in range(arr.shape[0]):
            self.assertEqual(arr[i, 0], arr[i, 1])

        # Write the whole array.
        arr[:] = orig_arr
        check_arrays(arr, orig_arr)

    @staticmethod
    def check_roundtrip(arr):
        assert_equal(
            arr.as_string_array(),
            LabelArray(
                arr.as_string_array(),
                arr.missing_value,
            ).as_string_array(),
        )

    @staticmethod
    def create_categories(width, plus_one):
        length = int(width / 8) + plus_one
        return [
            ''.join(cs)
            for cs in take(
                2 ** width + plus_one,
                product([chr(c) for c in range(256)], repeat=length),
            )
        ]

    def test_narrow_code_storage(self):
        create_categories = self.create_categories
        check_roundtrip = self.check_roundtrip

        # uint8
        categories = create_categories(8, plus_one=False)
        arr = LabelArray(
            categories,
            missing_value=categories[0],
            categories=categories,
        )
        self.assertEqual(arr.itemsize, 1)
        check_roundtrip(arr)

        # uint8 inference
        arr = LabelArray(categories, missing_value=categories[0])
        self.assertEqual(arr.itemsize, 1)
        check_roundtrip(arr)

        # just over uint8
        categories = create_categories(8, plus_one=True)
        arr = LabelArray(
            categories,
            missing_value=categories[0],
            categories=categories,
        )
        self.assertEqual(arr.itemsize, 2)
        check_roundtrip(arr)

        # fits in uint16
        categories = create_categories(16, plus_one=False)
        arr = LabelArray(
            categories,
            missing_value=categories[0],
            categories=categories,
        )
        self.assertEqual(arr.itemsize, 2)
        check_roundtrip(arr)

        # uint16 inference
        arr = LabelArray(categories, missing_value=categories[0])
        self.assertEqual(arr.itemsize, 2)
        check_roundtrip(arr)

        # just over uint16
        categories = create_categories(16, plus_one=True)
        arr = LabelArray(
            categories,
            missing_value=categories[0],
            categories=categories,
        )
        self.assertEqual(arr.itemsize, 4)
        check_roundtrip(arr)

        # uint32 inference
        arr = LabelArray(categories, missing_value=categories[0])
        self.assertEqual(arr.itemsize, 4)
        check_roundtrip(arr)

        # NOTE: we could do this for 32 and 64; however, no one has enough RAM
        # or time for that.

    def test_known_categories_without_missing_at_boundary(self):
        # This tests the case where we have exactly 256 unique categories but
        # we didn't include the missing value in the categories.
        categories = self.create_categories(8, plus_one=False)

        arr = LabelArray(
            categories,
            None,
            categories=categories,
        )
        self.check_roundtrip(arr)
        # the missing value pushes us into 2 byte storage
        self.assertEqual(arr.itemsize, 2)

    def test_narrow_condense_back_to_valid_size(self):
        categories = ['a'] * (2 ** 8 + 1)
        arr = LabelArray(categories, missing_value=categories[0])
        assert_equal(arr.itemsize, 1)
        self.check_roundtrip(arr)

        # longer than int16 but still fits when deduped
        categories = self.create_categories(16, plus_one=False)
        categories.append(categories[0])
        arr = LabelArray(categories, missing_value=categories[0])
        assert_equal(arr.itemsize, 2)
        self.check_roundtrip(arr)

    def test_map_shrinks_code_storage_if_possible(self):
        arr = LabelArray(
            # Drop the last value so we fit in a uint16 with None as a missing
            # value.
            self.create_categories(16, plus_one=False)[:-1],
            missing_value=None,
        )

        self.assertEqual(arr.itemsize, 2)

        def either_A_or_B(s):
            return ('A', 'B')[sum(ord(c) for c in s) % 2]

        result = arr.map(either_A_or_B)

        self.assertEqual(set(result.categories), {'A', 'B', None})
        self.assertEqual(result.itemsize, 1)

        assert_equal(
            np.vectorize(either_A_or_B)(arr.as_string_array()),
            result.as_string_array(),
        )

    def test_map_never_increases_code_storage_size(self):
        # This tests a pathological case where a user maps an impure function
        # that returns a different label on every invocation, which in a naive
        # implementation could cause us to need to **increase** the size of our
        # codes after a map.
        #
        # This doesn't happen, however, because we guarantee that the user's
        # mapping function will be called on each unique category exactly once,
        # which means we can never increase the number of categories in the
        # LabelArray after mapping.

        # Using all but one of the categories so that we still fit in a uint8
        # with an extra category for None as a missing value.
        categories = self.create_categories(8, plus_one=False)[:-1]

        larger_categories = self.create_categories(16, plus_one=False)

        # Double the length of the categories so that we have to increase the
        # required size after our map.
        categories_twice = categories + categories

        arr = LabelArray(categories_twice, missing_value=None)
        assert_equal(arr.itemsize, 1)

        gen_unique_categories = iter(larger_categories)

        def new_string_every_time(c):
            # Return a new unique category every time so that every result is
            # different.
            return next(gen_unique_categories)

        result = arr.map(new_string_every_time)

        # Result should still be of size 1.
        assert_equal(result.itemsize, 1)

        # Result should be the first `len(categories)` entries from the larger
        # categories, repeated twice.
        expected = LabelArray(
            larger_categories[:len(categories)] * 2,
            missing_value=None,
        )
        assert_equal(result.as_string_array(), expected.as_string_array())

    def manual_narrow_condense_back_to_valid_size_slow(self):
        """This test is really slow so we don't want it run by default.
        """
        # tests that we don't try to create an 'int24' (which is meaningless)
        categories = self.create_categories(24, plus_one=False)
        categories.append(categories[0])
        arr = LabelArray(categories, missing_value=categories[0])
        assert_equal(arr.itemsize, 4)
        self.check_roundtrip(arr)

    def test_copy_categories_list(self):
        """regression test for #1927
        """
        categories = ['a', 'b', 'c']

        LabelArray(
            [None, 'a', 'b', 'c'],
            missing_value=None,
            categories=categories,
        )

        # before #1927 we didn't take a copy and would insert the missing value
        # (None) into the list
        assert_equal(categories, ['a', 'b', 'c'])
