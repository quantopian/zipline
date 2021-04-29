from zipline.utils.sharedoc import copydoc


class TestSharedoc:
    def test_copydoc(self):
        def original_docstring_function():
            """
            My docstring brings the boys to the yard.
            """
            pass

        @copydoc(original_docstring_function)
        def copied_docstring_function():
            pass

        assert original_docstring_function.__doc__ == copied_docstring_function.__doc__
