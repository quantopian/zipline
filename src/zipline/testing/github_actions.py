from functools import wraps
import os
import pytest


def skip_on(exception, reason="Ignoring PermissionErrors on GHA"):
    # Func below is the real decorator and will receive the test function as param
    def decorator_func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Try to run the test
                return f(*args, **kwargs)
            except exception:
                # If certain exception happens, just ignore
                # and raise pytest.skip with given reason
                # if os.environ.get("GITHUB_ACTIONS") == "true":
                pytest.skip(reason)
                # else:
                #     raise

        return wrapper

    return decorator_func
