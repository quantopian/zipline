def stringify_args(*args, **kwargs):
    """Define a unique string for any set of args."""
    arg_string = '_'.join([str(arg) for arg in args])
    kwarg_string = '_'.join([str(key) + '=' + str(value) for key, value in kwargs])
    combined = ':'.join([arg_string, kwarg_string])
