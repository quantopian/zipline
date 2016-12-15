"""
Utilities for validating inputs to user-facing API functions.
"""
from textwrap import dedent
from types import CodeType
from functools import wraps
from inspect import getargspec
from uuid import uuid4

from toolz.curried.operator import getitem
from six import viewkeys, exec_, PY3


_code_argorder = (
    ('co_argcount', 'co_kwonlyargcount') if PY3 else ('co_argcount',)
) + (
    'co_nlocals',
    'co_stacksize',
    'co_flags',
    'co_code',
    'co_consts',
    'co_names',
    'co_varnames',
    'co_filename',
    'co_name',
    'co_firstlineno',
    'co_lnotab',
    'co_freevars',
    'co_cellvars',
)

NO_DEFAULT = object()


def preprocess(*_unused, **processors):
    """
    Decorator that applies pre-processors to the arguments of a function before
    calling the function.

    Parameters
    ----------
    **processors : dict
        Map from argument name -> processor function.

        A processor function takes three arguments: (func, argname, argvalue).

        `func` is the the function for which we're processing args.
        `argname` is the name of the argument we're processing.
        `argvalue` is the value of the argument we're processing.

    Usage
    -----
    >>> def _ensure_tuple(func, argname, arg):
    ...     if isinstance(arg, tuple):
    ...         return argvalue
    ...     try:
    ...         return tuple(arg)
    ...     except TypeError:
    ...         raise TypeError(
    ...             "%s() expected argument '%s' to"
    ...             " be iterable, but got %s instead." % (
    ...                 func.__name__, argname, arg,
    ...             )
    ...         )
    ...
    >>> @preprocess(arg=_ensure_tuple)
    ... def foo(arg):
    ...     return arg
    ...
    >>> foo([1, 2, 3])
    (1, 2, 3)
    >>> foo("a")
    ('a',)
    >>> foo(2)
    Traceback (most recent call last):
        ...
    TypeError: foo() expected argument 'arg' to be iterable, but got 2 instead.
    """
    if _unused:
        raise TypeError("preprocess() doesn't accept positional arguments")

    def _decorator(f):
        args, varargs, varkw, defaults = argspec = getargspec(f)
        if defaults is None:
            defaults = ()
        no_defaults = (NO_DEFAULT,) * (len(args) - len(defaults))
        args_defaults = list(zip(args, no_defaults + defaults))
        if varargs:
            args_defaults.append((varargs, NO_DEFAULT))
        if varkw:
            args_defaults.append((varkw, NO_DEFAULT))

        argset = set(args) | {varargs, varkw} - {None}

        # Arguments can be declared as tuples in Python 2.
        if not all(isinstance(arg, str) for arg in args):
            raise TypeError(
                "Can't validate functions using tuple unpacking: %s" %
                (argspec,)
            )

        # Ensure that all processors map to valid names.
        bad_names = viewkeys(processors) - argset
        if bad_names:
            raise TypeError(
                "Got processors for unknown arguments: %s." % bad_names
            )

        return _build_preprocessed_function(
            f, processors, args_defaults, varargs, varkw,
        )
    return _decorator


def call(f):
    """
    Wrap a function in a processor that calls `f` on the argument before
    passing it along.

    Useful for creating simple arguments to the `@preprocess` decorator.

    Parameters
    ----------
    f : function
        Function accepting a single argument and returning a replacement.

    Usage
    -----
    >>> @preprocess(x=call(lambda x: x + 1))
    ... def foo(x):
    ...     return x
    ...
    >>> foo(1)
    2
    """
    @wraps(f)
    def processor(func, argname, arg):
        return f(arg)
    return processor


def _build_preprocessed_function(func,
                                 processors,
                                 args_defaults,
                                 varargs,
                                 varkw):
    """
    Build a preprocessed function with the same signature as `func`.

    Uses `exec` internally to build a function that actually has the same
    signature as `func.
    """
    format_kwargs = {'func_name': func.__name__}

    def mangle(name):
        return 'a' + uuid4().hex + name

    format_kwargs['mangled_func'] = mangled_funcname = mangle(func.__name__)

    def make_processor_assignment(arg, processor_name):
        template = "{arg} = {processor}({func}, '{arg}', {arg})"
        return template.format(
            arg=arg,
            processor=processor_name,
            func=mangled_funcname,
        )

    exec_globals = {mangled_funcname: func, 'wraps': wraps}
    defaults_seen = 0
    default_name_template = 'a' + uuid4().hex + '_%d'
    signature = []
    call_args = []
    assignments = []
    star_map = {
        varargs: '*',
        varkw: '**',
    }

    def name_as_arg(arg):
        return star_map.get(arg, '') + arg

    for arg, default in args_defaults:
        if default is NO_DEFAULT:
            signature.append(name_as_arg(arg))
        else:
            default_name = default_name_template % defaults_seen
            exec_globals[default_name] = default
            signature.append('='.join([name_as_arg(arg), default_name]))
            defaults_seen += 1

        if arg in processors:
            procname = mangle('_processor_' + arg)
            exec_globals[procname] = processors[arg]
            assignments.append(make_processor_assignment(arg, procname))

        call_args.append(name_as_arg(arg))

    exec_str = dedent(
        """\
        @wraps({wrapped_funcname})
        def {func_name}({signature}):
            {assignments}
            return {wrapped_funcname}({call_args})
        """
    ).format(
        func_name=func.__name__,
        signature=', '.join(signature),
        assignments='\n    '.join(assignments),
        wrapped_funcname=mangled_funcname,
        call_args=', '.join(call_args),
    )
    compiled = compile(
        exec_str,
        func.__code__.co_filename,
        mode='exec',
    )

    exec_locals = {}
    exec_(compiled, exec_globals, exec_locals)
    new_func = exec_locals[func.__name__]

    code = new_func.__code__
    args = {
        attr: getattr(code, attr)
        for attr in dir(code)
        if attr.startswith('co_')
    }
    # Copy the firstlineno out of the underlying function so that exceptions
    # get raised with the correct traceback.
    # This also makes dynamic source inspection (like IPython `??` operator)
    # work as intended.
    try:
        # Try to get the pycode object from the underlying function.
        original_code = func.__code__
    except AttributeError:
        try:
            # The underlying callable was not a function, try to grab the
            # `__func__.__code__` which exists on method objects.
            original_code = func.__func__.__code__
        except AttributeError:
            # The underlying callable does not have a `__code__`. There is
            # nothing for us to correct.
            return new_func

    args['co_firstlineno'] = original_code.co_firstlineno
    new_func.__code__ = CodeType(*map(getitem(args), _code_argorder))
    return new_func
