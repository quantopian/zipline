"""
Shared docstrings for parameters that should be documented identically
across different functions.
"""
import re
from six import iteritems
from textwrap import dedent

PIPELINE_DOWNSAMPLING_FREQUENCY_DOC = dedent(
    """\
    frequency : {'year_start', 'quarter_start', 'month_start', 'week_start'}
        A string indicating desired sampling dates:

        * 'year_start'    -> first trading day of each year
        * 'quarter_start' -> first trading day of January, April, July, October
        * 'month_start'   -> first trading day of each month
        * 'week_start'    -> first trading_day of each week
    """
)

PIPELINE_ALIAS_NAME_DOC = dedent(
    """\
    name : str
        The name to alias this term as.
    """,
)


def pad_lines_after_first(prefix, s):
    """Apply a prefix to each line in s after the first."""
    return ('\n' + prefix).join(s.splitlines())


def format_docstring(owner_name, docstring, formatters):
    """
    Template ``formatters`` into ``docstring``.

    Parameters
    ----------
    owner_name : str
        The name of the function or class whose docstring is being templated.
        Only used for error messages.
    docstring : str
        The docstring to template.
    formatters : dict[str -> str]
        Parameters for a a str.format() call on ``docstring``.

        Multi-line values in ``formatters`` will have leading whitespace padded
        to match the leading whitespace of the substitution string.
    """
    # Build a dict of parameters to a vanilla format() call by searching for
    # each entry in **formatters and applying any leading whitespace to each
    # line in the desired substitution.
    format_params = {}
    for target, doc_for_target in iteritems(formatters):
        # Search for '{name}', with optional leading whitespace.
        regex = re.compile('^(\s*)' + '({' + target + '})$', re.MULTILINE)
        matches = regex.findall(docstring)
        if not matches:
            raise ValueError(
                "Couldn't find template for parameter {!r} in docstring "
                "for {}."
                "\nParameter name must be alone on a line surrounded by "
                "braces.".format(target, owner_name),
            )
        elif len(matches) > 1:
            raise ValueError(
                "Couldn't found multiple templates for parameter {!r}"
                "in docstring for {}."
                "\nParameter should only appear once.".format(
                    target, owner_name
                )
            )

        (leading_whitespace, _) = matches[0]
        format_params[target] = pad_lines_after_first(
            leading_whitespace,
            doc_for_target,
        )

    return docstring.format(**format_params)


def templated_docstring(**docs):
    """
    Decorator allowing the use of templated docstrings.

    Usage
    -----
    >>> @templated_docstring(foo='bar')
    ... def my_func(self, foo):
    ...     '''{foo}'''
    ...
    >>> my_func.__doc__
    'bar'
    """
    def decorator(f):
        f.__doc__ = format_docstring(f.__name__, f.__doc__, docs)
        return f
    return decorator
