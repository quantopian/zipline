def s(word, seq, suffix="s"):
    """Adds a suffix to ``word`` if some sequence has anything other than
    exactly one element.

    Parameters
    ----------
    word : str
        The string to add the suffix to.
    seq : sequence
        The sequence to check the length of.
    suffix : str, optional.
        The suffix to add to ``word``

    Returns
    -------
    maybe_plural : str
        ``word`` with ``suffix`` added if ``len(seq) != 1``.
    """
    if len(seq) == 1:
        return word

    return word + suffix


def plural(singular, plural, seq):
    """Selects a singular or plural word based on the length of a sequence.

    Parameters
    ----------
    singlular : str
        The string to use when ``len(seq) == 1``.
    plural : str
        The string to use when ``len(seq) != 1``.
    seq : sequence
        The sequence to check the length of.

    Returns
    -------
    maybe_plural : str
        Either ``singlular`` or ``plural``.
    """
    if len(seq) == 1:
        return singular

    return plural


def bulleted_list(items, indent=0, bullet_type="-"):
    """Format a bulleted list of values.

    Parameters
    ----------
    items : sequence
        The items to make a list.
    indent : int, optional
        The number of spaces to add before each bullet.
    bullet_type : str, optional
        The bullet type to use.

    Returns
    -------
    formatted_list : str
        The formatted list as a single string.
    """
    format_string = " " * indent + bullet_type + " {}"
    return "\n".join(map(format_string.format, items))
