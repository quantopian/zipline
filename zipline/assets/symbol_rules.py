import re


_symbol_delimiter_regex = re.compile(r'[./\-_]')


def split_nasdaq(symbol):
    sym = re.replace(_symbol_delimiter_regex, '', symbol)
    return sym[:4], sym[4:]


def split_nyse(symbol):
    return re.split(_symbol_delimiter_regex, symbol, maxsplit=1)
