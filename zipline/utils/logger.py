"""
Small classes to assist with timezone calculations, LOGGER configuration,
and other common operations.
"""


import logging
import logging.config
from os.path import join, abspath, dirname

def configure_logging():
    logging.config.fileConfig(
        logger_path(),
        disable_existing_loggers = False
    )

def logger_path():
    import zipline
    log_path = dirname(abspath(zipline.__file__))
    return join(log_path, 'logging.cfg')


# utility for tailing a log file.
def tail( f, window=20 ):
    """
    from
    http://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file- \
            with-python-similar-to-tail
    """
    BUFSIZ = 1024
    f.seek(0, 2)
    bytes = f.tell()
    size = window
    block = -1
    data = []
    while size > 0 and bytes > 0:
        if (bytes - BUFSIZ > 0):
            # Seek back one whole BUFSIZ
            f.seek(block*BUFSIZ, 2)
            # read BUFFER
            data.append(f.read(BUFSIZ))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            data.append(f.read(bytes))
        linesFound = data[-1].count('\n')
        size -= linesFound
        bytes -= BUFSIZ
        block -= 1
    return '\n'.join(''.join(data).splitlines()[-window:])
