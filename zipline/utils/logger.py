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
