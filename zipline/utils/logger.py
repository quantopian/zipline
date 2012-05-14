"""
Small classes to assist with timezone calculations, LOGGER configuration,
and other common operations.
"""

import logging
import logging.config

def configure_logging():
    logging.config.fileConfig('logging.cfg')
