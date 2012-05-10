"""
Small classes to assist with timezone calculations, LOGGER configuration,
and other common operations.
"""

import logging
import logging.config

#logging.config.fileConfig('logging.cfg')

def configure_logginer():
    logging.config.fileConfig('logging.cfg')
