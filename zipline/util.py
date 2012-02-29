"""
Small classes to assist with timezone calculations, LOGGER configuration,
and other common operations.
"""

import datetime
import pytz
import logging
import logging.handlers

LOGGER = logging.getLogger('ZiplineLogger')

def configure_logging(loglevel=logging.DEBUG):
    """
    Configures zipline.util.LOGGER to write a rotating file
    (10M per file, 5 files) to `` /var/log/zipline.log ``.
    """
    LOGGER.setLevel(loglevel)
    handler = logging.handlers.RotatingFileHandler(
        "/var/log/zipline/{lfn}.log".format(lfn="zipline"),
        maxBytes=10*1024*1024, backupCount=5
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s %(funcName)s - %(message)s",
        "%Y-%m-%d %H:%M:%S %Z")
    )
    LOGGER.addHandler(handler)
    LOGGER.info("logging started...")
