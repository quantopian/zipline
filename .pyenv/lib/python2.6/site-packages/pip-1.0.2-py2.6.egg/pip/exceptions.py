"""Exceptions used throughout package"""


class InstallationError(Exception):
    """General exception during installation"""


class UninstallationError(Exception):
    """General exception during uninstallation"""


class DistributionNotFound(InstallationError):
    """Raised when a distribution cannot be found to satisfy a requirement"""


class BadCommand(Exception):
    """Raised when virtualenv or a command is not found"""
