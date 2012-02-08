from pip.locations import build_prefix, src_prefix
from pip.util import display_path, backup_dir
from pip.log import logger
from pip.exceptions import InstallationError
from pip.commands.install import InstallCommand


class BundleCommand(InstallCommand):
    name = 'bundle'
    usage = '%prog [OPTIONS] BUNDLE_NAME.pybundle PACKAGE_NAMES...'
    summary = 'Create pybundles (archives containing multiple packages)'
    bundle = True

    def __init__(self):
        super(BundleCommand, self).__init__()

    def run(self, options, args):
        if not args:
            raise InstallationError('You must give a bundle filename')
        if not options.build_dir:
            options.build_dir = backup_dir(build_prefix, '-bundle')
        if not options.src_dir:
            options.src_dir = backup_dir(src_prefix, '-bundle')
        # We have to get everything when creating a bundle:
        options.ignore_installed = True
        logger.notify('Putting temporary build files in %s and source/develop files in %s'
                      % (display_path(options.build_dir), display_path(options.src_dir)))
        self.bundle_filename = args.pop(0)
        requirement_set = super(BundleCommand, self).run(options, args)
        return requirement_set


BundleCommand()
