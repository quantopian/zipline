from pip.req import InstallRequirement, RequirementSet, parse_requirements
from pip.basecommand import Command
from pip.exceptions import InstallationError

class UninstallCommand(Command):
    name = 'uninstall'
    usage = '%prog [OPTIONS] PACKAGE_NAMES ...'
    summary = 'Uninstall packages'

    def __init__(self):
        super(UninstallCommand, self).__init__()
        self.parser.add_option(
            '-r', '--requirement',
            dest='requirements',
            action='append',
            default=[],
            metavar='FILENAME',
            help='Uninstall all the packages listed in the given requirements file.  '
            'This option can be used multiple times.')
        self.parser.add_option(
            '-y', '--yes',
            dest='yes',
            action='store_true',
            help="Don't ask for confirmation of uninstall deletions.")

    def run(self, options, args):
        requirement_set = RequirementSet(
            build_dir=None,
            src_dir=None,
            download_dir=None)
        for name in args:
            requirement_set.add_requirement(
                InstallRequirement.from_line(name))
        for filename in options.requirements:
            for req in parse_requirements(filename, options=options):
                requirement_set.add_requirement(req)
        if not requirement_set.has_requirements:
            raise InstallationError('You must give at least one requirement '
                'to %(name)s (see "pip help %(name)s")' % dict(name=self.name))
        requirement_set.uninstall(auto_confirm=options.yes)

UninstallCommand()
