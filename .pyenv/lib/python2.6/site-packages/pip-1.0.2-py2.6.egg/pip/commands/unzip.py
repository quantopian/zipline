from pip.commands.zip import ZipCommand


class UnzipCommand(ZipCommand):
    name = 'unzip'
    summary = 'Unzip individual packages'


UnzipCommand()
