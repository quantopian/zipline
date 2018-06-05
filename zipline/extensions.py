import re


class ExtensionArgs(object):

    def __init__(self, args):
        self.extension_args = {}

        for arg in args:
            self.parse_extension_arg(arg)

        for name in sorted(self.extension_args, key=len):
            path = name.split('.')
            self.get_namespace(self, path, self.extension_args[name])

    def parse_extension_arg(self, arg):
        match = re.match(r'^(([^\d\W]\w*)(\.[^\d\W]\w*)*)=(.*)$', arg)
        if match is None:
            raise ValueError(
                "invalid extension argument %s, must be in key=value form" %
                arg)

        name = match.group(1)
        value = match.group(4)
        self.extension_args[name] = value

    def get_namespace(self, obj, path, name):
        if len(path) == 1:
            setattr(obj, path[0], name)
        else:
            if hasattr(obj, path[0]):
                if type(getattr(obj, path[0])) is str:
                    raise ValueError("Conflicting assignments at namespace"
                                     " level '%s'" % path[0])
                self.get_namespace(getattr(obj, path[0]), path[1:], name)
            else:
                a = Arg()
                setattr(obj, path[0], a)
                self.get_namespace(getattr(obj, path[0]), path[1:], name)


class Arg(object):

    pass
