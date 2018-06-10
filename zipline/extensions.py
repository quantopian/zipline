import re


def create_args(args, root):

    extension_args = {}

    for arg in args:
        parse_extension_arg(arg, extension_args)

    for name in sorted(extension_args, key=len):
        path = name.split('.')
        get_namespace(root, path, extension_args[name])


def parse_extension_arg(arg, arg_dict):
    match = re.match(r'^(([^\d\W]\w*)(\.[^\d\W]\w*)*)=(.*)$', arg)
    if match is None:
        raise ValueError(
            "invalid extension argument %s, must be in key=value form" %
            arg)

    name = match.group(1)
    value = match.group(4)
    arg_dict[name] = value


def get_namespace(obj, path, name):
    if len(path) == 1:
        setattr(obj, path[0], name)
    else:
        if hasattr(obj, path[0]):
            if type(getattr(obj, path[0])) is str:
                raise ValueError("Conflicting assignments at namespace"
                                 " level '%s'" % path[0])
            get_namespace(getattr(obj, path[0]), path[1:], name)
        else:
            a = Namespace()
            setattr(obj, path[0], a)
            get_namespace(getattr(obj, path[0]), path[1:], name)


class Namespace(object):

    pass
