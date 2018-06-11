import re
import click
from zipline.__main__ import run
from functools import partial
from zipline.finance.blotter import Blotter
from zipline.algorithm import TradingAlgorithm
from zipline.utils.compat import mappingproxy
from collections import OrderedDict


def add_cli_option(name, choices, help):
    run.params.append(
        click.core.Option(
            param_decls=[name],
            type=click.Choice(choices),
            default='default',
            help=help,
        ),
    )


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


class RegistrationManager(object):

    def __init__(self, dtype):
        self.dtype = dtype
        self._classes = {}
        self.classes = mappingproxy(self._classes)
        add_cli_option(
            name="--%s-class" % type(self.dtype).__name__,
            choices=[],
            help="The subclass of %s to use, defaults to 'default'"
            % type(self.dtype).__name__
        )

    def load(self, name):
        try:
            return self._classes[name]
        except KeyError:
            raise ValueError(
                "no class registered under name %r, options are: %r" % (
                    name,
                    sorted(self._classes),
                ),
            )

    def class_exists(self, name):
        return name in self._classes

    def register(self, name, custom_class=None):
        if custom_class is None:
            return partial(self.register, name)

        if self.class_exists(name):
            raise ValueError("class %r is already registered" % name)

        if not issubclass(custom_class, self.dtype):
            raise TypeError(
                "The class specified is not a subclass of %s"
                % type(self.dtype).__name__
            )

        self._classes[name] = custom_class
        global_index = list(custom_types.keys()).index(self.dtype)
        run.params[global_index].type.choices.append(name)

        return custom_class

    def unregister(self, name):
        try:
            del self._classes[name]
            global_index = list(custom_types.keys()).index(self.dtype)
            choice_index = run.params[global_index].type.choices.index(name)
            run.params[global_index].type.choices.pop(choice_index)
        except KeyError:
            raise ValueError("class %r was not already registered" % name)

    def clear(self):
        self._classes.clear()

    def get_registered_classes(self):
        return self.classes


def get_registration_manager(dtype):
    """
    Getter method for retrieving the registration manager
    instance for a given extendable type

    Parameters
    ----------
    dtype : type
        extendable type (base class)

    Returns
    -------
    manager : RegistrationManager
        The corresponding registration manager
    """
    try:
        return custom_types[dtype]
    except KeyError:
        raise ValueError("class specified is not an extendable type")


def load(dtype, name):
    """
    Retrieves a custom class whose name is given.

    Parameters
    ----------
    dtype : type
        The base class for which to perform this operation
    name : str
        The name of the class to be retrieved.

    Returns
    -------
    class : type
        The desired class.
    """
    return get_registration_manager(dtype).load(name)


def class_exists(dtype, name):
    """
    Whether or not the global dictionary of classes contains the
    class with the specified name

    Parameters
    ----------
    dtype : type
        The base class for which to perform this operation
    name : str
        The name of the class

    Returns
    -------
    result : bool
        Whether or not a given class is registered
    """

    return get_registration_manager(dtype).class_exists(name)


def register(dtype, name, custom_class=None):
    """
    Registers a class for retrieval by the load method

    Parameters
    ----------
    dtype : type
        The base class for which to perform this operation
    name : str
        The name of the subclass
    custom_class : type
        The subclass to register

    class : type
        The class to register, which must be a subclass of the
        abstract base class in self.dtype
    """

    return get_registration_manager(dtype).register(name, custom_class)


def unregister(dtype, name):
    """
    If a class is registered with the given name,
    it is unregistered.

    Parameters
    ----------
    dtype : type
        The base class for which to perform this operation
    name : str
        The name of the class to be unregistered.
    """
    get_registration_manager(dtype).unregister(name)


def clear(dtype):
    """
    Unregisters all current registered classes

    Parameters
    ----------
    dtype : type
        The base class for which to perform this operation
    """
    get_registration_manager(dtype).clear()


def get_registered_classes(dtype):
    """
    A getter method for the dictionary of registered classes

    Parameters
    ----------
    dtype : type
        The base class for which to perform this operation

    Returns
    -------
    classes : dict
        The dictionary of registered classes
    """
    return get_registration_manager(dtype).get_registered_classes()


# Add any base classes that can be extended here, along with
# instances of RegistrationManager with the corresponding type
custom_types = OrderedDict([
    (Blotter, RegistrationManager(Blotter)),
    (TradingAlgorithm, RegistrationManager(TradingAlgorithm)),
])
