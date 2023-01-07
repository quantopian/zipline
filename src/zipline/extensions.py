import re
from toolz import curry


def create_args(args, root):
    """
    Encapsulates a set of custom command line arguments in key=value
    or key.namespace=value form into a chain of Namespace objects,
    where each next level is an attribute of the Namespace object on the
    current level

    Parameters
    ----------
    args : list
        A list of strings representing arguments in key=value form
    root : Namespace
        The top-level element of the argument tree
    """

    extension_args = {}

    for arg in args:
        parse_extension_arg(arg, extension_args)

    for name in sorted(extension_args, key=len):
        path = name.split(".")
        update_namespace(root, path, extension_args[name])


def parse_extension_arg(arg, arg_dict):
    """
    Converts argument strings in key=value or key.namespace=value form
    to dictionary entries

    Parameters
    ----------
    arg : str
        The argument string to parse, which must be in key=value or
        key.namespace=value form.
    arg_dict : dict
        The dictionary into which the key/value pair will be added
    """

    match = re.match(r"^(([^\d\W]\w*)(\.[^\d\W]\w*)*)=(.*)$", arg)
    if match is None:
        raise ValueError(
            "invalid extension argument '%s', must be in key=value form" % arg
        )

    name = match.group(1)
    value = match.group(4)
    arg_dict[name] = value


def update_namespace(namespace, path, name):
    """
    A recursive function that takes a root element, list of namespaces,
    and the value being stored, and assigns namespaces to the root object
    via a chain of Namespace objects, connected through attributes

    Parameters
    ----------
    namespace : Namespace
        The object onto which an attribute will be added
    path : list
        A list of strings representing namespaces
    name : str
        The value to be stored at the bottom level
    """

    if len(path) == 1:
        setattr(namespace, path[0], name)
    else:
        if hasattr(namespace, path[0]):
            if isinstance(getattr(namespace, path[0]), str):
                raise ValueError(
                    "Conflicting assignments at namespace" " level '%s'" % path[0]
                )
        else:
            a = Namespace()
            setattr(namespace, path[0], a)

        update_namespace(getattr(namespace, path[0]), path[1:], name)


class Namespace:
    """
    A placeholder object representing a namespace level
    """


class Registry:
    """
    Responsible for managing all instances of custom subclasses of a
    given abstract base class - only one instance needs to be created
    per abstract base class, and should be created through the
    create_registry function/decorator. All management methods
    for a given base class can be called through the global wrapper functions
    rather than through the object instance itself.

    Parameters
    ----------
    interface : type
        The abstract base class to manage.
    """

    def __init__(self, interface):
        self.interface = interface
        self._factories = {}

    def load(self, name):
        """Construct an object from a registered factory.

        Parameters
        ----------
        name : str
            Name with which the factory was registered.
        """
        try:
            return self._factories[name]()
        except KeyError as exc:
            raise ValueError(
                "no %s factory registered under name %r, options are: %r"
                % (self.interface.__name__, name, sorted(self._factories)),
            ) from exc

    def is_registered(self, name):
        """Check whether we have a factory registered under ``name``."""
        return name in self._factories

    @curry
    def register(self, name, factory):
        if self.is_registered(name):
            raise ValueError(
                "%s factory with name %r is already registered"
                % (self.interface.__name__, name)
            )

        self._factories[name] = factory

        return factory

    def unregister(self, name):
        try:
            del self._factories[name]
        except KeyError as exc:
            raise ValueError(
                "%s factory %r was not already registered"
                % (self.interface.__name__, name)
            ) from exc

    def clear(self):
        self._factories.clear()


# Public wrapper methods for Registry:


def get_registry(interface):
    """
    Getter method for retrieving the registry
    instance for a given extendable type

    Parameters
    ----------
    interface : type
        extendable type (base class)

    Returns
    -------
    manager : Registry
        The corresponding registry
    """
    try:
        return custom_types[interface]
    except KeyError as exc:
        raise ValueError("class specified is not an extendable type") from exc


def load(interface, name):
    """
    Retrieves a custom class whose name is given.

    Parameters
    ----------
    interface : type
        The base class for which to perform this operation
    name : str
        The name of the class to be retrieved.

    Returns
    -------
    obj : object
        An instance of the desired class.
    """
    return get_registry(interface).load(name)


@curry
def register(interface, name, custom_class):
    """
    Registers a class for retrieval by the load method

    Parameters
    ----------
    interface : type
        The base class for which to perform this operation
    name : str
        The name of the subclass
    custom_class : type
        The class to register, which must be a subclass of the
        abstract base class in self.dtype
    """

    return get_registry(interface).register(name, custom_class)


def unregister(interface, name):
    """
    If a class is registered with the given name,
    it is unregistered.

    Parameters
    ----------
    interface : type
        The base class for which to perform this operation
    name : str
        The name of the class to be unregistered.
    """
    get_registry(interface).unregister(name)


def clear(interface):
    """
    Unregisters all current registered classes

    Parameters
    ----------
    interface : type
        The base class for which to perform this operation
    """
    get_registry(interface).clear()


def create_registry(interface):
    """
    Create a new registry for an extensible interface.

    Parameters
    ----------
    interface : type
        The abstract data type for which to create a registry,
        which will manage registration of factories for this type.

    Returns
    -------
    interface : type
        The data type specified/decorated, unaltered.
    """
    if interface in custom_types:
        raise ValueError(
            "there is already a Registry instance " "for the specified type"
        )
    custom_types[interface] = Registry(interface)
    return interface


extensible = create_registry

# A global dictionary for storing instances of Registry:
custom_types = {}
