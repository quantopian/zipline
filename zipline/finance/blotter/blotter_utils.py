from functools import partial
from zipline.finance.blotter.blotter import Blotter
from zipline.utils.compat import mappingproxy


class BlotterClassDispatcher(object):
    """
    A class for registering and dispatching custom blotter classes.

    Method of a global instance of this class are provided by
    zipline.utils.calendar_utils.

    Parameters
    ----------
    blotter_factories : dict[str -> function]
        Factories for lazy blotter creation.
    """
    def __init__(self, blotter_factories):
        self._blotter_factories = blotter_factories
        self.blotter_factories = mappingproxy(self._blotter_factories)

    def get_blotter_class(self, name):
        """
        Retrieves a blotter whose name is given.

        Parameters
        ----------
        name : str
            The name of the blotter class to be retrieved.

        Returns
        -------
        blotter_class : zipline.finance.blotter.Blotter
            The desired blotter class.
        """
        try:
            return self._blotter_factories[name]
        except KeyError:
            raise ValueError("no blotter class registered as %r, options "
                             "are: %r" % (name, sorted(self._blotter_factories
                                                       )))

    def has_blotter_class(self, name):
        """
        Whether or not the global list of blotter classes contains the
        class with the specified name

        :param name: The name of the blotter class

        :return: True/False
        """

        return name in self._blotter_factories

    def register_blotter_class(self, name, blotter_class=None):
        """
        Registers a blotter class for retrieval by the
        get_blotter_class method

        :param name: The name of the blotter class
        :param blotter_class: The class to register, which must be a
        subclass of the abstract class zipline.finance.blotter.Blotter
        """

        if blotter_class is None:
            return partial(register_blotter_class, name)

        if self.has_blotter_class(name):
            raise ValueError("blotter class %r is already registered" % name)

        if not issubclass(blotter_class, Blotter):
            raise TypeError("The class specified is not a subclass of Blotter")

        self._blotter_factories[name] = blotter_class

        return blotter_class

    def unregister_blotter_class(self, name):
        """
        If a blotter class is registered with the given name,
        it is unregistered.

        Parameters
        ----------
        name : str
            The name of the blotter class to be unregistered.
        """
        try:
            del self._blotter_factories[name]
        except KeyError:
            raise ValueError("blotter class %r was not already registered"
                             % name)

    def clear_blotter_classes(self):
        """
        Unregisters all current registered calendars
        """
        self._blotter_factories.clear()


# Global blotter class dispatcher
global_blotter_class_dispatcher = BlotterClassDispatcher(
    blotter_factories={}
)

get_blotter_class = global_blotter_class_dispatcher.get_blotter_class
clear_blotter_classes = global_blotter_class_dispatcher.clear_blotter_classes
unregister_blotter_class = \
    global_blotter_class_dispatcher.unregister_blotter_class
register_blotter_class = \
    global_blotter_class_dispatcher.register_blotter_class
blotter_classes = global_blotter_class_dispatcher.blotter_factories
