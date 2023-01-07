from abc import ABC, abstractmethod

# Consistent error to be thrown in various cases regarding overriding
# `final` attributes.
_type_error = TypeError("Cannot override final attribute")


def bases_mro(bases):
    """
    Yield classes in the order that methods should be looked up from the
    base classes of an object.
    """
    for base in bases:
        for class_ in base.__mro__:
            yield class_


def is_final(name, mro):
    """
    Checks if `name` is a `final` object in the given `mro`.
    We need to check the mro because we need to directly go into the __dict__
    of the classes. Because `final` objects are descriptor, we need to grab
    them _BEFORE_ the `__call__` is invoked.
    """
    return any(
        isinstance(getattr(c, "__dict__", {}).get(name), final) for c in bases_mro(mro)
    )


class FinalMeta(type):
    """A metaclass template for classes the want to prevent subclassess from
    overriding some methods or attributes.
    """

    def __new__(metacls, name, bases, dict_):
        for k, _ in dict_.items():
            if is_final(k, bases):
                raise _type_error

        setattr_ = dict_.get("__setattr__")
        if setattr_ is None:
            # No `__setattr__` was explicitly defined, look up the super
            # class's. `bases[0]` will have a `__setattr__` because
            # `object` does so we don't need to worry about the mro.
            setattr_ = bases[0].__setattr__

        if not is_final("__setattr__", bases) and not isinstance(setattr_, final):
            # implicitly make the `__setattr__` a `final` object so that
            # users cannot just avoid the descriptor protocol.
            dict_["__setattr__"] = final(setattr_)

        return super(FinalMeta, metacls).__new__(metacls, name, bases, dict_)

    def __setattr__(metacls, name, value):
        """This stops the `final` attributes from being reassigned on the
        class object.
        """
        if is_final(name, metacls.__mro__):
            raise _type_error

        super(FinalMeta, metacls).__setattr__(name, value)


class final(ABC):
    """
    An attribute that cannot be overridden.
    This is like the final modifier in Java.

    Example usage:
    >>> class C(object, metaclass=FinalMeta):
    ...    @final
    ...    def f(self):
    ...        return 'value'
    ...

    This constructs a class with final method `f`. This cannot be overridden
    on the class object or on any instance. You cannot override this by
    subclassing `C`; attempting to do so will raise a `TypeError` at class
    construction time.
    """

    def __new__(cls, attr):
        # Decide if this is a method wrapper or an attribute wrapper.
        # We are going to cache the `callable` check by creating a
        # method or attribute wrapper.
        if hasattr(attr, "__get__"):
            return object.__new__(finaldescriptor)
        else:
            return object.__new__(finalvalue)

    def __init__(self, attr):
        self._attr = attr

    def __set__(self, instance, value):
        """
        `final` objects cannot be reassigned. This is the most import concept
        about `final`s.

        Unlike a `property` object, this will raise a `TypeError` when you
        attempt to reassign it.
        """
        raise _type_error

    @abstractmethod
    def __get__(self, instance, owner):
        raise NotImplementedError("__get__")


class finalvalue(final):
    """
    A wrapper for a non-descriptor attribute.
    """

    def __get__(self, instance, owner):
        return self._attr


class finaldescriptor(final):
    """A final wrapper around a descriptor."""

    def __get__(self, instance, owner):
        return self._attr.__get__(instance, owner)
