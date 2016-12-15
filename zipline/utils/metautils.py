from operator import attrgetter

import six


def compose_types(a, *cs):
    """Compose multiple classes together.

    Parameters
    ----------
    *mcls : tuple[type]
        The classes that you would like to compose

    Returns
    -------
    cls : type
        A type that subclasses all of the types in ``mcls``.

    Notes
    -----
    A common use case for this is to build composed metaclasses, for example,
    imagine you have some simple metaclass ``M`` and some instance of ``M``
    named ``C`` like so:

    .. code-block:: python

       >>> class M(type):
       ...     def __new__(mcls, name, bases, dict_):
       ...         dict_['ayy'] = 'lmao'
       ...         return super(M, mcls).__new__(mcls, name, bases, dict_)


       >>> from six import with_metaclass
       >>> class C(with_metaclass(M, object)):
       ...     pass


    We now want to create a sublclass of ``C`` that is also an abstract class.
    We can use ``compose_types`` to create a new metaclass that is a subclass
    of ``M`` and ``ABCMeta``. This is needed because a subclass of a class
    with a metaclass must have a metaclass which is a subclass of the metaclass
    of the superclass.


    .. code-block:: python

       >>> from abc import ABCMeta, abstractmethod
       >>> class D(with_metaclass(compose_types(M, ABCMeta), C)):
       ...     @abstractmethod
       ...     def f(self):
       ...         raise NotImplementedError('f')


    We can see that this class has both metaclasses applied to it:

    .. code-block:: python

       >>> D.ayy
       'lmao'
       >>> D()
       Traceback (most recent call last):
          ...
       TypeError: Can't instantiate abstract class D with abstract methods f


    An important note here is that ``M`` did not use ``type.__new__`` and
    instead used ``super()``. This is to support cooperative multiple
    inheritence which is needed for ``compose_types`` to work as intended.
    After we have composed these types ``M.__new__``\'s super will actually
    go to ``ABCMeta.__new__`` and not ``type.__new__``.

    Always using ``super()`` to dispatch to your superclass is best practices
    anyways so most classes should compose without much special considerations.
    """
    if not cs:
        # if there are no types to compose then just return the single type
        return a

    mcls = (a,) + cs
    return type(
        'compose_types(%s)' % ', '.join(map(attrgetter('__name__'), mcls)),
        mcls,
        {},
    )


def with_metaclasses(metaclasses, *bases):
    """Make a class inheriting from ``bases`` whose metaclass inherits from
    all of ``metaclasses``.

    Like :func:`six.with_metaclass`, but allows multiple metaclasses.

    Parameters
    ----------
    metaclasses : iterable[type]
        A tuple of types to use as metaclasses.
    *bases : tuple[type]
        A tuple of types to use as bases.

    Returns
    -------
    base : type
        A subtype of ``bases`` whose metaclass is a subtype of ``metaclasses``.

    Notes
    -----
    The metaclasses must be written to support cooperative multiple
    inheritance. This means that they must delegate all calls to ``super()``
    instead of inlining their super class by name.
    """
    return six.with_metaclass(compose_types(*metaclasses), *bases)
