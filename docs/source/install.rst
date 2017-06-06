Install
=======

Installing with ``pip``
-----------------------

Installing Zipline via ``pip`` is slightly more involved than the average
Python package.

There are two reasons for the additional complexity:

1. Zipline ships several C extensions that require access to the CPython C API.
   In order to build the C extensions, ``pip`` needs access to the CPython
   header files for your Python installation.

2. Zipline depends on `numpy <http://www.numpy.org/>`_, the core library for
   numerical array computing in Python.  Numpy depends on having the `LAPACK
   <http://www.netlib.org/lapack>`_ linear algebra routines available.

Because LAPACK and the CPython headers are non-Python dependencies, the correct
way to install them varies from platform to platform.  If you'd rather use a
single tool to install Python and non-Python dependencies, or if you're already
using `Anaconda <http://continuum.io/downloads>`_ as your Python distribution,
you can skip to the :ref:`Installing with Conda <conda>` section.

Once you've installed the necessary additional dependencies (see below for
your particular platform), you should be able to simply run

.. code-block:: bash

   $ pip install zipline

If you use Python for anything other than Zipline, we **strongly** recommend
that you install in a `virtualenv
<https://virtualenv.readthedocs.org/en/latest>`_.  The `Hitchhiker's Guide to
Python`_ provides an `excellent tutorial on virtualenv
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

GNU/Linux
~~~~~~~~~

On `Debian-derived`_ Linux distributions, you can acquire all the necessary
binary dependencies from ``apt`` by running:

.. code-block:: bash

   $ sudo apt-get install libatlas-base-dev python-dev gfortran pkg-config libfreetype6-dev

On recent `RHEL-derived`_ derived Linux distributions (e.g. Fedora), the
following should be sufficient to acquire the necessary additional
dependencies:

.. code-block:: bash

   $ sudo dnf install atlas-devel gcc-c++ gcc-gfortran libgfortran python-devel redhat-rep-config

On `Arch Linux`_, you can acquire the additional dependencies via ``pacman``:

.. code-block:: bash

   $ pacman -S lapack gcc gcc-fortran pkg-config

There are also AUR packages available for installing `Python 3.4
<https://aur.archlinux.org/packages/python34/>`_ (Arch's default python is now
3.5, but Zipline only currently supports 3.4), and `ta-lib
<https://aur.archlinux.org/packages/ta-lib/>`_, an optional Zipline dependency.
Python 2 is also installable via:

.. code-block:: bash

   $ pacman -S python2

OSX
~~~

The version of Python shipped with OSX by default is generally out of date, and
has a number of quirks because it's used directly by the operating system.  For
these reasons, many developers choose to install and use a separate Python
installation. The `Hitchhiker's Guide to Python`_ provides an excellent guide
to `Installing Python on OSX <http://docs.python-guide.org/en/latest/>`_, which
explains how to install Python with the `Homebrew`_ manager.

Assuming you've installed Python with Homebrew, you'll also likely need the
following brew packages:

.. code-block:: bash

   $ brew install freetype pkg-config gcc openssl

Windows
~~~~~~~

For windows, the easiest and best supported way to install zipline is to use
:ref:`Conda <conda>`.

.. _conda:

Installing with ``conda``
-------------------------

Another way to install Zipline is via the ``conda`` package manager, which
comes as part of Continuum Analytics' `Anaconda
<http://continuum.io/downloads>`_ distribution.

The primary advantage of using Conda over ``pip`` is that conda natively
understands the complex binary dependencies of packages like ``numpy`` and
``scipy``.  This means that ``conda`` can install Zipline and its dependencies
without requiring the use of a second tool to acquire Zipline's non-Python
dependencies.

For instructions on how to install ``conda``, see the `Conda Installation
Documentation <http://conda.pydata.org/docs/download.html>`_

Once conda has been set up you can install Zipline from our ``Quantopian``
channel:

.. code-block:: bash

    conda install -c Quantopian zipline

.. _`Debian-derived`: https://www.debian.org/misc/children-distros
.. _`RHEL-derived`: https://en.wikipedia.org/wiki/Red_Hat_Enterprise_Linux_derivatives
.. _`Arch Linux` : https://www.archlinux.org/
.. _`Hitchhiker's Guide to Python` : http://docs.python-guide.org/en/latest/
.. _`Homebrew` : http://brew.sh
