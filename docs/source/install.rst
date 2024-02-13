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

2. Zipline depends on `numpy <https://www.numpy.org/>`_, the core library for
   numerical array computing in Python.  Numpy depends on having the `LAPACK
   <https://www.netlib.org/lapack>`_ linear algebra routines available.

Because LAPACK and the CPython headers are non-Python dependencies, the correct
way to install them varies from platform to platform.  If you'd rather use a
single tool to install Python and non-Python dependencies, or if you're already
using `Anaconda <https://www.anaconda.com/distribution/>`_ as your Python distribution,
you can skip to the :ref:`Installing with Conda <conda>` section.

Once you've installed the necessary additional dependencies (see below for
your particular platform), you should be able to simply run

.. code-block:: bash

   $ pip install zipline

If you use Python for anything other than Zipline, we **strongly** recommend
that you install in a `virtualenv
<https://virtualenv.readthedocs.org/en/latest>`_.  The `Hitchhiker's Guide to
Python`_ provides an `excellent tutorial on virtualenv
<https://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

GNU/Linux
~~~~~~~~~

On `Debian-derived`_ Linux distributions, you can acquire all the necessary
binary dependencies from ``apt`` by running:

.. code-block:: bash

   $ sudo apt-get install libatlas-base-dev python-dev gfortran pkg-config libfreetype6-dev hdf5-tools

On recent `RHEL-derived`_ derived Linux distributions (e.g. Fedora), the
following should be sufficient to acquire the necessary additional
dependencies:

.. code-block:: bash

   $ sudo dnf install atlas-devel gcc-c++ gcc-gfortran libgfortran python-devel redhat-rpm-config hdf5

On `Arch Linux`_, you can acquire the additional dependencies via ``pacman``:

.. code-block:: bash

   $ pacman -S lapack gcc gcc-fortran pkg-config hdf5

There are also AUR packages available for installing `ta-lib
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
to `Installing Python on OSX <https://docs.python-guide.org/en/latest/>`_, which
explains how to install Python with the `Homebrew`_ manager.

Assuming you've installed Python with Homebrew, you'll also likely need the
following brew packages:

.. code-block:: bash

   $ brew install freetype pkg-config gcc openssl hdf5

Windows
~~~~~~~

For windows, the easiest and best supported way to install zipline is to use
:ref:`Conda <conda>`.

.. _conda:

Installing with ``conda``
-------------------------

Another way to install Zipline is via the ``conda`` package manager, which
comes as part of Continuum Analytics' `Anaconda
<https://www.anaconda.com/distribution/>`_ distribution.

The primary advantage of using Conda over ``pip`` is that conda natively
understands the complex binary dependencies of packages like ``numpy`` and
``scipy``.  This means that ``conda`` can install Zipline and its dependencies
without requiring the use of a second tool to acquire Zipline's non-Python
dependencies.

For instructions on how to install ``conda``, see the `Conda Installation
Documentation <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_

Once ``conda`` has been set up you can install Zipline from the ``conda-forge`` channel:

.. code-block:: bash

    conda install -c conda-forge zipline

.. _`Debian-derived`: https://www.debian.org/derivatives/
.. _`RHEL-derived`: https://en.wikipedia.org/wiki/Red_Hat_Enterprise_Linux_derivatives
.. _`Arch Linux` : https://www.archlinux.org/
.. _`Hitchhiker's Guide to Python` : https://docs.python-guide.org/en/latest/
.. _`Homebrew` : https://brew.sh

.. _managing-conda-environments:

Managing ``conda`` environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is recommended to install Zipline in an isolated ``conda`` environment.
Installing Zipline in ``conda`` environments will not interfere your default
Python deployment or site-packages, which will prevent any possible conflict
with your global libraries. For more information on ``conda`` environment, see
the `Conda User Guide <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Assuming ``conda`` has been set up, you can create a ``conda`` environment:

.. code-block:: bash

    $ conda create -n env_zipline python=3.6


Now you have set up an isolated environment called ``env_zipline``, a sandbox-like
structure to install Zipline. Then you should activate the conda environment
by using the command

.. code-block:: bash

    $ conda activate env_zipline

You can install Zipline by running

.. code-block:: bash

    (env_zipline) $ conda install -c conda-forge zipline

.. note::

    The ``conda-forge`` channel so far only has zipline 1.4.0+ packages for python 3.6.
    Conda packages for previous versions of zipline for pythons 2.7/3.5/3.6 are
    still available on Quantopian's anaconda channel, but are not being updated.
    They can be installed with:

    .. code-block:: bash

        (env_zipline35) $ conda install -c Quantopian zipline

To deactivate the ``conda`` environment:

.. code-block:: bash

    (env_zipline) $ conda deactivate

.. note::
   ``conda activate`` and ``conda deactivate`` only work on conda 4.6 and later versions. For conda versions prior to 4.6, run:

      * Windows: ``activate`` or ``deactivate``
      * Linux and macOS: ``source activate`` or ``source deactivate``
