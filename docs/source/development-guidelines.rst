Development Guidelines
======================
This page is intended for developers of Zipline, people who want to contribute to the Zipline codebase or documentation, or people who want to install from source and make local changes to their copy of Zipline.

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome. We `track issues`__ on `GitHub`__ and also have a `mailing list`__ where you can ask questions.

__ https://github.com/quantopian/zipline/issues
__ https://github.com/
__ https://groups.google.com/forum/#!forum/zipline

Creating a Development Environment
----------------------------------

First, you'll need to clone Zipline by running:

.. code-block:: bash

   $ git clone git@github.com:your-github-username/zipline.git

Then check out to a new branch where you can make your changes:

.. code-block:: bash
		
   $ git checkout -b some-short-descriptive-name

If you don't already have them, you'll need some C library dependencies. You can follow the `install guide`__ to get the appropriate dependencies.

__ install.html

The following section assumes you already have virtualenvwrapper and pip installed on your system. Suggested installation of Python library dependencies used for development:

.. code-block:: bash

   $ mkvirtualenv zipline
   $ ./etc/ordered_pip.sh ./etc/requirements.txt
   $ pip install -r ./etc/requirements_dev.txt
   $ pip install -r ./etc/requirements_blaze.txt 

Finally, you can build the C extensions by running:

.. code-block:: bash

   $ python setup.py build_ext --inplace

To finish, make sure `tests`__ pass.

__ #style-guide-running-tests

If you get an error running nosetests after setting up a fresh virtualenv, please try running

.. code-block:: bash

   # where zipline is the name of your virtualenv
   $ deactivate zipline
   $ workon zipline


Development with Docker
-----------------------

If you want to work with zipline using a `Docker`__ container, you'll need to build the ``Dockerfile`` in the Zipline root directory, and then build ``Dockerfile-dev``. Instructions for building both containers can be found in ``Dockerfile`` and ``Dockerfile-dev``, respectively.

__ https://docs.docker.com/get-started/


Style Guide & Running Tests
---------------------------

We use `flake8`__ for checking style requirements and `nosetests`__ to run Zipline tests. Our `continuous integration`__ tools will run these commands.

__ http://flake8.pycqa.org/en/latest/
__ http://nose.readthedocs.io/en/latest/
__ https://en.wikipedia.org/wiki/Continuous_integration

Before submitting patches or pull requests, please ensure that your changes pass when running:

.. code-block:: bash

   $ flake8 zipline tests

In order to run tests locally, you'll need `TA-lib`__, which you can install on Linux by running:

__ https://mrjbq7.github.io/ta-lib/install.html

.. code-block:: bash

   $ wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   $ tar -xvzf ta-lib-0.4.0-src.tar.gz
   $ cd ta-lib/
   $ ./configure --prefix=/usr
   $ make
   $ sudo make install

And for ``TA-lib`` on OS X you can just run:

.. code-block:: bash

   $ brew install ta-lib

Then run ``pip install`` TA-lib:

.. code-block:: bash

   $ pip install -r ./etc/requirements_talib.txt

You should now be free to run tests:

.. code-block:: bash
		
   $ nosetests


Continuous Integration
----------------------

We use `Travis CI`__ for Linux-64 bit builds and `AppVeyor`__ for Windows-64 bit builds.

.. note::

   We do not currently have CI for OSX-64 bit builds. 32-bit builds may work but are not included in our integration tests.

__ https://travis-ci.org/quantopian/zipline
__ https://ci.appveyor.com/project/quantopian/zipline


Packaging
---------
To learn about how we build Zipline conda packages, you can read `this`__ section in our release process notes.

__ release-process.html#uploading-conda-packages
   
Contributing to the Docs
------------------------

If you'd like to contribute to the documentation on zipline.io, you can navigate to ``docs/source/`` where each `reStructuredText`__ (``.rst``) file is a separate section there. To add a section, create a new file called ``some-descriptive-name.rst`` and add ``some-descriptive-name`` to ``appendix.rst``. To edit a section, simply open up one of the existing files, make your changes, and save them.

__ https://en.wikipedia.org/wiki/ReStructuredText

We use `Sphinx`__ to generate documentation for Zipline, which you will need to install by running:

__ http://www.sphinx-doc.org/en/stable/


.. code-block:: bash

   $ pip install -r ./etc/requirements_docs.txt

To build and view the docs locally, run:

.. code-block:: bash

   # assuming you're in the Zipline root directory
   $ cd docs
   $ make html
   $ {BROWSER} build/html/index.html


Commit messages
---------------

Standard prefixes to start a commit message:

.. code-block:: text

   BLD: change related to building Zipline
   BUG: bug fix
   DEP: deprecate something, or remove a deprecated object
   DEV: development tool or utility
   DOC: documentation
   ENH: enhancement
   MAINT: maintenance commit (refactoring, typos, etc)
   REV: revert an earlier commit
   STY: style fix (whitespace, PEP8, flake8, etc)
   TST: addition or modification of tests
   REL: related to releasing Zipline
   PERF: performance enhancements


Some commit style guidelines:

Commit lines should be no longer than `72 characters`__. The first line of the commit should include one of the above prefixes. There should be an empty line between the commit subject and the body of the commit. In general, the message should be in the imperative tense. Best practice is to include not only what the change is, but why the change was made.

__ https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project

**Example:**

.. code-block:: text

   MAINT: Remove unused calculations of max_leverage, et al.

   In the performance period the max_leverage, max_capital_used,
   cumulative_capital_used were calculated but not used.

   At least one of those calculations, max_leverage, was causing a
   divide by zero error.
   
   Instead of papering over that error, the entire calculation was
   a bit suspect so removing, with possibility of adding it back in
   later with handling the case (or raising appropriate errors) when
   the algorithm has little cash on hand.


Formatting Docstrings
---------------------

When adding or editing docstrings for classes, functions, etc, we use `numpy`__ as the canonical reference.

__ https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt


Updating the Whatsnew
---------------------

We have a set of `whatsnew <https://github.com/quantopian/zipline/tree/master/docs/source/whatsnew>`__ files that are used for documenting changes that have occurred between different versions of Zipline.
Once you've made a change to Zipline, in your Pull Request, please update the most recent ``whatsnew`` file with a comment about what you changed. You can find examples in previous ``whatsnew`` files.
