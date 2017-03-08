Zipline Release Process
-----------------------

.. include:: dev-doc-message.txt


Updating the Release Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~

When we are ready to ship a new release of zipline, edit the :doc:`releases`
page. We will have been maintaining a whatsnew file while working on the release
with the new version. First, find that file in:
``docs/source/whatsnew/<version>.txt``. It will be the highest version number.
Edit the release date field to be today's date in the format:

::

   <month> <day>, <year>


for example, November 6, 2015.
Remove the active development warning from the whatsnew, since it will no
longer be pending release.
Update the title of the release from "Development" to "Release x.x.x" and
update the underline of the title to match the title's width.

If you are renaming the release at this point, you'll need to git mv the file
and also update releases.rst to reference the renamed file.

To build and view the docs locally, run:

.. code-block:: bash

   $ cd docs
   $ make html
   $ {BROWSER} build/html/index.html

Updating the Python stub files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyCharm and other linters and type checkers can use `Python stub files
<https://www.python.org/dev/peps/pep-0484/#stub-files>`__ for type hinting. For
example, we generate stub files for the :mod:`~zipline.api` namespace, since that
namespace is populated at import time by decorators on TradingAlgorithm
methods. Those functions are therefore hidden from static analysis tools, but
we can generate static files to make them available. Under **Python 3**, run
the following to generate any stub files:

.. code-block:: bash

   $ python etc/gen_type_stubs.py

.. note::

   In order to make stub consumers aware of the classes referred to in the
   stub, the stub file should import those classes.  However, since
   ``... import *`` and ``... import ... as ...`` in a stub file will export
   those imports, we import the names explicitly.  For the stub for
   ``zipline.api``, this is done in a header string in the
   ``gen_type_stubs.py`` script mentioned above.  If new classes are added as
   parameters or return types of ``zipline.api`` functions, then new imports
   should be added to that header.

Updating the ``__version__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use `versioneer <https://github.com/warner/python-versioneer>`__ to
manage the ``__version__`` and ``setup.py`` version. This means that we pull
this information from our version control's tags to ensure that they stay in
sync and to have very fine grained version strings for development installs.

To upgrade the version use the git tag command like:

.. code-block:: bash

   $ git tag <major>.<minor>.<micro>
   $ git push && git push --tags


This will push the the code and the tag information.

Next, click the "Draft a new release" button on the `zipline releases page
<https://github.com/quantopian/zipline/releases>`__.  For the new release,
choose the tag you just pushed, and publish the release.

Uploading PyPI packages
~~~~~~~~~~~~~~~~~~~~~~~

``sdist``
^^^^^^^^^

To build the ``sdist`` (source distribution) run:

.. code-block:: bash

   $ python setup.py sdist


from the zipline root. This will create a gzipped tarball that includes all the
python, cython, and miscellaneous files needed to install zipline. To test that
the source dist worked correctly, ``cd`` into an empty directory, create a new
virtualenv and then run:


.. code-block:: bash

   $ pip install <zipline-root>/dist/zipline-<major>.<minor>.<micro>.tar.gz
   $ python -c 'import zipline;print(zipline.__version__)'

This should print the version we are expecting to release.

.. note::

   It is very important to both ``cd`` into a clean directory and make a clean
   virtualenv. Changing directories ensures that we have included all the needed
   files in the manifest. Using a clean virtualenv ensures that we have listed
   all the required packages.

Now that we have tested the package locally, it should be tested using the test
PyPI server.

Edit your ``~/.pypirc`` file to look like:

::

   [distutils]
   index-servers =
       pypi
       pypitest

   [pypi]
   username:
   password:

   [pypitest]
   repository: https://testpypi.python.org/pypi
   username:
   password:

after that, run:

.. code-block:: bash

   $ python setup.py sdist upload -r pypitest


.. note::

   If the package version has been taken: locally update your setup.py to
   override the version with a new number. Do not use the next version, just
   append a ``.<nano>`` section to the current version. PyPI prevents the same
   package version from appearing twice, so we need to work around this when
   debugging packaging problems on the test server.

   .. warning::

      Do not commit the temporary version change.


This will upload zipline to the pypi test server. To test installing from pypi,
create a new virtualenv, ``cd`` into a clean directory and then run:

.. code-block:: bash

   $ pip install --extra-index-url https://testpypi.python.org/pypi zipline
   $ python -c 'import zipline;print(zipline.__version__)'


This should pull the package you just uploaded and then print the version
number.

Now that we have tested locally and on PyPI test, it is time to upload to PyPI:

.. code-block:: bash

   $ python setup.py sdist upload

``bdist``
^^^^^^^^^

Because zipline now supports multiple versions of numpy, we're not building
binary wheels, since they are not tagged with the version of numpy with which
they were compiled.

Documentation
~~~~~~~~~~~~~

To update `zipline.io <http://www.zipline.io/index.html>`__, checkout the
latest master and run:

.. code-block:: python

    python <zipline_root>/docs/deploy.py

This will build the documentation, checkout a fresh copy of the ``gh-pages``
git branch, and copy the built docs into the zipline root.

.. note::

   The docs should always be built with **Python 3**. Many of our api functions
   are wrapped by preprocessing functions which accept \*args and \**kwargs. In
   Python 3, sphinx will respect the ``__wrapped__`` attribute and display the
   correct arguments.

Now, using our browser of choice, view the ``index.html`` page and verify that
the docs look correct.

Once we are happy, push the updated docs to the GitHub ``gh-pages`` branch.

.. code-block:: bash

   $ git add .
   $ git commit -m "DOC: update zipline.io"
   $ git push origin gh-pages

`zipline.io <http://www.zipline.io/index.html>`__ will update in a few moments.

Uploading conda packages
~~~~~~~~~~~~~~~~~~~~~~~~

Travis and AppVeyor build zipline conda packages for us.  Once they have built
and uploaded to anaconda.org the packages (and their dependencies) for the
release commit to master, we should move those packages from the "ci" label to
the "main" label.  You can do this from the anaconda.org web interface.  This
is also a good time to remove all the old "ci" packages from anaconda.

Travis and AppVeyor only build and upload linux-64 and win-64 packages.  We'll
need to build and upload osx-64 packages manually on an OSX machine.

To build the conda packages for zipline locally, run:

.. code-block:: bash

   $ python etc/conda_build_matrix.py

If all of the builds succeed, then this will not print anything and exit with
``EXIT_SUCCESS``. If there are build issues, we must address them and decide
what to do.

Once all of the builds in the matrix pass, we can upload them to anaconda with:

.. code-block:: bash

   $ python etc/conda_build_matrix.py --upload

If you would like to test this command by uploading to a different user, this
may be specified with the ``--user`` flag.

Next Commit
~~~~~~~~~~~

Push a new commit post-release that adds the whatsnew for the next release,
which should be titled according to a micro version increment. If that next
release turns out to be a major/minor version increment, the file can be
renamed when that's decided. You can use ``docs/source/whatsnew/skeleton.txt``
as a template for the new file.

Include the whatsnew file in ``docs/source/releases.rst``. New releases should
appear at the top. The syntax for this is:

::

   .. include:: whatsnew/<version>.txt
