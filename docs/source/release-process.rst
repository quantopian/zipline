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


then include this file in ``docs/source/releases.rst``. New releases should
appear at the top. The syntax for this is:

::

   .. include:: whatsnew/<version>.txt


Updating the ``__version__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use `versioneer <https://github.com/warner/python-versioneer>`__ to
manage the ``__version__`` and ``setup.py`` version. This means that we pull
this information from our version control's tags to ensure that they stay in
sync and to have very fine grained version strings for development installs.

To upgrade the version use the git tag command like:

.. code-block:: bash

   $ git tag <major>.<minor>.<micro>
   $ git push
   $ git push --tags


This will push the the code and the tag information.


Uploading PyPI packages
~~~~~~~~~~~~~~~~~~~~~~~

``sdist``
^^^^^^^^^

To build the ``sdist`` (source distribution) run:

.. code-block:: bash

   $ python setup.by sdist


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

   $ pip install -i https://testpypi.python.org/pypi zipline
   $ python -c 'import zipline;print(zipline.__version__)'


This should pull the package you just uploaded and then print the version
number.

Now that we have tested locally and on PyPI test, it is time to upload to PyPI:

.. code-block:: bash

   $ python setup.py sdist upload

``bdist``
^^^^^^^^^

.. note::

   If you are running on GNU/Linux, then you cannot upload any binary wheels.

First, build the wheels locally with:

.. code-block:: bash

   $ python setup.py bdist_wheel


Just like the ``sdist``, we need to ``cd`` into a clean directory and use a
clean virtualenv. Then, test that the wheel was built successfully with:

.. code-block:: bash

   $ pip install <zipline_root>/dist/<wheel_name>
   $ python -c 'import zipline;print(zipline.__version__)'

The version number should be the same as the version you are releasing.
We must repeat this process for both python 2 and 3.
Once you have tested the package, it can be uploaded to PyPI with:

.. code-block:: bash

   $ python setup.py bdist_wheel upload

Documentation
~~~~~~~~~~~~~

To update `zipline.io <http://www.zipline.io/index.html>`__, checkout the
latest master and run:

.. code-block:: python

    python <zipline_root>/docs/deploy.py

This will build the documentation, checkout a fresh copy of the ``gh-pages``
git branch, and copy the built docs into the zipline root.

Now, using our browser of choice, view the ``index.html`` page and verify that
the docs look correct.

Once we are happy, push the updated docs to the GitHub ``gh-pages`` branch.

.. code-block:: bash

   $ git add .
   $ git commit -m "DOC: update zipline.io"
   $ git push origin gh-pages

`zipline.io <http://www.zipline.io/index.html>`__ will update in a few moments.
