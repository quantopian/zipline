..
   --- BEGIN EMACS POWER USER STUFF ---
   Local Variables:
   mode: rst
   compile-command: "make -C .. html"
   End:
   --- END EMACS POWER USER STUFF ---

   To compile the Zipline docs from Emacs, run M-x compile.
   To compile from the command line, cd to zipline_repo/sphinx-doc and run ``make html``.

=======================
Quantopian Modeling API
=======================

This is the **ALPHA** documentation for the new Quantopian_ **Modeling API**.

.. warning::

   This documentation is a work in progress, as are the APIs described here.
   Many sections may be incomplete or innaccurate.  Please report
   inconsistencies or confusing elements you encounter.

   - All references below to ``zipline.modelling.*`` are slated to be changed in
     the near future to ``zipline.modeling.*``.

   - The name of the ``Factor`` class is likely to change in the future, due to
     worries that it collides with a term of art that specifically refers to
     computations involving correlations with known time-series data.

     Suggestions for alternative names currently include:
       - ``Indicator``
       - ``Metric``
       - ``Function``
       - ``Signal``

     Feedback on these names (or other suggestions) is encouraged and
     appeciated.


Contents:

.. toctree::
   :maxdepth: 3

   motivation.rst
   getting-started.rst
   api-reference.rst
   examples.rst

.. _Quantopian: www.quantopian.com
