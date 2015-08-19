..
   --- BEGIN EMACS POWER USER STUFF ---
   Local Variables:
   mode: rst
   compile-command: "make -C .. html"
   End:
   --- END EMACS POWER USER STUFF ---

Design
======

Design Principles
-----------------

The design of the Modelling API is guided by a few important technical goals:

* Separate the **description** of a computation from the **execution** of that
  computation.
* Gracefully handle pricing data adjustments due to splits and dividends.
* Make it easy to re-use previously-defined factors.
