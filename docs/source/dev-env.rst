Zipline development guidelines
==============================

Commit messages
---------------

Standard acronyms to start the commit message with are:

.. code-block:: bash

   BLD: change related to building zipline
   BUG: bug fix
   DEP: deprecate something, or remove a deprecated object
   DEV: development tool or utility
   DOC: documentation
   ENH: enhancement
   MAINT: maintenance commit (refactoring, typos, etc.)
   REV: revert an earlier commit
   STY: style fix (whitespace, PEP8)
   TST: addition or modification of tests
   REL: related to releasing Zipline
   PERF: Performance enhancements

Some commit style guidelines:

Commit lines should be no longer than 72 characters. First line of the commit should include a the aforementioned prefix. There should be an empty line between the commit subject and the body of the commit. In general, the message should be in the imperative tense. Best practice is to include not only what change, by why.

.e.g.

.. code-block:: bash

   MAINT: Remove unused calculations of max_leverage, et al.

   In the performance period the max_leverage, max_capital_used,
   cumulative_capital_used were calculated but not used.

   At least one of those calculations, max_leverage, was causing a
   divide by zero error.
   Instead of papering over that error, the entire calculation was
   a bit suspect so removing, with possibility of adding it back in
   later with handling the case (or raising appropriate errors) when
   the algorithm has little cash on hand.

Pulling in Pull Requests (PRs)
------------------------------

Pulling in Pull Requests (PRs)

.. code-block:: bash

   (master) $ git checkout -b PR-135
   $ curl https://github.com/quantopian/zipline/pull/135.patch | git am
   # Clean up commit history
   $ git rebase -i master
   # Merge (use no-ff for many commits and ff for few)
   $ git merge --no-ff --edit


