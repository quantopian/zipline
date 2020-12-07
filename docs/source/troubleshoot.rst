
Troubleshooting/FAQ
==========================

| In this section I will put issues/questions users faced, so it might help new users too.

Dos And Donts
----------------
* Don't use this to ingest:

  .. code-block:: bash

    zipline ingest -b alpaca_api

| I changed the way we ingest new data bundles. please refer to

* Make sure you put the alpaca credentials in the right place
| First make sure the name of the file is ``alpaca.yaml``. (avoid mistakes like using ``.yml`` postfix)
| Put it in the right location - your root **python** folder. not inside the zipline-trader folder.