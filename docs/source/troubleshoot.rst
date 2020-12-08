
Troubleshooting/FAQ
==========================

| In this section I will put issues/questions users faced, so it might help new users too.

Dos And Donts
----------------

Right Way to Ingest Data
)))))))))))))))))))))))))))

| Old Zipline users are used to do this with the ``zipline`` cli. For now, avoid it.
| Don't use this to ingest:

  .. code-block:: bash

    zipline ingest -b alpaca_api

| I changed the way we ingest new data bundles. please refer to the `Alpaca Data Bundle`_ and read iut again.

Alpaca.yaml file issues
)))))))))))))))))))))))))))))))

| When using the Alpaca bundle, you must pass credentials to the Alpaca servers. It can't be avoided.
| The easiest way to do this is by using a local file. The format is not important, I chose ``yaml``. It's simple.
| Make sure you put the alpaca credentials in the right place
| First make sure the name of the file is ``alpaca.yaml``. (avoid mistakes like using ``.yml`` postfix)
| Put it in the right location - your root **python** folder. not inside the zipline-trader folder.


SQLite file doesn't exist.
)))))))))))))))))))))))))))))))))))))

| If you happen to get this error when trying to work with data you just downloaded:

.. code-block:: py

    ValueError: SQLite file '/home/ubuntu/.zipline/data/alpaca_api/2020-12-07T02;06;17.365878/assets-7.sqlite' doesn't exist.

| It means you didn't define the ``ZIPLINE_ROOT`` correctly. You need to make sure this environment
  variable is defined for every python code you execute.


Mac OS
))))))))))

| Currently we have issues installing on Mac OS due to usage of Bcolz. I will be resolved eventually.
  If anyone from the community want to resolve this for everyone else - you are welcome.
| In the meantime - use a linux docker container to bypass that.



.. _`Alpaca Data Bundle`: ../latest/alpaca-bundle-ingestion.html