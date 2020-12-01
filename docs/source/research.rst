Research & backtesting in the Notebook environment
===================================================
| To run your research environment you first need to make sure jupyter is installed.
| Follow the instructions in Jupyter.org_

.. code-block::

    e.g. pip install notebook

| Start your Jupyter server

.. code-block::

    jupyter notebook

Working With The Research Environment
---------------------------------
| This was one of Quantopian's strengths and now you could run it locally too.
| In the next few examples we will see how to:
* Load your Alpaca (or any other) data bundle
* How to get pricing data from the bundle
* How to create and run a pipeline
* How tu run a backtest INSIDE the notebook (using python files will follow)
* How to analyze your results with pyfolio (for that you will need to install `pyfolio`_)


Loading Your Data Bundle
-----------------------------
| Now that you have a jupyter notebook running you could load your previously ingested data bundle.
| Follow this notebook for a usage example: `Load Data Bundle`_.

.. _Load Data Bundle: notebooks/LoadDataBundle.ipynb

.. _`Jupyter.org` : https://jupyter.org/install

Simple Pipeline
--------------------------
| You can work with pipeline just as it was on Quantopian, and in the following example
  you could see hwo to create a simple pipeline and get the data:  `Simple Pipeline`_.

.. _Simple Pipeline: notebooks/SimplePipeline.ipynb


Run and analyze a backtest
--------------------------
| Running a backtest is the way to test your ideas. You could do it inside a notebook
  or in your python IDE (your choice).
| The advantage of using the notebook is the ability
  to use Pyfolio to analyze the results in a simple manner as could be seen here: `Bactesting`_.

.. _Bactesting: notebooks/backtest.ipynb


.. _`pyfolio` : https://github.com/quantopian/pyfolio