#!/bin/bash

if [[ "$RUNNER_OS" == "macOS" ]]; then
    # On OSX, sed refuses to edit in place, so give it an empty extension for the rename.
    function sed_inplace() {
        sed -i '' "$@"
    }
else
    function sed_inplace() {
        sed -i "$@"
    }
fi

sed_inplace "s/numpy==.*/numpy==$NUMPY_VERSION/" etc/requirements.txt
sed_inplace "s/pandas==.*/pandas==$PANDAS_VERSION/" etc/requirements.txt
sed_inplace "s/scipy==.*/scipy==$SCIPY_VERSION/" etc/requirements.txt
if [ -n matrix.pandas-datareader-version ]; then
    export PANDAS_DATAREADER_VERSION=${{ matrix.pandas-datareader-version }}
    sed_inplace "s/pandas-datareader==.*/pandas-datareader==$PANDAS_DATAREADER_VERSION/" etc/requirements.txt
fi
if [ -n matrix.dask-version ]; then
    export DASK_VERSION=${{ matrix.dask-version }}
    sed_inplace "s/dask\[dataframe\]==.*/dask\[dataframe\]==$DASK_VERSION/" etc/requirements.txt
fi
