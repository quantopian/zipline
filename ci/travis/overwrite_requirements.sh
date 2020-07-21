#!/bin/bash

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    # On OSX, sed refuses to edit in place, so give it an empty extension for the rename.
    function sed_inplace() {
        sed -i '' "$@"
    }
else
    function sed_inplace() {
        sed -i "$@"
    }
fi

sed_inplace "s/numpy==.*/numpy==$NUMPY_VERSION/" etc/requirements_locked.txt
sed_inplace "s/pandas==.*/pandas==$PANDAS_VERSION/" etc/requirements_locked.txt
sed_inplace "s/scipy==.*/scipy==$SCIPY_VERSION/" etc/requirements_locked.txt
if [ -n "$STATSMODELS_VERSION" ]; then
    sed_inplace "s/statsmodels==.*/statsmodels==$STATSMODELS_VERSION/" etc/requirements_locked.txt
fi
if [ -n "$PANDAS_DATAREADER_VERSION" ]; then
    sed_inplace "s/pandas-datareader==.*/pandas-datareader==$PANDAS_DATAREADER_VERSION/" etc/requirements_locked.txt
fi
if [ -n "$DASK_VERSION" ]; then
    sed_inplace "s/dask\[dataframe\]==.*/dask\[dataframe\]==$DASK_VERSION/" etc/requirements_locked.txt
fi
