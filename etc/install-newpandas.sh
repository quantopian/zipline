#!/usr/bin/env bash
set -euvxo pipefail

echo
echo "Installing zipline using $(which python)"
echo

pip install -v -r etc/requirements_build.in -c etc/constraints_newpandas.txt
pip install -v -r .[all] -r etc/requirements_blaze.in -c etc/constraints_newpandas.txt
