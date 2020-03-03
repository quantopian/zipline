#!/usr/bin/env bash
set -euvxo pipefail

echo
echo "Installing zipline using $(which python)"
echo

pip install -r etc/requirements_build.in -c etc/constraints_newpandas.txt
pip install -r .[all] -r etc/requirements_blaze.in -c etc/constraints_newpandas.txt
