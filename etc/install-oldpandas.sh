#!/usr/bin/env bash
set -euvxo pipefail

echo
echo "Installing zipline using $(which python)"
echo

python -m pip install -v -r etc/requirements_build.in -c etc/constraints_oldpandas.txt
python -m pip install -v "$@"
