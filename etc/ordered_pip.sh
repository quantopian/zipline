#!/bin/bash -e

a=0
while read line
do     
    if [[ -n "$line" && "$line" != \#* ]] ; then
        # forward to pip any args after the reqs filename
        pip install --exists-action w $line "${@:2}"
        ((a = a + 1))
    fi
done < $1
echo "$0: Final package count is $a";
