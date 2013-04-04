#!/bin/bash -e

a=0
while read line
do     
    if [[ -n "$line" && "$line" != \#* ]] ; then
        pip install --use-mirrors $line
    fi
    ((a = a + 1))
done < $1
echo "$0: Final package count is $a";
