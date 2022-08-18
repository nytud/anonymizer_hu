#! /bin/bash

# This piece of code is from https://github.com/nytud/marcell_hu/blob/master/docker/freeportfinder.sh

find_free_port() {
    res=""
    for port in {10001..15000} ; do
        (echo '' >/dev/tcp/localhost/$port) >/dev/null 2>&1 || { res=$port; break; } ;
    done
    echo "$res"
}

free_port=$(find_free_port)
echo "${free_port}"
