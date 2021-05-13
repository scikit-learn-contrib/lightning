#!/bin/bash

# download a file with either wget or curl

if [ "$1foo" = "foo" ]; then
    echo "usage: `basename $0` url"
    exit 1
fi

wget_path=`which wget`

if [ $? = 0 ]; then
    $wget_path $1
    exit $?
fi

curl_path=`which curl`

if [ $? = 0 ]; then
    $curl_path -O $1
    exit $?
fi

