#!/bin/bash
if [ ! -d highway ]; then
    git clone "https://github.com/google/highway.git"
else
    # if it exists, just update it
    cd highway
    git fetch origin
    git pull origin master
    cd ..
fi