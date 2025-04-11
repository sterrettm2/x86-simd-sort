#!/bin/bash

CXX=clang++ make test_asan

echo "Spawning 50 processes"
for i in {1..50}
do
    ( ~/sde/sde64 -spr -- ./builddir/testexe > /dev/null & )
done
