#!/bin/bash

width=0.0

seed=1
echo "${seed} ${width}"
OMP_NUM_THREADS=12 ./svd.o 4 ${seed} ${width}

for Ls in 24 22 20 18 16 14 12 10 8
do
    echo "${Ls} ${seed} ${width}"
    OMP_NUM_THREADS=12 ./eig.o ${Ls} ${seed} ${width}
done

width=0.0001
for seed in 1 2 3 4
do
    echo "${seed} ${width}"
    OMP_NUM_THREADS=12 ./svd.o 4 ${seed} ${width}
    for Ls in 24 22 20 18 16 14 12 10 8
    do
        echo "${Ls} ${seed} ${width}"
        OMP_NUM_THREADS=12 ./eig.o ${Ls} ${seed} ${width}
    done
done

