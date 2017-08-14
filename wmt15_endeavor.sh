#!/bin/sh

export I_MPI_FABRICS=shm:tmi
export I_MPI_TMI_PROVIDER=psm2
export I_MPI_FALLBACK=0
export PSM2_IDENTIFY=1
export OMP_NUM_THREADS=72
export I_MPI_DEBUG=2

path=data/wmt15-de-en
nparallel=4
lr=0.0032
optim=adam
log=wmt15_np${nparallel}_lr${lr}.txt

mpirun -n 32 th train.lua -data $path/wmt15-all-en-de-train.t7 -save_model $path/wmt15-all-en-de -nparallel $nparallel -learning_rate $lr -curriculum 1 -optim $optim -save_every_epochs 0 -start_decay_at 3

