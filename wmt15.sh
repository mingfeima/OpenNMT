#!/bin/sh

path=data/wmt15-de-en
nparallel=8
lr=0.0016
optim=adam

log=wmt15_np${nparallel}_lr${lr}.txt


OMP_NUM_THREADS=44 th train.lua -data $path/wmt15-all-en-de-train.t7 -save_model $path/wmt15-all-en-de -nparallel $nparallel -learning_rate $lr -curriculum 1 -optim $optim -start_decay_at 1  #-start_decay_ppl_delta 1

