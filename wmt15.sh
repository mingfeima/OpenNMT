#!/bin/sh

path=data/wmt15-de-en
nparallel=4
lr=0.006

optim=adam

log=wmt15_np${nparallel}_lr${lr}.txt


OMP_NUM_THREADS=72 th train.lua -data $path/wmt15-all-en-de-train.t7 -save_model $path/wmt15-all-en-de -nparallel $nparallel -learning_rate $lr -curriculum 1 -optim $optim #-start_decay_ppl_delta 1

