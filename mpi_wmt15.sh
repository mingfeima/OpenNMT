#!/bin/sh

path=data/wmt15-de-en
nodes=2
lr=0.0008
optim=adam

mpirun -n $nodes th train.lua -data $path/wmt15-all-en-de-train.t7 -save_model $path/wmt15-all-en-de -learning_rate $lr -curriculum 1 -optim $optim #-start_decay_ppl_delta 1

