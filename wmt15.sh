#!/bin/sh

path=data/wmt15-de-en
nparallel=4
lr=0.0032
optim=adam
log=wmt15_np${nparallel}_lr${lr}.txt

th train.lua -data $path/wmt15-all-en-de-train.t7 -save_model $path/wmt15-all-en-de -nparallel $nparallel -learning_rate $lr -curriculum 1 -optim $optim -save_every_epochs 0 -start_decay_at 3 | tee $log

