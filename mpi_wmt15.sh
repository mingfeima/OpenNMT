#!/bin/sh

path=data/wmt15-de-en
lr=0.0008
optim=adam

th train.lua -data $path/wmt15-all-en-de-train.t7 -save_model $path/wmt15-all-en-de -learning_rate $lr -curriculum 1 -optim $optim -debug_print true
