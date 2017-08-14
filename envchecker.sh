#!/bin/sh
# Environment checked on slurm
# Usage:
#   salloc -p [partition] -N [num_of_nodes]
#   srun envchecker.sh

HOSTNAME=`hostname`
FREQ=`lscpu | grep '\bIntel' |awk '{print $9}'`
FREE_MEM=`free -m | grep Mem | awk '{print $2}'`

if [ $FREE_MEM -gt 100000 ]; then
    CACHE_MODE="flat"
else
    CACHE_MODE="cache"
fi

echo "Hostname: $HOSTNAME; Freq: $FREQ; Free $FREE_MEM MB; MCDRAM: $CACHE_MODE"
