#!/bin/bash

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # .
export DIR_TSS=$DIR_CURRENT                         # .

# Add python path
export PYTHONPATH=$PYTHONPATH:$PWD                  # .

export CUDA_LAUNCH_BLOCKING=1

START_TIME="$(date -u +%s.%N)"
###########################################################################################################

echo "###########################"
echo "STARTING"
echo "###########################"

# NOTE: DETECTION
echo "*********"
echo "DETECTION"
echo "*********"
python main.py  \
    --detection  \
    --run_image  \
    --config $DIR_TSS"/configs/pbvs25_thermal_mot.yaml"

echo "###########################"
echo "ENDING"
echo "###########################"

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
