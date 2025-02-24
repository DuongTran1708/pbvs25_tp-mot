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

# NOTE: DETECTION PROCESS
#echo "*****************"
#echo "DETECTION PROCESS"
#echo "*****************"
#python main.py  \
#    --detection  \
#    --run_image  \
#    --drawing  \
#    --config $DIR_TSS"/configs/pbvs25_thermal_mot_sort.yaml"

# NOTE: DETECTION EVALUATION
#echo "********************"
#echo "DETECTION EVALUATION"
#echo "********************"
#python utilities/detection_evaluation.py  \
#  --img_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/images/val/  \
#  --gt_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/  \
#  --det_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/detection/yolov8s_1600_imgz_1600/  \
#  --ou_file /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/detection/yolov8s_1600_imgz_1600/object_detection_result.txt  \

# NOTE: TRACKING PROCESS
echo "****************"
echo "TRACKING PROCESS"
echo "****************"
python main.py  \
    --tracking  \
    --run_image  \
    --drawing  \
    --config $DIR_TSS"/configs/pbvs25_thermal_mot.yaml"

# NOTE: TRACKING EVALUATION
#echo "******************"
#echo "TRACKING EVALUATION"
#echo "******************"
#python utilities/tracking_evaluation.py  \
#  --gt_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/  \
#  --mot_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/tracking/sort/  \
#  --ou_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/tracking/sort/  \

# ensemble MOT
#python utilities/tracking_ensemble.py

# evaluate ensemble MOT
#python utilities/tracking_evaluation.py  \
#  --gt_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/  \
#  --mot_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/tracking/ensemble/sort_boosttrack/  \
#  --ou_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/tracking/ensemble/sort_boosttrack/  \


echo "###########################"
echo "ENDING"
echo "###########################"

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
