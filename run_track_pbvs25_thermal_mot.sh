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
# shellcheck disable=SC2128

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
#yaml_files=(pbvs25_thermal_mot_sort.yaml pbvs25_thermal_mot_boosttrack.yaml pbvs25_thermal_mot_diffmot.yaml pbvs25_thermal_mot_bytetrack.yaml pbvs25_thermal_mot_botsort.yaml)
yaml_files=(pbvs25_thermal_mot_bytetrack.yaml)
# shellcheck disable=SC2068
for yaml_file in ${yaml_files[@]};
do
    python main.py  \
        --tracking  \
        --run_image  \
        --config $DIR_TSS"/configs/"$yaml_file
        #        --drawing  \
done


# NOTE: TRACKING EVALUATION
echo "******************"
echo "TRACKING EVALUATION"
echo "******************"
#track_folder_results=(sort sort_backward boosttrack boosttrack_backward diffmot diffmot_backward bytetrack bytetrack_backward botsort botsort_backward)
track_folder_results=(bytetrack bytetrack_backward)
# shellcheck disable=SC2068
for track_folder_result in ${track_folder_results[@]};
do
    echo "Evaluation "$track_folder_result
    python utilities/tracking_evaluation.py  \
        --gt_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/  \
        --mot_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/tracking/$track_folder_result/  \
        --ou_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/tracking/$track_folder_result/
done

# ensemble MOT
#python utilities/tracking_ensemble.py

# evaluate ensemble MOT
#python utilities/tracking_evaluation.py  \
#  --gt_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/  \
#  --mot_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/tracking/ensemble/sort_sort_backward_boosttrack_boosttrack_backward_bytetrack_bytetrack_backward_botsort_botsort_backward/  \
#  --ou_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/tracking/ensemble/sort_sort_backward_boosttrack_boosttrack_backward_bytetrack_bytetrack_backward_botsort_botsort_backward/  \


echo "###########################"
echo "ENDING"
echo "###########################"

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
