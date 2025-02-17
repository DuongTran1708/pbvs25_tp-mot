import math
import os
import re
import sys
import glob

from copy import deepcopy

import json
import random
import shutil

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from os.path import exists
from pathlib import Path, PurePath

from pylabel.shared import _ReindexCatIds
from tqdm import tqdm
# from pylabel import importer

# from pylabel.shared import schema
from pylabel.dataset import Dataset
from pylabel.exporter import Export


# Get from pylabel.shared
schema = [
	"img_folder",
	"img_filename",
	"img_path",
	"img_id",
	"img_width",
	"img_height",
	"img_depth",
	"ann_id", # SUGAR: add id in schema so the dataset does not drop it in format
	"ann_segmented",
	"ann_bbox_xmin",
	"ann_bbox_ymin",
	"ann_bbox_xmax",
	"ann_bbox_ymax",
	"ann_bbox_width",
	"ann_bbox_height",
	"ann_area",
	"ann_segmentation",
	"ann_iscrowd",
	"ann_track_id",  # SUGAR: add track id in schema so the dataset does not drop it in format
	"ann_score",    # SUGAR: add score for detection in schema so the dataset does not drop it in format
	"ann_keypoints",
	"ann_pose",
	"ann_truncated",
	"ann_difficult",
	"cat_id",
	"cat_name",
	"cat_supercategory",
	"split",
	"annotated",
]


classes = [
	"person" # on PBVS dataset, there is only one class
]


def checked_and_changed_copyed_dataset():
	'''
	Because the name of images is not in order, so we need to check the dataset and change the name of the images.
	The name of image must be in order, so we can use it for tracking.
	We also need to copy the images into the new folder.

	Returns:

	'''
	# Init file
	split_types = ["train","val"]
	img_types   = ["thermal","rgb"]
	for img_type in tqdm(img_types):
		for split_type in tqdm(split_types, desc=f"Image type: {img_type}"):
			folder_lbl_in = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/annotations/{split_type}/"
			folder_img_in = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/images/{split_type}/"
			folder_lbl_ou = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/{split_type}/"
			folder_img_ou = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/images/{split_type}/"

			# create folder
			os.makedirs(folder_lbl_ou, exist_ok=True)
			os.makedirs(folder_img_ou, exist_ok=True)

			# get list folder
			list_seqs = os.listdir(folder_lbl_in)

			# loop to export dataset
			for seq in tqdm(list_seqs, desc=f"Split type: {split_type}"):
				# DEBUG:
				# if seq not in "seq62":
				# 	continue

				# get sequence index
				seq_index = int(''.join(i for i in seq if i.isdigit()))

				# get path file
				file_json_in = os.path.join(folder_lbl_in, seq, f'{img_type}/COCO/annotations.json')
				dir_img_in   = os.path.join(folder_img_in, seq, f'{img_type}')

				# define new path
				file_json_ou = os.path.join(folder_lbl_ou, seq, f'{img_type}/COCO/annotations.json')
				dir_img_ou   = os.path.join(folder_img_ou, seq, f'{img_type}')

				# create folder
				os.makedirs(os.path.dirname(file_json_ou), exist_ok=True)
				os.makedirs(dir_img_ou, exist_ok=True)

				# load dataset from coco datasets
				dataset_ori_json = json.load(open(file_json_in))
				dataset_new_json = deepcopy(dataset_ori_json)

				# change name of images and copy images
				for img in dataset_new_json["images"]:
					img["file_name"] = f"{int(seq_index):06d}{int(img['id']):08d}{os.path.splitext(img['file_name'])[1]}"

				# copy images
				for img_new in tqdm(dataset_new_json["images"], desc=f"Copy images: {seq}"):
					for img_ori in dataset_ori_json["images"]:
						if int(img_new["id"]) == int(img_ori["id"]):
							shutil.copyfile(
								os.path.join(dir_img_in, img_ori["file_name"]),
								os.path.join(dir_img_ou, img_new["file_name"]),
							)
							break

				# write new json file
				json.dump(dataset_new_json, open(file_json_ou, 'w'), indent=4)

				# DEBUG:
				# break


def main():
	checked_and_changed_copyed_dataset()


if __name__ == "__main__":
	main()