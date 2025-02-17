import os
import sys
import glob

import json
import random
import shutil

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from os.path import exists
from pathlib import Path, PurePath

from tqdm import tqdm
# from pylabel import importer

# from pylabel.shared import schema
from pylabel.dataset import Dataset
from pylabel.exporter import Export

# sys.path.append(".")  # add ROOT to PATH

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
	"person"
]

# get from pylabel
def ImportCoco_PBVS(path, path_to_images=None, name=None, encoding="utf-8"):
	"""
	This function takes the path to a JSON file in COCO format as input. It returns a PyLabel dataset object that contains the annotations.

	Returns:
		PyLabel dataset object.

	Args:
		path (str):The path to the JSON file with the COCO annotations.
		path_to_images (str): The path to the images relative to the json file.
			If the images are in the same directory as the JSON file then omit this parameter.
			If the images are in a different directory on the same level as the annotations then you would
			set `path_to_images='../images/'`
		name (str): This will set the dataset.name property for this dataset.
			If not specified, the filename (without extension) of the COCO annotation file file will be used as the dataset name.
		encoding (str): Default is 'utf-8. Encoding of the annotations file(s).
	Example:
		>>> from pylabel import importer
		>>> dataset = importer.ImportCoco("coco_annotations.json")
	"""
	with open(path, encoding=encoding) as cocojson:
		annotations_json = json.load(cocojson)

	# Store the 3 sections of the json as seperate json arrays
	images = pd.json_normalize(annotations_json["images"])
	images.columns = "img_" + images.columns
	try:
		images["img_folder"]
	except:
		images["img_folder"] = ""
	# print(images)

	# If the user has specified a different image folder then use that one
	if path_to_images != None:
		images["img_folder"] = path_to_images

	astype_dict = {"img_width": "int64", "img_height": "int64", "img_depth": "int64"}
	astype_keys = list(astype_dict.keys())
	for element in astype_keys:
		if element not in images.columns:
			astype_dict.pop(element)
	# print(astype_dict)
	# images = images.astype({'img_width': 'int64','img_height': 'int64','img_depth': 'int64'})

	# DEBUG:
	# print(images)

	images = images.astype(astype_dict)

	annotations = pd.json_normalize(annotations_json["annotations"])
	annotations.columns = "ann_" + annotations.columns

	categories = pd.json_normalize(annotations_json["categories"])
	categories.columns = "cat_" + categories.columns

	# Converting this to string resolves issue #23
	categories.cat_id = categories.cat_id.astype(str)

	df = annotations

	# Converting this to string resolves issue #23
	df.ann_category_id = df.ann_category_id.astype(str)

	df[
		["ann_bbox_xmin", "ann_bbox_ymin", "ann_bbox_width", "ann_bbox_height"]
	] = pd.DataFrame(df.ann_bbox.tolist(), index=df.index)
	df.insert(8, "ann_bbox_xmax", df["ann_bbox_xmin"] + df["ann_bbox_width"])
	df.insert(10, "ann_bbox_ymax", df["ann_bbox_ymin"] + df["ann_bbox_height"])

	# debug print(df.info())
	# DEBUG:
	# print(df.info())
	# print(categories.info())

	# Join the annotions with the information about the image to add the image columns to the dataframe
	df = pd.merge(images, df, left_on="img_id", right_on="ann_image_id", how="left")
	df = pd.merge(
		df, categories, left_on="ann_category_id", right_on="cat_id", how="left"
	)

	# DEBUG:
	# print(df.columns)
	# print(df.info())
	# sys.exit()

	# Rename columns if needed from the coco column name to the pylabel column name
	df.rename(columns={"img_file_name": "img_filename"}, inplace=True)

	# Drop columns that are not in the schema
	df = df[df.columns.intersection(schema)]

	# Add missing columns that are in the schema but not part of the table
	df[list(set(schema) - set(df.columns))] = ""

	# Reorder columns
	df = df[schema]
	df.index.name = "id"
	df.annotated = 1

	# Fill na values with empty strings which resolved some errors when
	# working with images that don't have any annotations
	df.fillna("", inplace=True)

	# These should be strings
	df.cat_id = df.cat_id.astype(str)

	# These should be integers
	df.img_width = df.img_width.astype(int)
	df.img_height = df.img_height.astype(int)

	dataset = Dataset(df)

	# Assign the filename (without extension) as the name of the dataset
	if name == None:
		dataset.name = Path(path).stem
	else:
		dataset.name = name

	dataset.path_to_annotations = PurePath(path).parent

	return dataset


def convert_all_folder_coco_to_yolo():
	split_types    = ["train","val"]
	for split_type in split_types:
		# Init file
		path_folder_lbl_in = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/{split_type}/"
		path_folder_img_in = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/images/{split_type}/"

		# get list folder
		list_seqs = os.listdir(path_folder_lbl_in)

		# loop to export dataset
		for seq in tqdm(list_seqs):
			# get path file
			path_file_in      = os.path.join(path_folder_lbl_in, seq, 'thermal/COCO/annotations.json')
			path_dir_ou       = os.path.join(path_folder_lbl_in, seq, 'thermal/yolo')
			path_dir_img      = os.path.join(path_folder_img_in, seq, 'thermal')

			# create folder
			os.makedirs(path_dir_ou, exist_ok=True)

			# load dataset
			dataset = ImportCoco_PBVS(path_file_in, path_to_images=path_dir_img, name="pbvs_coco")

			# export txts
			dataset.export.ExportToYoloV5(
				output_path   = path_dir_ou,
				cat_id_index = 0  # 0: cat_id
			)[0]

			# DEBUG:
			# print(dataset.df.columns)
			# break


def convert_coco_to_mot(seq, path_file_in, path_file_ou, path_dir_img):
	# load dataset
	dataset = ImportCoco_PBVS(path_file_in, path_to_images=path_dir_img, name="pbvs_coco")

	# DEBUG:
	# print(dataset.df.columns)
	# print(len(dataset.df.index))
	# sys.exit()

	# Convert empty bbox coordinates to nan to avoid math errors
	# If an image has no annotations then an empty label file will be created
	dataset.df.ann_bbox_xmin = dataset.df.ann_bbox_xmin.replace(
		r"^\s*$", np.nan, regex=True
	)
	dataset.df.ann_bbox_ymin = dataset.df.ann_bbox_ymin.replace(
		r"^\s*$", np.nan, regex=True
	)
	dataset.df.ann_bbox_width = dataset.df.ann_bbox_width.replace(
		r"^\s*$", np.nan, regex=True
	)
	dataset.df.ann_bbox_height = dataset.df.ann_bbox_height.replace(
		r"^\s*$", np.nan, regex=True
	)

	# write out file
	img_filename = ""
	img_index    = 0
	with open(path_file_ou, "w") as file_write:
		for index in tqdm(dataset.df.index, desc=f"{seq}"):
			# get row
			row = dataset.df.iloc[index]

			if row.img_filename != img_filename:
				img_filename = row.img_filename
				img_index    = img_index + 1

			# '{frame},{id},{x1},{y1},{w},{h},{s},{label},-1,-1\n'
			str_out = (f"{img_index},"
			           f"{row.ann_track_id},"
			           f"{round(row.ann_bbox_xmin, 1)},"
			           f"{round(row.ann_bbox_ymin, 1)},"
			           f"{round(row.ann_bbox_width, 1)},"
			           f"{round(row.ann_bbox_height, 1)},"
			           f"1.0,"
			           f"{row.cat_id},"  # print ID of class
			           f"-1,-1\n")

			# write out
			file_write.write(str_out)


def convert_all_format_coco_to_mot():
	split_types    = ["train","val"]
	for split_type in split_types:
		# Init file
		path_folder_lbl_in = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/{split_type}/"
		path_folder_img_in = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/images/{split_type}/"

		# get list folder
		list_seqs = os.listdir(path_folder_lbl_in)

		# loop to export dataset
		for seq in tqdm(list_seqs):
			# get path file
			path_file_in      = os.path.join(path_folder_lbl_in, seq, 'thermal/COCO/annotations.json')
			# path_file_ou      = os.path.join(path_folder_lbl_in, seq, 'thermal/annotations_mot.txt')
			path_file_ou      = os.path.join(path_folder_lbl_in, seq, f'thermal/{seq}_thermal.txt')
			path_dir_img      = os.path.join(path_folder_img_in, seq, 'thermal')

			# '{frame},{id},{x1},{y1},{w},{h},{s},{label},-1,-1\n'
			# export txts
			convert_coco_to_mot(seq, path_file_in, path_file_ou, path_dir_img)


def copy_yolo_dataset_for_training():
	split_types    = ["train","val"]
	for split_type in split_types:
		# Init file
		path_folder_lbl_in  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/{split_type}/"
		path_folder_img_in  = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/images/{split_type}/"
		path_folder_yolo_ou = f"/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/yolo_format/"

		# create folder
		path_folder_lbl_yolo_ou = os.path.join(path_folder_yolo_ou, f"{split_type}/labels")
		path_folder_img_yolo_ou = os.path.join(path_folder_yolo_ou, f"{split_type}/images")
		os.makedirs(path_folder_lbl_yolo_ou, exist_ok=True)
		os.makedirs(path_folder_img_yolo_ou, exist_ok=True)

		# get list folder
		list_seqs = os.listdir(path_folder_lbl_in)

		# loop to export dataset
		for seq in tqdm(list_seqs):
			# get path file
			path_dir_lbl_in = os.path.join(path_folder_lbl_in, seq, 'thermal/yolo')
			path_dir_img_in = os.path.join(path_folder_img_in, seq, 'thermal')

			# get list file
			list_files = os.listdir(path_dir_lbl_in)

			# loop to export dataset
			for file in list_files:
				# get path file
				path_file_lbl_in = os.path.join(path_dir_lbl_in, file)
				path_file_img_in = os.path.join(path_dir_img_in, file.replace('.txt', '.png'))

				# check file exist
				if not exists(path_file_img_in):
					continue

				# get path file out
				path_file_lbl_ou = os.path.join(path_folder_lbl_yolo_ou, file)
				path_file_img_ou = os.path.join(path_folder_img_yolo_ou, file.replace('.txt', '.png'))

				# copy file
				shutil.copyfile(path_file_lbl_in, path_file_lbl_ou)
				shutil.copyfile(path_file_img_in, path_file_img_ou)


def main():
	# convert_all_folder_coco_to_yolo()

	# convert_all_format_coco_to_mot()

	copy_yolo_dataset_for_training()
	pass


if __name__ == "__main__":
	main()