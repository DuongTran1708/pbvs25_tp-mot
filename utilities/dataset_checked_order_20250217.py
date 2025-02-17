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
	images = images.astype(astype_dict)

	# DEBUG:
	# print("0______________________________")
	# print(images)
	# print("______________________________")

	annotations = pd.json_normalize(annotations_json["annotations"])
	annotations.columns = "ann_" + annotations.columns

	categories = pd.json_normalize(annotations_json["categories"])
	categories.columns = "cat_" + categories.columns

	# Converting this to string resolves issue #23
	categories.cat_id = categories.cat_id.astype(str)

	df = annotations

	# DEBUG:
	# print("1______________________________")
	# print(images)
	# print(df)
	# print("______________________________")

	# Converting this to string resolves issue #23
	df.ann_category_id = df.ann_category_id.astype(str)

	df[
		["ann_bbox_xmin", "ann_bbox_ymin", "ann_bbox_width", "ann_bbox_height"]
	] = pd.DataFrame(df.ann_bbox.tolist(), index=df.index)
	df.insert(8, "ann_bbox_xmax", df["ann_bbox_xmin"] + df["ann_bbox_width"])
	df.insert(10, "ann_bbox_ymax", df["ann_bbox_ymin"] + df["ann_bbox_height"])

	# DEBUG:
	# print("2______________________________")
	# print(images)
	# print(df)
	# print("______________________________")

	# debug print(df.info())

	# Join the annotions with the information about the image to add the image columns to the dataframe
	df = pd.merge(images, df, left_on="img_id", right_on="ann_image_id", how="left")
	# DEBUG:
	# print("3______________________________")
	# print(categories)
	# print(df)
	# print("______________________________")

	df = pd.merge(
		df, categories, left_on="ann_category_id", right_on="cat_id", how="left"
	)

	# DEBUG:
	# print("4______________________________")
	# print(categories)
	# print(df)
	# print("______________________________")
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

def ExportToCoco_PBVS(
		dataset,
		output_path  = None,
		cat_id_index = None
):
	"""
	Writes COCO annotation files to disk (in JSON format) and returns the path to files.

	Args:
		output_path (str):
			This is where the annotation files will be written. If not-specified then the path will be derived from the path_to_annotations and
			name properties of the dataset object.
		cat_id_index (int):
			Reindex the cat_id values so that they start from an int (usually 0 or 1) and
			then increment the cat_ids to index + number of categories continuously.
			It's useful if the cat_ids are not continuous in the original dataset.
			Some models like Yolo require starting from 0 and others like Detectron require starting from 1.

	Returns:
		A list with 1 or more paths (strings) to annotations files.

	Example:
		>>> dataset.exporter.ExportToCoco()
		['data/labels/dataset.json']

	"""
	# Copy the dataframe in the dataset so the original dataset doesn't change when you apply the export tranformations
	df = dataset.df.copy(deep=True)
	# Replace empty string values with NaN
	df = df.replace(r"^\s*$", np.nan, regex=True)
	pd.to_numeric(df["cat_id"])

	df["ann_iscrowd"] = df["ann_iscrowd"].fillna(0)

	if cat_id_index != None:
		assert isinstance(cat_id_index, int), "cat_id_index must be an int."
		_ReindexCatIds(df, cat_id_index)

	df_outputI = []
	df_outputA = []
	df_outputC = []
	list_i = []
	list_c = []
	json_list = []

	pbar = tqdm(desc="Exporting to COCO file...", total=df.shape[0])
	for i in range(0, df.shape[0]):
		images = [
			{
				"id"       : df["img_id"][i],
				"folder"   : df["img_folder"][i],
				"file_name": df["img_filename"][i],
				# "path"     : df["img_path"][i],
				"width"    : df["img_width"][i],
				"height"   : df["img_height"][i],
				# "depth"    : df["img_depth"][i],
			}
		]

		# DEBUG:
		# print("______________________________")
		# print(df)
		# print(pd.isna(df["cat_id"][i]))
		# print("______________________________")

		# SUGAR:
		annotations = None
		# Skip this if cat_id is na
		if not pd.isna(df["cat_id"][i]):
			# print(df["ann_score"][i])
			# sys.exit()

			annotations = [
				{
					"id"       : int(df["ann_id"][i]) if df.isnan(df["ann_id"][i]) else df.index[i],
					"image_id" : df["img_id"][i],
					"segmented": df["ann_segmented"][i],
					"bbox": [
						df["ann_bbox_xmin"][i],
						df["ann_bbox_ymin"][i],
						df["ann_bbox_width"][i],
						df["ann_bbox_height"][i],
					],
					"area"        : df["ann_area"][i],
					"segmentation": df["ann_segmentation"][i],
					"iscrowd"     : df["ann_iscrowd"][i],
					"score"       : df["ann_score"][i] if not math.isnan(df["ann_score"][i]) else float(random.randint(60, 90) / 100),
					"pose"        : df["ann_pose"][i],
					"truncated"   : df["ann_truncated"][i],
					"category_id" : int(df["cat_id"][i]),
					"difficult"   : df["ann_difficult"][i],
				}
			]

			# include keypoints, if available
			if "ann_keypoints" in df.keys() and (not np.isnan(df["ann_keypoints"][i]).all()):
				keypoints = df["ann_keypoints"][i]
				if isinstance(keypoints, list):
					n_keypoints = int(len(keypoints) / 3)  # 3 numbers per keypoint: x,y,visibility
				elif isinstance(keypoints, np.ndarray):
					n_keypoints = int(keypoints.size / 3)  # 3 numbers per keypoint: x,y,visibility
				else:
					raise TypeError('The keypoints array is expected to be either a list or a numpy array')
				annotations[0]["num_keypoints"] = n_keypoints
				annotations[0]["keypoints"] = keypoints
			else:
				pass

			categories = [
				{
					"id": int(df["cat_id"][i]),
					"name": df["cat_name"][i],
					"supercategory": df["cat_supercategory"][i],
				}
			]

			# Check if the list is empty
			if list_c:
				if categories[0]["id"] in list_c:
					pass
				else:
					categories[0]["id"] = int(categories[0]["id"])
					df_outputC.append(pd.DataFrame([categories]))
			elif not pd.isna(categories[0]["id"]):
				categories[0]["id"] = int(categories[0]["id"])
				df_outputC.append(pd.DataFrame([categories]))
			else:
				pass
			list_c.append(categories[0]["id"])

		if list_i:
			if images[0]["id"] in list_i or np.isnan(images[0]["id"]):
				pass
			else:
				df_outputI.append(pd.DataFrame([images]))
		elif ~np.isnan(images[0]["id"]):
			df_outputI.append(pd.DataFrame([images]))
		else:
			pass
		list_i.append(images[0]["id"])

		# If the class id is blank, then there is no annotation to add
		if annotations is not None and not pd.isna(categories[0]["id"]):
			df_outputA.append(pd.DataFrame([annotations]))

		pbar.update()

	mergedI = pd.concat(df_outputI, ignore_index=True)
	mergedA = pd.concat(df_outputA, ignore_index=True)
	mergedC = pd.concat(df_outputC, ignore_index=True)

	resultI = mergedI[0].to_json(orient="split", default_handler=str)
	resultA = mergedA[0].to_json(orient="split", default_handler=str)
	resultC = mergedC[0].to_json(orient="split", default_handler=str)

	parsedI = json.loads(resultI)
	del parsedI["index"]
	del parsedI["name"]
	parsedI["images"] = parsedI["data"]
	del parsedI["data"]

	parsedA = json.loads(resultA)
	del parsedA["index"]
	del parsedA["name"]
	parsedA["annotations"] = parsedA["data"]
	del parsedA["data"]

	parsedC = json.loads(resultC)
	del parsedC["index"]
	del parsedC["name"]
	parsedC["categories"] = parsedC["data"]
	del parsedC["data"]

	parsedI.update(parsedA)
	parsedI.update(parsedC)
	json_output = parsedI

	if output_path == None:
		output_path = Path(
			dataset.path_to_annotations, (dataset.name + ".json")
		)

	with open(output_path, "w") as outfile:
		json.dump(obj=json_output, fp=outfile, indent=4)
	return [str(output_path)]

def checked_and_changed_copyed_dataset():
	'''
	Because the name of images is not in order, so we need to check the dataset and change the name of the images.
	The name of image must be in order, so we can use it for tracking.
	We also need to copy the images into the new folder.

	Returns:

	'''
	# Init file
	split_types    = ["train","val"]
	for split_type in split_types:
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
		for seq in tqdm(list_seqs):
			# DEBUG:
			# if seq not in "seq62":
			# 	continue

			# get sequence index
			seq_index = int(''.join(i for i in seq if i.isdigit()))

			# get path file
			file_json_in        = os.path.join(folder_lbl_in, seq, 'thermal/COCO/annotations.json')
			dir_thermal_img_in  = os.path.join(folder_img_in, seq, 'thermal')
			dir_rgb_img_in      = os.path.join(folder_img_in, seq, 'rgb')

			# define new path
			file_json_ou        = os.path.join(folder_lbl_ou, seq, 'thermal/COCO/annotations.json')
			dir_thermal_img_ou  = os.path.join(folder_img_ou, seq, 'thermal')
			dir_rgb_img_ou      = os.path.join(folder_img_ou, seq, 'rgb')

			# create folder
			os.makedirs(os.path.dirname(file_json_ou), exist_ok=True)
			os.makedirs(dir_thermal_img_ou, exist_ok=True)
			os.makedirs(dir_rgb_img_ou, exist_ok=True)

			# load dataset from coco datasets
			dataset_ori = ImportCoco_PBVS(file_json_in, path_to_images=dir_thermal_img_in, name="pbvs_coco")
			dataset_new = deepcopy(dataset_ori)

			# current file image name
			img_filename_cur = ""

			# checked dataset via json file
			for index in tqdm(dataset_ori.df.index, desc=f"Checking {seq}"):
				# change into new value
				dataset_new.df.loc[index, 'img_folder'] = dir_thermal_img_ou

				# change file name
				img_ext      = os.path.splitext(dataset_ori.df.loc[index, 'img_filename'])[1]
				img_name_new = f"{seq_index:05d}{dataset_ori.df.loc[index, 'img_id']:08d}{img_ext}"
				dataset_new.df.loc[index, 'img_filename'] = img_name_new

				# copy image, only if the image name is different, mean we do not copy many times
				if img_filename_cur is not img_name_new:
					img_filename_cur = img_name_new

					# copy thermal image
					shutil.copyfile(
						os.path.join(dir_thermal_img_in, dataset_ori.df.loc[index, 'img_filename']),
						os.path.join(dir_thermal_img_ou, dataset_new.df.loc[index, 'img_filename']),
					)

					# copy rgb image
					shutil.copyfile(
						os.path.join(dir_rgb_img_in, dataset_ori.df.loc[index, 'img_filename']),
						os.path.join(dir_rgb_img_ou, dataset_new.df.loc[index, 'img_filename']),
					)

				# DEBUG:
				# print(f"{dataset_ori.df.loc[index, 'img_folder']}")
				# print(f"{dataset_new.df.loc[index, 'img_folder']}")
				#
				# print(f"\n {dataset_ori.df.loc[index]}\n")
				# print(f"\n {dataset_new.df.loc[index]}\n")

				pass
				# DEBUG:
				# sys.exit()


			# export checked json
			print(ExportToCoco_PBVS(
				dataset      = dataset_new,
				output_path  = file_json_ou,
				cat_id_index = 1
			))

			# DEBUG:
			# break


def main():
	checked_and_changed_copyed_dataset()


if __name__ == "__main__":
	main()