import json
import math
import os
import copy
import random
import sys
from typing import List
from os.path import exists
from pathlib import PurePath, Path

import pandas as pd
import numpy as np
import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# from pylabel import importer
from pylabel.shared import _ReindexCatIds
from pylabel.dataset import Dataset
from tqdm import tqdm


# Get from pylabel.shared
schema = [
	"img_folder",
	"img_filename",
	"img_path",
	"img_id",
	"img_width",
	"img_height",
	"img_depth",
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

def ImportYoloV5_PBVS(
		path,
		img_ext        = "jpg,jpeg,png,webp",
		cat_names      = [],
		path_to_images = "",
		name           = "dataset",
		encoding       = "utf-8",
):
	"""
	Provide the path a directory with annotations in YOLO format and it returns a PyLabel dataset object that contains the annotations.
	The Yolo format does not store much information about the images, such as the height and width. When you import a
	Yolo dataset PyLabel will extract this information from the images.

	Returns:
		PyLabel dataset object.

	Args:
		path (str): The path to the directory with the annotations in YOLO format.
		img_ext (str, comma separated): Specify the file extension(s) of the images used in your dataset:
		 .jpeg, .png, etc. This is required because the YOLO format does not store the filename of the images.
		 It could be any of the image formats supported by YoloV5. PyLabel will iterate through the file extensions
		 specified until it finds a match.
		cat_names (list): YOLO annotations only store a class number, not the name. You can provide a list of class ids
			that correspond to the int used to represent that class in the annotations. For example `['Squirrel,'Nut']`.
			If you have the class names already stored in a YOLO YAML file then use the ImportYoloV5WithYaml method to
			automatically read the class names from that file.
		path_to_images (str): The path to the images relative to the annotations.
			If the images are in the same directory as the annotation files then omit this parameter.
			If the images are in a different directory on the same level as the annotations then you would
			set `path_to_images='../images/'`
		name (str): Default is 'dataset'. This will set the dataset.name property for this dataset.
		encoding (str): Default is 'utf-8. Encoding of the annotations file(s).

	Example:
		>>> from pylabel import importer
		>>> dataset = importer.ImportYoloV5(path="labels/", path_to_images="../images/")
	"""

	def GetCatNameFromId(cat_id, cat_names):
		cat_id = int(cat_id)
		if len(cat_names) > int(cat_id):
			return cat_names[cat_id]

	# Create an empty dataframe
	df = pd.DataFrame(columns=schema)

	# the dictionary to pass to pandas dataframe
	d = {}

	row_id = 0
	img_id = 0

	# iterate over files in that directory
	pbar = tqdm(desc="Importing YOLO files...", total=len(os.listdir(path)))
	for filename in os.scandir(path):
		if filename.is_file() and filename.name.endswith(".txt"):
			filepath = filename.path
			file = open(filepath, "r", encoding=encoding)  # Read file
			row = {}

			# First find the image files and extract the metadata about the image
			row["img_folder"] = path_to_images

			# Figure out what the extension is of the corresponding image file
			# by looping through the extension in the img_ext parameter
			found_image = False
			for ext in img_ext.split(","):
				image_filename = filename.name.replace("txt", ext)

				# Get the path to the image file to extract the height, width, and depth
				image_path = PurePath(path, path_to_images, image_filename)
				if exists(image_path):
					found_image = True
					break

			# Check if there is a file at this location.
			assert (
					found_image == True
			), f"No image file found: {image_path}. Check path_to_images and img_ext arguments."

			row["img_filename"] = image_filename

			imgstream = open(str(image_path), "rb")
			imgbytes = bytearray(imgstream.read())
			numpyarray = np.asarray(imgbytes, dtype=np.uint8)

			im = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

			img_height = im.shape[0]
			img_width = im.shape[1]
			# If the image is grayscale then there is no img_depth
			if len(im.shape) == 2:
				img_depth = 1
			else:
				img_depth = im.shape[2]  # 3 for color images

			row["img_id"] = img_id
			row["img_width"] = img_width
			row["img_height"] = img_height
			row["img_depth"] = img_depth

			# Read the annotation in the file
			# Check if the file has at least one line:
			numlines = len(open(filepath, encoding=encoding).readlines())
			if numlines == 0:
				# Create a row without annotations
				d[row_id] = row
				row_id += 1
			else:
				for line in file:
					line = line.strip()

					# check if the row is empty, leave annotation columns blank
					if line:
						d[row_id] = copy.deepcopy(row)
						(
							cat_id,
							x_center_norm,
							y_center_norm,
							width_norm,
							height_norm,
						) = line.split()

						row["ann_bbox_width"] = float(width_norm) * img_width
						row["ann_bbox_height"] = float(height_norm) * img_height
						row["ann_bbox_xmin"] = float(x_center_norm) * img_width - (
							(row["ann_bbox_width"] / 2)
						)
						row["ann_bbox_ymax"] = float(y_center_norm) * img_height + (
							(row["ann_bbox_height"] / 2)
						)
						row["ann_bbox_xmax"] = (
								row["ann_bbox_xmin"] + row["ann_bbox_width"]
						)
						row["ann_bbox_ymin"] = (
								row["ann_bbox_ymax"] - row["ann_bbox_height"]
						)

						row["ann_area"] = row["ann_bbox_width"] * row["ann_bbox_height"]

						row["cat_id"] = cat_id
						row["cat_name"] = GetCatNameFromId(cat_id, cat_names)

						d[row_id] = dict(row)
						row_id += 1
						# Copy the image data to use for the next row
					else:
						# Create a row without annotations
						d[row_id] = row
						row_id += 1

				# Add this row to the dict
		# increment the image id
		img_id += 1
		pbar.update()

	df = pd.DataFrame.from_dict(d, "index", columns=schema)
	df.index.name = "id"
	df.annotated = 1
	df.fillna("", inplace=True)

	# These should be strings
	df.cat_id = df.cat_id.astype(str)

	# These should be integers
	df.img_width = df.img_width.astype(int)
	df.img_height = df.img_height.astype(int)

	# Reorder columns
	dataset = Dataset(df)
	dataset.name = name
	dataset.path_to_annotations = path

	return dataset


def ExportToCoco_PBVS(dataset, output_path=None, cat_id_index=None):
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
				"id": df["img_id"][i],
				"folder": df["img_folder"][i],
				"file_name": df["img_filename"][i],
				"path": df["img_path"][i],
				"width": df["img_width"][i],
				"height": df["img_height"][i],
				"depth": df["img_depth"][i],
			}
		]

		# Skip this if cat_id is na
		if not pd.isna(df["cat_id"][i]):
			# print(df["ann_score"][i])
			# sys.exit()

			annotations = [
				{
					"id": df.index[i],
					"image_id": df["img_id"][i],
					"segmented": df["ann_segmented"][i],
					"bbox": [
						df["ann_bbox_xmin"][i],
						df["ann_bbox_ymin"][i],
						df["ann_bbox_width"][i],
						df["ann_bbox_height"][i],
					],
					"area": df["ann_area"][i],
					"segmentation": df["ann_segmentation"][i],
					"iscrowd": df["ann_iscrowd"][i],
					"score": df["ann_score"][i] if not math.isnan(df["ann_score"][i]) else float(random.randint(60, 90) / 100),
					"pose": df["ann_pose"][i],
					"truncated": df["ann_truncated"][i],
					"category_id": int(df["cat_id"][i]),
					"difficult": df["ann_difficult"][i],
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
		if not pd.isna(categories[0]["id"]):
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

def convert_yolo_result_to_coco_result(folder_img, folder_yolo, file_re):
	yoloclasses = classes
	dataset = ImportYoloV5_PBVS(
		path           = folder_yolo,
		path_to_images = folder_img,
		cat_names      = yoloclasses,
		name           = "pbvs_dataset"
	)

	print(f"Info: {dataset.df.info()}")
	print(f"Number of images: {dataset.analyze.num_images}")
	print(f"Number of classes: {dataset.analyze.num_classes}")
	print(f"Classes:{dataset.analyze.classes}")
	print(f"Class counts:\n{dataset.analyze.class_counts}")

	# DEBUG:
	count = 0
	name  = ""
	set_file_name = set()
	for index in tqdm(dataset.df.index, desc=f""):
		# get row
		set_file_name.add((os.path.splitext(dataset.df.iloc[index]["img_filename"])[0], dataset.df.iloc[index]["img_id"]))

	list_file_name = np.array(list(set_file_name))
	list_file_name = sorted(list_file_name, key=lambda x: int(x[0]))
	for item in list_file_name:
		print(item[0], " ", item[1])
	# print(set_file_name)
	sys.exit()

	dataset.export.ExportToYoloV5(
		output_path   = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/result/yolo_format_test/labels/",
		copy_images = True,
		cat_id_index = 0  # 0: cat_id
	)[0]



	# SUGAR: export to COCO, image_id, object_id must start from 1, PBVS rule
	print(ExportToCoco_PBVS(
		dataset      = dataset,
		output_path  = file_re,
		cat_id_index = 1
	))

	# SUGAR:
	# Because the json file result only the array of annotations, not the whole json file
	# so we need to extract only annotations
	with open(file_re) as f_in:
		anns = json.load(f_in)
	anns = anns["annotations"]

	# SUGAR: add 1 into image_id, object_id, image_id, object_id must start from 1, PBVS rule
	for ann in anns:
		ann["image_id"] += 1
		ann["id"] += 1

	with open(file_re, 'w') as f_ou:
		json.dump(anns, f_ou, indent=4)


def evaliation_coco_result(annFile, resFile):
	annType = ['segm','bbox','keypoints']
	annType = annType[1]      #specify type here
	print('Running demo for *%s* results.'%(annType))

	#initialize COCO ground truth api
	# annFile = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/annotations/val/seq2/thermal/COCO/annotations.json"
	cocoGt  = COCO(annFile)

	#initialize COCO detections api
	# resFile = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/result/seq2_od_result_example.json"
	try:
		cocoDt  = cocoGt.loadRes(resFile)
	except AssertionError:
		# Because the json file result only the array of annotations, not the whole json file
		# so we need to extract only annotations
		anns = None
		with open(resFile) as f_in:
			anns = json.load(f_in)
			anns = anns["annotations"]
		with open(resFile, 'w') as f_ou:
			json.dump(anns, f_ou)

		# load again
		# cocoDt  = cocoGt.loadRes(resFile)

	imgIds = sorted(cocoGt.getImgIds())
	imgIds = imgIds[0 : 100]

	# running evaluation
	cocoEval = COCOeval(cocoGt, cocoDt, annType)
	cocoEval.params.imgIds  = imgIds
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()


def evaluate_detection():
	# Init file
	file_gt     = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/annotations/val/seq2/thermal/COCO/annotations.json"
	file_re     = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/result/seq2_od_result_conversion.json"
	folder_yolo = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/annotations/val/seq2/thermal/yolo"
	folder_img  = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/images/val/seq2/thermal/"

	convert_yolo_result_to_coco_result(folder_img, folder_yolo, file_re)

	# file_re     = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/result/seq2_od_result_example.json"
	evaliation_coco_result(file_gt, file_re)

def check_json():
	with open("/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/result/seq2_od_result_example.json") as f_in:
		anns_origin = json.load(f_in)
	with open("/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/result/seq2_od_result_example.txt", "w") as f_write:
		for ann in anns_origin:
			f_write.write(f"{ann['image_id']} {ann['id']}\n")


	with open("/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/result/seq2_od_result_conversion.json") as f_in:
		anns_conver = json.load(f_in)
	with open("/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset/result/seq2_od_result_conversion.txt", "w") as f_write:
		for ann in anns_conver:
			f_write.write(f"{ann['image_id']} {ann['id']}\n")

if __name__ == '__main__':
	# parser = argparse.ArgumentParser(description='MTMC Evaluation')
	# parser.add_argument('--re', help='Tracking result', type=str)
	# parser.add_argument('--gt', help='Ground-truth annotation', type=str)
	# parser.add_argument('--ou', help='Evaluation result', type=str)
	# args = parser.parse_args()

	evaluate_detection()

	check_json()