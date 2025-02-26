import os
import sys

import numpy as np
from os.path import join, exists

from tqdm import tqdm


def tiou(track1, track2, iou_s):
	fstart1, fstop1 = track1[0, 0], track1[-1, 0]
	fstart2, fstop2 = track2[0, 0], track2[-1, 0]
	if fstop1 < fstart2 or fstop2 < fstart1: return 0.
	len1, len2 = track1.shape[0], track2.shape[0]

	assert len1 >= len2

	frames1, frames2 = set(track1[:, 0]), set(track2[:, 0])
	inter_frames = tuple(frames1.intersection(frames2))
	inter_track1 = track1[np.isin(track1[:, 0], inter_frames)]
	inter_track2 = track2[np.isin(track2[:, 0], inter_frames)]
	assert np.all(inter_track1[:, 0] == inter_track2[:, 0])

	'''Spatial-IoU'''
	x1_1, y1_1 = inter_track1[:, 2], inter_track1[:, 3]
	w1, h1 = inter_track1[:, 4], inter_track1[:, 5]
	x2_1, y2_1 = x1_1 + w1, y1_1 + h1
	x1_2, y1_2 = inter_track2[:, 2], inter_track2[:, 3]
	w2, h2 = inter_track2[:, 4], inter_track2[:, 5]
	x2_2, y2_2 = x1_2 + w2, y1_2 + h2
	areas_1, areas_2 = w1 * h1, w2 * h2
	x1_ = np.maximum(x1_1, x1_2)
	y1_ = np.maximum(y1_1, y1_2)
	x2_ = np.minimum(x2_1, x2_2)
	y2_ = np.minimum(y2_1, y2_2)
	w_  = np.maximum(0.  , x2_ - x1_)
	h_  = np.maximum(0.  , y2_ - y1_)
	inter_spatial = w_ * h_
	iou_spatial = inter_spatial / (areas_1 + areas_2 - inter_spatial)

	'''Temporal-IoU'''
	inter_temporal = np.sum(iou_spatial > iou_s)
	iou_temporal = inter_temporal / len2
	return iou_temporal


def merge_ID(track1, track2, mode='mean'):
	assert mode in ('mean', 'track1')
	frames1, frames2 = set(track1[:, 0]), set(track2[:, 0])
	track2[:, 1] = track1[0, 1]  # Unified ID
	inter_frames = tuple(frames1.intersection(frames2))
	diff_frames1 = tuple(frames1.difference(inter_frames))
	diff_frames2 = tuple(frames2.difference(inter_frames))

	track1_inter = track1[np.isin(track1[:, 0], inter_frames)]
	track2_inter = track2[np.isin(track2[:, 0], inter_frames)]
	assert np.all(track1_inter[:, 0] == track2_inter[:, 0])

	track1_diff = track1[np.isin(track1[:, 0], diff_frames1)]
	track2_diff = track2[np.isin(track2[:, 0], diff_frames2)]

	if mode == 'mean':
		track_inter = (track1_inter + track2_inter) / 2
	else:
		track_inter = track1_inter
	track_res = np.concatenate([track_inter, track1_diff, track2_diff], axis=0)
	track_res = track_res[np.argsort(track_res[:, 0])]
	return track_res


def nms(tracks, thres):
	bboxes = tracks[:, 2:6].copy()
	bboxes[:, 2:4] = bboxes[:, :2] + bboxes[:, 2:4]
	frames = set(tracks[:, 0])
	tracks_res = np.empty((0, 10))
	id2len = {id_: len(tracks[tracks[:, 1] == id_]) for id_ in set(tracks[:, 1])}
	length = np.array([id2len[row[1]] for row in tracks])
	for frame in frames:
		mask = tracks[:, 0] == frame
		bboxes_ = bboxes[mask]
		length_ = length[mask]
		x1, y1, x2, y2 = bboxes_[:, 0], bboxes_[:, 1], bboxes_[:, 2], bboxes_[:, 3]
		areas = (x2 - x1) * (y2 - y1)
		keep = []
		index = np.argsort(length_)[::-1]
		while index.shape[0]:
			index_max = index[0]
			x1_ = np.maximum(x1[index_max], x1[index])
			y1_ = np.maximum(y1[index_max], y1[index])
			x2_ = np.minimum(x2[index_max], x2[index])
			y2_ = np.minimum(y2[index_max], y2[index])
			w_ = np.maximum(0., x2_ - x1_)
			h_ = np.maximum(0., y2_ - y1_)
			inter_ = w_ * h_
			iou = inter_ / (areas[index_max] + areas[index] - inter_)
			index = index[np.where(iou <= thres)[0]]
			keep.append(index_max)
		keep = np.array(keep)
		tracks_ = tracks[mask][keep]
		tracks_res = np.concatenate([tracks_res, tracks_], axis=0)
	tracks_res = tracks_res[np.argsort(tracks_res[:, 0])]
	return tracks_res


def filter_by_length(tracks, thres):
	ids = set(tracks[:, 1])
	tracks_res = np.empty((0, 10))
	for id_ in ids:
		track_id = tracks[tracks[:, 1] == id_]
		if len(track_id) >= thres:
			tracks_res = np.concatenate((tracks_res, track_id), axis=0)
	return tracks_res


def ensemble(tracks, iou_s, iou_t, merge_mode='mean'):
	ids        = set(tracks[:, 1])
	tracks     = tracks[np.argsort(tracks[:, 0])]  # sort by frame index
	tracks_res = np.empty((0, 10))

	ids        = sorted(ids, key=lambda x: len(tracks[tracks[:, 1] == x]), reverse=True)
	ids_used   = []

	for index, id1 in enumerate(ids):

		if id1 in ids_used:
			continue

		track1 = tracks[tracks[:, 1] == id1]
		track2 = []
		for j, id2 in enumerate(ids[index + 1:], start=index + 1):
			if id2 in ids_used:
				continue

			track2_ = tracks[tracks[:, 1] == id2]
			tiou_   = tiou(track1, track2_, iou_s)

			if tiou_ > iou_t:
				track2.append(track2_)
				ids_used.append(id2)

		if track2:
			for track2_ in track2:
				track1 = merge_ID(track1, track2_, merge_mode)

		tracks_res = np.concatenate([tracks_res, track1], axis=0)
	return tracks_res


def main():
	# Init parameter
	folder_mot_in = '/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/output_pbvs25/tracking/'
	seqs = ["seq2", "seq17", "seq22", "seq47", "seq54", "seq66"]
	methods = [
		'sort',
		'sort_backward',
		'boosttrack',
		'boosttrack_backward',
		'botsort',
		'botsort_backward',
	]

	# create folder out
	folder_en_out = join(folder_mot_in, 'ensemble', "_".join(methods))
	os.makedirs(folder_en_out, exist_ok=True)

	merge_mode = 'track1'
	for seq in tqdm(seqs):
		print(f'Processing video {seq}...')
		file_mot_en_ou = join(folder_en_out, seq, 'thermal', f'{seq}_thermal.txt')
		file_mot_in    = join(folder_mot_in, methods[0], seq, 'thermal', f'{seq}_thermal.txt')

		# create folder
		os.makedirs(os.path.dirname(file_mot_en_ou), exist_ok=True)

		# load the first mot result
		preds     = np.loadtxt(file_mot_in, delimiter=',')

		# load the rest mot results
		# '{frame},{id},{x1},{y1},{w},{h},{s},{label},-1,-1\n'
		for method in methods[1:]:
			file_mot_in   = join(folder_mot_in, method, seq, 'thermal', f'{seq}_thermal.txt')
			preds_        = np.loadtxt(file_mot_in, delimiter=',')
			max_id        = np.max(preds[:, 1])
			preds_[:, 1] += max_id + 1
			preds = np.concatenate([preds, preds_], axis=0)

		# ensemble process
		preds = ensemble(preds, iou_s=0.5, iou_t=0.5, merge_mode=merge_mode)
		preds = nms(preds, thres=.7)
		preds = filter_by_length(preds, thres=20)

		# write out
		np.savetxt(file_mot_en_ou, preds, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d')

	# Run evaluation
	os.system(f"python utilities/tracking_evaluation.py  \
				  --gt_folder /media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/  \
				  --mot_folder {folder_en_out}  \
				  --ou_folder {folder_en_out}  \
				")


if __name__ == '__main__':
	main()
