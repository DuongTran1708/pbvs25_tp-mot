import os
import sys
import numpy as np
import argparse
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from easydict import EasyDict as edict
from tqdm import tqdm


# region PROCESS FOR INPUT_OUTPUT


def print_format(widths, formaters, values, form_attr):
	return ' '.join([(form_attr % (width, form)).format(val) for (
		form, width, val) in zip(formaters, widths, values)])


def print_format_name(widths, values, form_attr):
	return ' '.join([(form_attr % (width)).format(val) for (width, val) in zip(
		widths, values)])


def print_metrics_ext(seq, file_re_ou, header, metrics, extra_info, banner=30):
	print('\n{} {} {}'.format('*' * banner, header, '*' * banner))

	metric_names_short = ['IDF1', 'IDP', 'IDR',
						  'Rcll', 'Prcn', 'FAR',
						  'GT', 'MT', 'PT', 'ML',
						  'FP', 'FN', 'IDs', 'FM',
						  'MOTA', 'MOTP', 'MOTAL']

	metric_widths_short = [5, 4, 4, 5, 5, 5, 4, 4, 4, 4, 6, 6, 5, 5, 5, 5, 5]

	metric_format_long = ['.1f', '.1f', '.1f',
						  '.1f', '.1f', '.2f',
						  '.0f', '.0f', '.0f', '.0f',
						  '.0f', '.0f', '.0f', '.0f',
						  '.1f', '.1f', '.1f']

	splits = [(0, 3), (3, 6), (6, 10), (10, 14), (14, 17)]

	# DEBUG:
	# print(dir(metric_widths_short))
	# print(dir(metric_names_short))
	# print(dir(metrics))
	# print(metrics)
	# print(extra_info)
	# sys.exit()

	print(' | '.join([print_format_name(
		metric_widths_short[start:end],
		metric_names_short[start:end], '{0: <%d}')
		for (start, end) in splits]))

	print(' | '.join([print_format(
		metric_widths_short[start:end],
		metric_format_long[start:end],
		metrics[start:end], '{:%d%s}')
		for (start, end) in splits]))
	print('\n\n')

	with open(file_re_ou, "a") as f_write:
		f_write.write(f"\n\n--Evaluate {seq} -- \n")

		f_write.write(' | '.join([print_format_name(
			metric_widths_short[start:end],
			metric_names_short[start:end], '{0: <%d}')
			for (start, end) in splits]))

		f_write.write("\n")

		f_write.write(' | '.join([print_format(
			metric_widths_short[start:end],
			metric_format_long[start:end],
			metrics[start:end], '{:%d%s}')
			for (start, end) in splits]))

def print_idmetrics(seq, file_re_ou, header, metrics, extra_info, banner=30):
	metric_names  = ['IDF1']
	metric_values = [extra_info.idmetrics.IDF1]

	print(metric_names)
	print(metric_values)

	for name, value in zip(metric_names, metric_values):
		print(f"{name} : {value:0.2f}")

	with open(file_re_ou, 'a') as f_write:
		for name, value in zip(metric_names, metric_values):
			f_write.write(f"{name} : {value:0.2f}\n")


def read_txt_to_struct_MTMC(fname):
	"""
	Read txt to structure, the column represents:
		[frame number] [identity number] [bbox left] [bbox top] [bbox width] [bbox height]
		[DET: detection score, GT: ignored class flag] [class] [visibility ratio]
		MOT : <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
		MTMC: <camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <yworld>
	"""
	data = []
	with open(fname, 'r') as fid:
		lines = fid.readlines()
		for line in lines:
			line = list(map(int, line.strip().split(' ')))
			data.append(line)
	data = np.array(data)
	# change point-size format to two-points format
	data[:, 5:7]          += data[:, 3:5]
	for element in data:
		element[1], element[2] = element[2], element[1]

	# plit cameras
	cams_data   = []
	index       = 0
	data_remain = len(data)

	while data_remain > 0:
		index       += 1
		data_temp    = [line[1:] for line in data if line[0] == index]
		# data_temp    = data[:, data[:][0] == index]
		cams_data.append(np.array(data_temp))
		data_remain -= len(data_temp)

	# DEBUG:
	# print(cams_data[0])

	return cams_data


def read_txt_to_struct_MOT(fname):
	"""
	Read txt to structure, the column represents:
		[frame number] [identity number] [bbox left] [bbox top] [bbox width] [bbox height]
		[DET: detection score, GT: ignored class flag] [class] [visibility ratio]
		MOT : <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
		MTMC: <camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <yworld>
	"""
	data = []
	with open(fname, 'r') as fid:
		lines = fid.readlines()
		for line in lines:
			line = list(map(float, line.strip().split(',')))
			data.append(line)
	data = np.array(data)
	# change point-size format to two-points format
	data[:, 4:6] += data[:, 2:4]
	return data

# endregion


# region EVALUATION


def compute_distance(traj1, traj2, matched_pos):
	"""
	Compute the loss hit in traj2 regarding to traj1
	"""
	distance = np.zeros((len(matched_pos), ), dtype=float)
	for i in range(len(matched_pos)):
		if matched_pos[i] == -1:
			continue
		else:
			iou = bbox_overlap(traj1[i, 2:6], traj2[matched_pos[i], 2:6])
			distance[i] = iou
	return distance


def corresponding_frame(traj1, len1, traj2, len2):
	"""
	Find the matching position in traj2 regarding to traj1
	Assume both trajectories in ascending frame ID
	"""
	p1, p2 = 0, 0
	loc = -1 * np.ones((len1, ), dtype=int)
	while p1 < len1 and p2 < len2:
		if traj1[p1] < traj2[p2]:
			loc[p1] = -1
			p1 += 1
		elif traj1[p1] == traj2[p2]:
			loc[p1] = p2
			p1 += 1
			p2 += 1
		else:
			p2 += 1
	return loc


def cost_between_trajectories(traj1, traj2, threshold):
	[npoints1, dim1] = traj1.shape
	[npoints2, dim2] = traj2.shape
	# find start and end frame of each trajectories
	start1 = traj1[0, 0]
	end1 = traj1[-1, 0]
	start2 = traj2[0, 0]
	end2 = traj2[-1, 0]

	# check frame overlap
	has_overlap = max(start1, start2) < min(end1, end2)
	if not has_overlap:
		fn = npoints1
		fp = npoints2
		return fp, fn

	# gt trajectory mapping to st, check gt missed
	matched_pos1 = corresponding_frame(
		traj1[:, 0], npoints1, traj2[:, 0], npoints2)
	# st trajectory mapping to gt, check computed one false alarms
	matched_pos2 = corresponding_frame(
		traj2[:, 0], npoints2, traj1[:, 0], npoints1)
	dist1 = compute_distance(traj1, traj2, matched_pos1)
	dist2 = compute_distance(traj2, traj1, matched_pos2)
	# FN
	fn = sum([1 for i in range(npoints1) if dist1[i] < threshold])
	# FP
	fp = sum([1 for i in range(npoints2) if dist2[i] < threshold])
	return fp, fn


def cost_between_gt_pred(groundtruth, prediction, threshold):
	n_gt = len(groundtruth)
	n_st = len(prediction)
	cost = np.zeros((n_gt, n_st), dtype=float)
	fp = np.zeros((n_gt, n_st), dtype=float)
	fn = np.zeros((n_gt, n_st), dtype=float)
	for i in range(n_gt):
		for j in range(n_st):
			fp[i, j], fn[i, j] = cost_between_trajectories(
				groundtruth[i], prediction[j], threshold)
			cost[i, j] = fp[i, j] + fn[i, j]
	return cost, fp, fn


def idmeasures(gtDB, stDB, threshold):
	"""
	compute MTMC metrics
	[IDP, IDR, IDF1]
	"""
	st_ids = np.unique(stDB[:, 1])
	gt_ids = np.unique(gtDB[:, 1])
	n_st = len(st_ids)
	n_gt = len(gt_ids)
	groundtruth = [gtDB[np.where(gtDB[:, 1] == gt_ids[i])[0], :]
				   for i in range(n_gt)]
	prediction = [stDB[np.where(stDB[:, 1] == st_ids[i])[0], :]
				  for i in range(n_st)]
	cost = np.zeros((n_gt + n_st, n_st + n_gt), dtype=float)
	cost[n_gt:, :n_st] = sys.maxsize  # float('inf')
	cost[:n_gt, n_st:] = sys.maxsize  # float('inf')

	fp = np.zeros(cost.shape)
	fn = np.zeros(cost.shape)
	# cost matrix of all trajectory pairs
	cost_block, fp_block, fn_block = cost_between_gt_pred(
		groundtruth, prediction, threshold)

	cost[:n_gt, :n_st] = cost_block
	fp[:n_gt, :n_st] = fp_block
	fn[:n_gt, :n_st] = fn_block

	# computed trajectory match no groundtruth trajectory, FP
	for i in range(n_st):
		cost[i + n_gt, i] = prediction[i].shape[0]
		fp[i + n_gt, i] = prediction[i].shape[0]

	# groundtruth trajectory match no computed trajectory, FN
	for i in range(n_gt):
		cost[i, i + n_st] = groundtruth[i].shape[0]
		fn[i, i + n_st] = groundtruth[i].shape[0]
	try:
		matched_indices = linear_assignment(cost)
	except:
		import pdb
		pdb.set_trace()
	nbox_gt = sum([groundtruth[i].shape[0] for i in range(n_gt)])
	nbox_st = sum([prediction[i].shape[0] for i in range(n_st)])

	IDFP = 0
	IDFN = 0
	for matched in zip(*matched_indices):
		IDFP += fp[matched[0], matched[1]]
		IDFN += fn[matched[0], matched[1]]
	IDTP = nbox_gt - IDFN
	assert IDTP == nbox_st - IDFP
	IDP = IDTP / (IDTP + IDFP) * 100               # IDP = IDTP / (IDTP + IDFP)
	IDR = IDTP / (IDTP + IDFN) * 100               # IDR = IDTP / (IDTP + IDFN)
	# IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
	IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100

	measures = edict()
	measures.IDP = IDP
	measures.IDR = IDR
	measures.IDF1 = IDF1
	measures.IDTP = IDTP
	measures.IDFP = IDFP
	measures.IDFN = IDFN
	measures.nbox_gt = nbox_gt
	measures.nbox_st = nbox_st

	return measures


def intersection(a, b):
	x = np.maximum(a[:, 0], b[:, 0])
	y = np.maximum(a[:, 1], b[:, 1])
	w = np.minimum(a[:, 2], b[:, 2]) - x
	h = np.minimum(a[:, 3], b[:, 3]) - y
	return np.maximum(w, 0) * np.maximum(h, 0)


def areasum(a, b):
	return (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + \
		(b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def bbox_overlap(ex_box, gt_box):
	ex_box = ex_box.reshape(-1, 4)
	gt_box = gt_box.reshape(-1, 4)
	paded_gt = np.tile(gt_box, [ex_box.shape[0], 1])
	insec = intersection(ex_box, paded_gt)

	uni = areasum(ex_box, paded_gt) - insec
	return insec / uni


VERBOSE = False
def clear_mot_hungarian(stDB, gtDB, threshold):
	"""
	compute CLEAR_MOT and other metrics
	[recall, precision, FAR, GT, MT, PT, ML, falsepositives, false negatives,
	 idswitches, FRA, MOTA, MOTP, MOTAL]
	"""
	st_frames = np.unique(stDB[: , 0])
	gt_frames = np.unique(gtDB[: , 0])
	st_ids    = np.unique(stDB[: , 1])
	gt_ids    = np.unique(gtDB[: , 1])
	# f_gt = int(max(max(st_frames), max(gt_frames)))
	# n_gt = int(max(gt_ids))
	# n_st = int(max(st_ids))
	f_gt = len(gt_frames)
	n_gt = len(gt_ids)
	n_st = len(st_ids)

	mme = np.zeros((f_gt, ), dtype=float)          # ID switch in each frame
	# matches found in each frame
	c = np.zeros((f_gt, ), dtype=float)
	# false positives in each frame
	fp = np.zeros((f_gt, ), dtype=float)
	missed = np.zeros((f_gt, ), dtype=float)       # missed gts in each frame

	g = np.zeros((f_gt, ), dtype=float)            # gt count in each frame
	d = np.zeros((f_gt, n_gt), dtype=float)         # overlap matrix
	allfps = np.zeros((f_gt, n_st), dtype=float)

	gt_inds = [{} for i in range(f_gt)]
	st_inds = [{} for i in range(f_gt)]
	# matched pairs hashing gid to sid in each frame
	M = [{} for i in range(f_gt)]

	# hash the indices to speed up indexing
	for i in range(gtDB.shape[0]):
		frame = np.where(gt_frames == gtDB[i, 0])[0][0]
		gid   = np.where(gt_ids == gtDB[i, 1])[0][0]
		gt_inds[frame][gid] = i

	gt_frames_list = list(gt_frames)
	for i in range(stDB.shape[0]):
		# sometimes detection missed in certain frames, thus should be
		#  assigned to groundtruth frame id for alignment

		# DEBUG:
		# print(type(gt_frames_list))
		# print(len(gt_frames_list))
		# print(stDB[i, 0])

		if stDB[i, 0] in gt_frames_list:
			frame = gt_frames_list.index(stDB[i, 0])
			sid   = np.where(st_ids == stDB[i, 1])[0][0]
			st_inds[frame][sid] = i

	for t in range(f_gt):
		g[t] = len(list(gt_inds[t].keys()))

		# preserving original mapping if box of this trajectory has large
		# enough iou in avoid of ID switch
		if t > 0:
			mappings = list(M[t - 1].keys())
			sorted(mappings)
			for k in range(len(mappings)):
				if mappings[k] in list(gt_inds[t].keys()) and \
						M[t - 1][mappings[k]] in list(st_inds[t].keys()):
					row_gt = gt_inds[t][mappings[k]]
					row_st = st_inds[t][M[t - 1][mappings[k]]]
					dist = bbox_overlap(
						stDB[row_st, 2:6], gtDB[row_gt, 2:6])
					if dist >= threshold:
						M[t][mappings[k]] = M[t - 1][mappings[k]]
						if VERBOSE:
							print('perserving mapping: %d to %d' %
								  (mappings[k], M[t][mappings[k]]))
		# mapping remaining groundtruth and estimated boxes
		unmapped_gt, unmapped_st = [], []
		unmapped_gt = [key for key in gt_inds[t].keys()
					   if key not in list(M[t].keys())]
		unmapped_st = [key for key in st_inds[t].keys(
		) if key not in list(M[t].values())]
		if len(unmapped_gt) > 0 and len(unmapped_st) > 0:
			overlaps = np.zeros((n_gt, n_st), dtype=float)
			for i in range(len(unmapped_gt)):
				row_gt = gt_inds[t][unmapped_gt[i]]
				for j in range(len(unmapped_st)):
					row_st = st_inds[t][unmapped_st[j]]
					dist = bbox_overlap(stDB[row_st, 2:6], gtDB[row_gt, 2:6])
					if dist[0] >= threshold:
						overlaps[i][j] = dist[0]
			matched_indices = linear_assignment(1 - overlaps)

			for matched in zip(*matched_indices):
				if overlaps[matched[0], matched[1]] == 0:
					continue
				M[t][unmapped_gt[matched[0]]] = unmapped_st[matched[1]]
				if VERBOSE:
					print(
						'adding mapping: %d to %d' % (
							unmapped_gt[matched[0]],
							M[t][unmapped_gt[matched[0]]]))

		# compute statistics
		cur_tracked = list(M[t].keys())
		st_tracked = list(M[t].values())
		fps = [key for key in st_inds[t].keys()
			   if key not in list(M[t].values())]
		for k in range(len(fps)):
			allfps[t][fps[k]] = fps[k]
		# check miss match errors
		if t > 0:
			for i in range(len(cur_tracked)):
				ct = cur_tracked[i]
				est = M[t][ct]
				last_non_empty = -1
				for j in range(t - 1, 0, -1):
					if ct in M[j].keys():
						last_non_empty = j
						break
				if ct in gt_inds[t - 1].keys() and last_non_empty != -1:
					mtct, mlastnonemptyct = -1, -1
					if ct in M[t]:
						mtct = M[t][ct]
					if ct in M[last_non_empty]:
						mlastnonemptyct = M[last_non_empty][ct]

					if mtct != mlastnonemptyct:
						mme[t] += 1
		c[t] = len(cur_tracked)
		fp[t] = len(list(st_inds[t].keys()))
		fp[t] -= c[t]
		missed[t] = g[t] - c[t]
		for i in range(len(cur_tracked)):
			ct = cur_tracked[i]
			est = M[t][ct]
			row_gt = gt_inds[t][ct]
			row_st = st_inds[t][est]
			d[t][ct] = bbox_overlap(stDB[row_st, 2:6], gtDB[row_gt, 2:6])
	return mme, c, fp, g, missed, d, M, allfps


def evaluate_sequence(trackDB, gtDB, distractor_ids=np.ndarray([]), iou_thres=0.5, minvis=0):
	"""
	Evaluate single sequence
	trackDB: tracking result data structure
	gtDB: ground-truth data structure
	iou_thres: bounding box overlap threshold
	minvis: minimum tolerent visibility
	"""
	# trackDB, gtDB = preprocessingDB(
	# 	trackDB, gtDB, distractor_ids, iou_thres, minvis)
	mme, c, fp, g, missed, d, M, allfps = clear_mot_hungarian(
		trackDB, gtDB, iou_thres)

	gt_frames = np.unique(gtDB[:, 0])
	gt_ids    = np.unique(gtDB[:, 1])
	st_ids    = np.unique(trackDB[:, 1])
	f_gt      = len(gt_frames)
	n_gt      = len(gt_ids)
	n_st      = len(st_ids)

	FN  = sum(missed)
	FP  = sum(fp)
	IDS = sum(mme)
	# MOTP = sum(iou) / # corrected boxes
	MOTP = (sum(sum(d)) / sum(c)) * 100
	# MOTAL = 1 - (# fp + # fn + #log10(ids)) / # gts
	MOTAL = (1 - (sum(fp) + sum(missed) + np.log10(sum(mme) + 1)) / sum(g)) * 100
	# MOTA = 1 - (# fp + # fn + # ids) / # gts
	MOTA  = (1 - (sum(fp) + sum(missed) + sum(mme)) / sum(g)) * 100
	# recall = TP / (TP + FN) = # corrected boxes / # gt boxes
	recall = sum(c) / sum(g) * 100
	# precision = TP / (TP + FP) = # corrected boxes / # det boxes
	precision = sum(c) / (sum(fp) + sum(c)) * 100
	# FAR = sum(fp) / # frames
	FAR      = sum(fp) / f_gt
	MT_stats = np.zeros((n_gt, ), dtype=float)
	for i in range(n_gt):
		gt_in_person   = np.where(gtDB[:, 1] == gt_ids[i])[0]
		gt_total_len   = len(gt_in_person)
		gt_frames_tmp  = gtDB[gt_in_person, 0].astype(int)
		gt_frames_list = list(gt_frames)
		st_total_len   = sum(
			[1 if i in M[gt_frames_list.index(f)].keys() else 0
			 for f in gt_frames_tmp])
		ratio = float(st_total_len) / gt_total_len

		if ratio < 0.2:
			MT_stats[i] = 1
		elif ratio >= 0.8:
			MT_stats[i] = 3
		else:
			MT_stats[i] = 2

	ML = len(np.where(MT_stats == 1)[0])
	PT = len(np.where(MT_stats == 2)[0])
	MT = len(np.where(MT_stats == 3)[0])

	# fragment
	fr = np.zeros((n_gt, ), dtype=int)
	M_arr = np.zeros((f_gt, n_gt), dtype=int)

	for i in range(f_gt):
		for gid in M[i].keys():
			M_arr[i, gid] = M[i][gid] + 1

	for i in range(n_gt):
		occur = np.where(M_arr[:, i] > 0)[0]
		occur = np.where(np.diff(occur) != 1)[0]
		fr[i] = len(occur)
	FRA = sum(fr)
	idmetrics = idmeasures(gtDB, trackDB, iou_thres)
	metrics = [idmetrics.IDF1, idmetrics.IDP, idmetrics.IDR, recall,
			   precision, FAR, n_gt, MT, PT, ML, FP, FN, IDS, FRA,
			   MOTA, MOTP, MOTAL]
	extra_info     = edict()
	extra_info.mme = sum(mme)
	extra_info.c   = sum(c)
	extra_info.fp  = sum(fp)
	extra_info.g   = sum(g)
	extra_info.missed = sum(missed)
	extra_info.d = d
	# extra_info.m = M
	extra_info.f_gt = f_gt
	extra_info.n_gt = n_gt
	extra_info.n_st = n_st
	#    extra_info.allfps = allfps

	extra_info.ML  = ML
	extra_info.PT  = PT
	extra_info.MT  = MT
	extra_info.FRA = FRA
	extra_info.idmetrics = idmetrics
	return metrics, extra_info


# endregion


def evaluate_tracking(args):
	# Init file
	# folder_lbl_gt_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/"  # ground truth
	# folder_lbl_tr_in = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/"  # tracking result
	# folder_re_ou     = "/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/result/"  # result folder

	folder_lbl_gt_in = args.gt_folder  # ground truth
	folder_lbl_tr_in = args.mot_folder  # tracking result
	folder_re_ou     = args.ou_folder  # result folder


	# create output file
	file_re_ou  = os.path.join(folder_re_ou, f'thermal_tracking_evaluation_result.txt')
	with open(file_re_ou, 'w') as f_write:
		pass

	# get list folder
	list_seqs = sorted(os.listdir(folder_lbl_gt_in))


	metrics_seqs = []
	# loop to export dataset
	for seq in tqdm(list_seqs):
		# get path file
		file_gt_in  = os.path.join(folder_lbl_gt_in, seq, f'thermal/{seq}_thermal.txt')
		file_tr_in  = os.path.join(folder_lbl_tr_in, seq, f'thermal/{seq}_thermal.txt')

		# read result of each file
		gtDB    = read_txt_to_struct_MOT(file_gt_in)
		trackDB = read_txt_to_struct_MOT(file_tr_in)

		# evaluate sequence
		metrics, extra_info  = evaluate_sequence(trackDB, gtDB)

		# print metrics
		print_metrics_ext(seq, file_re_ou, f' Evaluation {seq}', metrics, extra_info)
		# print_idmetrics(seq, file_re_ou, ' Evaluation', metrics, extra_info)

		# append metrics sequences
		metrics_seqs.append(metrics)

	# calculate avarage
	metrics_seqs = np.array(metrics_seqs)
	metrics      = np.mean(metrics_seqs, axis=0)
	print_metrics_ext("average", file_re_ou, f' Evaluation average', metrics, extra_info)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Object detection evaluation bases on COCO format')
	parser.add_argument('--gt_folder', help='Folder of Ground-truth annotation', type=str,
	                    default="/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/"
	                    )
	parser.add_argument('--mot_folder', help='Folder of Detection result', type=str,
	                    default="/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/annotations/val/"
	                    )
	parser.add_argument('--ou_folder', help='Folder of Output Evaluation result', type=str,
	                    default="/media/sugarubuntu/DataSKKU3/3_Dataset/PBVS_challenge/tmot_dataset_after_checked/result/"
	                    )
	args = parser.parse_args()

	evaluate_tracking(args)
