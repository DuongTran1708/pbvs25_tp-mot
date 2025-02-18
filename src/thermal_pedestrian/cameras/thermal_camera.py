# ==================================================================== #
# Copyright (C) 2022 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
import threading
import uuid
import glob
import copy
import random
from queue import Queue
from operator import itemgetter
from timeit import default_timer as timer
from typing import Union, Optional

import cv2
import torch
import numpy as np
from tqdm import tqdm

from thermal_pedestrian.configuration import (
	data_dir,
	config_dir,
	result_dir
)
from thermal_pedestrian.cameras.base import BaseCamera

__all__ = [
	"ThermalCamera"
]

from thermal_pedestrian.core.factory.builder import CAMERAS, DETECTORS
from thermal_pedestrian.core.io.frame import FrameLoader
from thermal_pedestrian.core.io.video import VideoLoader

# NOTE: only for PBVS 2025
classes_pbvs = ['person']


# MARK: - ThermalCamera

# noinspection PyAttributeOutsideInit

@CAMERAS.register(name="thermal_camera")
class ThermalCamera(BaseCamera):

	# MARK: Magic Functions

	def __init__(
			self,
			data         : dict,
			dataset      : str,
			name         : str,
			detector     : dict,
			data_loader  : dict,
			data_writer  : Union[FrameWriter,  dict],
			process      : dict,
			id_          : Union[int, str] = uuid.uuid4().int,
			verbose      : bool            = False,
			drawing      : bool            = False,
			queue_size   : int             = 10,
			*args, **kwargs
	):
		"""

		Args:
			dataset (str):
				Dataset name. It is also the name of the directory inside
				`data_dir`.
			subset (str):
				Subset name. One of: [`dataset_a`, `dataset_b`].
			name (str):
				Camera name. It is also the name of the camera's config
				files.
			class_labels (ClassLabels, dict):
				ClassLabels object or a config dictionary.
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
			data_loader (FrameLoader, dict):
				Data loader object or a data loader's config dictionary.
			data_writer (VideoWriter, dict):
				Data writer object or a data writer's config dictionary.
			id_ (int, str):
				Camera's unique ID.
			verbose (bool):
				Verbosity mode. Default: `False`.
			queue_size (int):
				Size of queue store the information
		"""
		super().__init__(id_=id_, dataset=dataset, name=name)
		# NOTE: Init attributes
		self.start_time = None
		self.pbar       = None
		self.detector   = None

		# NOTE: Define attributes
		self.process         = process
		self.verbose         = verbose
		self.drawing         = drawing

		# NOTE: Define configurations
		self.data_cfg        = data
		self.detector_cfg    = detector
		self.data_loader_cfg = data_loader
		self.data_writer_cfg = data_writer

		# NOTE: Queue
		self.frames_queue                 = Queue(maxsize = self.data_loader_cfg['queue_size'])
		self.detections_queue_identifier  = Queue(maxsize = self.detector_cfg['queue_size'])
		self.writer_queue                 = Queue(maxsize = self.data_writer_cfg['queue_size'])

		# NOTE: Init modules
		self.init_dirs()

		# NOTE: Init for output
		self.init_data_output()

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs.
		"""
		self.root_dir    = os.path.join(data_dir)
		self.configs_dir = os.path.join(config_dir)
		self.outputs_dir = os.path.join(result_dir, self.data_writer_cfg["dst"])
		self.video_dir   = os.path.join(self.root_dir, self.data_loader_cfg["data"])
		self.image_dir   = os.path.join(self.root_dir, self.data_loader_cfg["data"])

	def init_detector(self, detector: Union[BaseDetector, dict]):
		"""Initialize detector.

		Args:
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
		"""
		pass

	def init_data_loader(self, data_loader_cfg: dict):
		"""Initialize data loader.

		Args:
			data_loader_cfg (dict):
				Data loader object or a data loader's config dictionary.
		"""
		if self.process["run_image"]:
			self.data_loader = FrameLoader(data=os.path.join(data_dir, "images", data_loader_cfg["data_path"]), batch_size=data_loader_cfg['batch_size'])
		else:
			self.data_loader = VideoLoader(data=os.path.join(data_dir, "videos", data_loader_cfg["data_path"]), batch_size=data_loader_cfg['batch_size'])

	def init_data_output(self):
		"""Initialize data writer."""
		pass

	# MARK: Run

	def run_detector(self):
		"""Run detection model"""
		pass

	def run_tracker(self):
		"""Run tracking model"""
		pass


	def run_heuristic(self):
		"""Run heuristic model"""
		# NOTE: init parameter
		pass

	def writing_final_result(self, data_path_start, data_path_end):
		"""Write the final result to the file."""
		# NOTE: run writing
		pass

	def run(self):
		"""Main run loop."""
		self.run_routine_start()

		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.start_time = timer()
		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

	def run_routine_end(self):
		"""Perform operations when run routine ends."""
		# NOTE: clear detector
		if self.detector is not None:
			self.detector.clear_model_memory()
			self.detector = None

		cv2.destroyAllWindows()
		self.stop_time = timer()
		if self.pbar is not None:
			self.pbar.close()

	def postprocess(self, image: np.ndarray, *args, **kwargs):
		"""Perform some postprocessing operations when a run step end.

		Args:
			image (np.ndarray):
				Image.
		"""
		if not self.verbose and not self.save_image and not self.save_video:
			return

		torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

		elapsed_time = timer() - self.start_time
		if self.verbose:
			# cv2.imshow(self.name, result)
			cv2.waitKey(1)

	# MARK: Visualize

	def draw(
			self,
			drawing     : np.ndarray,
			gmos        : list       = None,
			rois        : list       = None,
			mois        : list       = None,
			elapsed_time: float      = None,
	) -> np.ndarray:
		"""Visualize the results on the drawing.

		Args:
			drawing (np.ndarray):
				Drawing canvas.
			elapsed_time (float):
				Elapsed time per iteration.

		Returns:
			drawing (np.ndarray):
				Drawn canvas.
		"""
		pass




