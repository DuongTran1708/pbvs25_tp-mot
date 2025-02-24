from __future__ import annotations

from typing import (Any, Union, List)
import sys
import warnings
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import platform

from torch import Tensor
import torch

from BoostTrack_source.default_settings import GeneralSettings
from thermal_pedestrian.core.factory.builder import TRACKERS
from thermal_pedestrian.core.objects.gmo import General_Moving_Object
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.core.utils.image import to_channel_first
from thermal_pedestrian.trackers import BaseTracker

from DiffMOT.tracker.DiffMOTtracker import diffmottracker


__all__ = [
	"BoostTrack_Adapter"
]


# MARK: - BoostTrack

@TRACKERS.register(name="boosttrack")
class BoostTrack_Adapter(BaseTracker):
	"""BoostTrack

	Attributes:
		Same as ``Tracker``
	"""
	# MARK: Magic Functions

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.init_model()

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		self.model = BoostTrack()

	# MARK: Update

	def update(self, detections: List[Instance], image: Any, *args, **kwargs):
		"""Update ``self.tracks`` with new detections.

		Args:
			detections (list):
				The list of newly ``Instance`` objects.

		Requires:
			This method must be called once for each frame even with empty detections, just call update with empty list [].

		Returns:

		"""

		pass


	def update_matched_tracks(
			self,
			matched   : Union[List, np.ndarray],
			detections: List[Instance]
	):
		"""Update the track that has been matched with new detection

		Args:
			matched (list or np.ndarray):
				Matching between self.tracks index and detection index.
			detections (any):
				The newly detections.
		"""
		pass

	def create_new_tracks(
			self,
			unmatched_dets: Union[List, np.ndarray],
			detections    : List[Instance]
	):
		"""Create new tracks.

		Args:
			unmatched_dets (list or np.ndarray):
				Index of the newly detection in ``detections`` that has not matched with any tracks.
			detections (any):
				The newly detections.
		"""
		pass

	def delete_dead_tracks(
			self
	):
		"""Delete dead tracks.
		"""
		pass

	def associate_detections_to_tracks(
			self,
			dets: np.ndarray,
			trks: np.ndarray,
			**kwargs
	):
		"""Assigns detections to ``self.tracks``

		Args:
			dets (np.ndarray):
				The list of newly ``Instance`` objects.
			trks (np.ndarray):

		Returns:
			3 lists of matches, unmatched_detections and unmatched_trackers
		"""
		pass

	def clear_model_memory(self):
		"""Free the memory of model

		Returns:
			None
		"""
		if self.model is not None:
			del self.model
			torch.cuda.empty_cache()