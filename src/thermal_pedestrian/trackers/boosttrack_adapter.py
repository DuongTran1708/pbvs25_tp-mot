from __future__ import annotations

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

from thermal_pedestrian.core.factory.builder import TRACKERS
from thermal_pedestrian.trackers import BaseTracker
from third_party.BoostTrack.tracker.boost_track import BoostTrack

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
		self.model = BoostTrack(video_name=video_name)

	# MARK: Update

	def update(self, detections: List[Instance]):
		"""Update ``self.tracks`` with new detections.

		Args:
			detections (list):
				The list of newly ``Instance`` objects.

		Requires:
			This method must be called once for each frame even with empty detections, just call update with empty list [].

		Returns:

		"""
		self.frame_count += 1  # Should be the same with VideoReader.frame_idx

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

	def delete_dead_tracks(self):
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