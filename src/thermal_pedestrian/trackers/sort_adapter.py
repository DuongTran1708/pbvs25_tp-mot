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

__all__ = [
	"sort"
]


# MARK: - SORT

@TRACKERS.register(name="sort")
class SORT(BaseTracker):

	# MARK: Magic Functions

	def __init__(self,
	             name: str = "sort",
	             *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)

	# MARK: Configure