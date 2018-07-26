"""
Model wrapper for TCAV.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import torch
from torch.autograd import grad

class ModelWrapper():
	__metaclass__ = ABCMeta

	@abstractmethod
	def __init__(self):
		self.bottleneck_tensors = None
		self.ends = None
		self.model_name = None
		self.image_shape = None
		self.y_input = None
		self.loss = None

	def _make_gradient_tensors(self):
		"""
		Makes gradient tensors for all bottleneck tensors.
		"""
		for bottleneck in self.bottlenecks_tensors:
			self.bottlenecks_gradients[bottleneck] = grad(
				self.loss, self.bottlenecks_tensors[bottleneck])

	def get_gradient(self, acts, y, bottleneck_name):
		"""
		Return the gradient of the loss w.r.t. the bottleneck name.

		:param acts: activation of the bottleneck
		:param y: index of the logit layer
		:param bottleneck_name: name of the bottleneck to get gradient w.r.t.
		:return: the gradient array
		"""
		self.bottlenecks_tensors = acts
		self.y_input = y
		return self.bottlenecks_gradients[bottleneck_name]

	def get_predictions(self, imgs):
		"""
		Get prediction of the images.

		:param imgs: array of images to get predictions
		:return: array of predictions
		"""
		return self.adjust_prediction(self.ends[])





















































































