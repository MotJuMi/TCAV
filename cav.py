import os
import pickle
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.training import HParams
import torch
import utils as utils

class CAV():
	"""
	CAV class contains methods for concept activation vector.
	CAV represents semantically meaningful vector directions in 
	network's embeddings (bottlenecks).
	"""

	@staticmethod
	def default_hparams():
		"""
		Hyper-params used to train the CAV.
		
		:return: tf.hparams for training
		"""
		return HParams(model_type='linear', alpha=.01)

	@staticmethod
	def load_cav(cav_path):
		"""
		Make a CAV instance from a saved CAV (pickle file).

		:param cav_path: location of the saved CAV.
		:return: CAV instance
		"""
		with open(cav_path) as pkl_file:
			save_dict = pickle.load(pkl_file)

		cav = CAV(save_dict['concepts'], save_dict['bottleneck'],
				  save_dict['hparams'], save_dict['saved_path'])
		cav.accuracies = save_dict['accuracies']
		cav.cavs = save_dict['cavs']
		return cav

	@staticmethod
	def cav_keys(concepts, bottleneck, model_type, alpha):
		"""
		A key of this cav.

		:param concepts: set of concepts used for CAV
		:param bottleneck: the bottleneck used for CAV
		:param model_type: the name of model for CAV
		:param alpha: a parameter used to learn CAV
		:return: a string cav_key
		"""
		return '-'.join([str(c) for c in concepts
						]) + '-' + bottleneck + \
							 '-' + model_type + \
							 '-' + str(alpha)

	@staticmethod
	def check_cav_exists(cav_dir, concepts, bottleneck, cav_hparams):
		"""
		Check if a CAV is saved in cav_dir.

		:param cav_dir: directory where cav pickles might be saved
		:param concepts: set of concepts used for CAV
		:param bottleneck: the bottleneck used for CAV
		:param cav_hparams: a parameter used to learn CAV
		:return: true if exists, false otherwise.
		"""
		cav_path = os.path.join(
			cav_dir,
			CAV.cav_key(concepts, bottleneck, cav_hparams.model_type,
						cav_hparams.alpha) + '.pkl'
		)
		return os.path.exists(cav_path)

	@staticmethod
	def _create_cav_training_set(concepts, bottleneck, acts):
		"""
		Flatten acts, make mock-labels and returns the info.

		Labels are assigned in the order that concepts exists.

		:param concepts: names of concepts
		:param bottleneck: the name of bottleneck where acts come from
		:param acts: a dict that contains activations
		:return x: flattened acts
		:return labels: corresponding labels (int) 
		:return labels2text: map between labels and text.
		"""
		x, labels, labels2text = [], [], []
		min_data_points = np.min(
			[acts[concept][bottleneck].shape[0] for concept in acts.keys()])

		for i, concept in enumerate(concepts):
			x.extend(acts[concept][bottleneck][:min_data_points].reshape(
				min_data_points, -1))
			labels.extend([i] * min_data_points)
			labels2text[i] = concept
		x = np.array(x)
		labels = np.array(labels)

		return x, labels, labels2text

	def __init__(self, concepts, bottleneck, hparams, save_path=None):
		"""
		Initialize CAV class.

		:param concepts: set of concepts used for CAV
		:param bottleneck: the bottleneck used for CAV
		:param hparams: a parameter used to learn CAV
		:param save_path: where to save this CAV
		"""
		self.concepts = concepts
		self.bottleneck = bottleneck
		self.hparams = hparams
		self.save_path = save_path

	def train(self, acts):
		"""
		Train the CAVs from the activations.

		:param acts: dict of activations.
					 {
					 	'concept1': {
					 		'bottleneck name1':[...act array...],
					 		'bottleneck name2':[...act array...],...			
					 	},
					 	'concept2': {
					 		'bottleneck name1':[...act array...],
					 		'bottleneck name2':[...act array...],...			
					 	}
					 }
		:raise ValueError: if the mode_type in hparam is not compatible.
		"""

		tf.logging.info('training with alpha={}'.format(self.hparams.alpha))
		x, labels, labels2text = CAV._create_cav_training_set(
			self.concepts, self.bottleneck, acts)

		if self.hparams.model_type == 'linear':
			model = linear_model.SGDClassifier(alpha=self.hparams.alpha)
		elif self.hparams.model_type == 'logistic':
			model = linear_model.LogisticRegression()
		else:
			raise ValueError('Invalid hparams.model_type: {}'.format(
				self.hparams.model_type))

		self.accuracies = self._train_model(model, x, labels, labels2text)
		if len(model.coef_) == 1:
			self.cavs = [-1 * model.coef_[0], model.coef_[0]]
		else:
			self.cavs = [coef for coef in model.coef_]
		self._save_cavs()

	def perturb_acts(self, act, concept, operation=np.add, alpha=1.0):
		"""
		Make a perturbation of acts with a direction of this CAV.
		:param act: activations to be perturbed
		:param concept: the concept to perturbed act with
		:param operation: the operation will be ran to perturb
		:param alpha: size of the step
		:return perturbed activation: same shape as act.
		"""
		flat_act = np.reshape(act, -1)
		perturbation = operation(flat_act, alpha * self.get_direction(concept))
		return np.reshape(perturbation, act.shape)

	def get_key(self):
		"""
		Return cav_key.
		"""
		return CAV.cav_key(self.concepts, self.bottleneck,
						   self.hparams.model_type, self.hparams.alpha)

	def get_direction(self, concept):
		"""
		Get a CAV direction.
		:param concept: the concept of interest
		:return: CAV vector.
		"""
		return self.cavs[self.concepts.index(concept)]

	def _save_cavs(self):
		"""
		Save a dict of this CAV to a pickle.
		"""
		save_dict = {
			'concepts': self.concepts,
			'bottleneck': self.bottleneck,
			'hparams': self.hparams,
			'accuracies': self.accuracies,
			'cavs': self.cavs,
			'saved_path': self.saved_path
		}
		if self.save_path is not None:
			with open(self.save_path, 'w') as pkl_file:
				pickle.dump(save_dict, pkl_file)
		else:
			tf.logging.info('save_path is None. Not saving anything')

	def _train_model(self, model, x, y, labels2text):
		"""
		Train a model to get CAVs.

		Modifies model by calling model.fit functions. The cav coefficients
		are then in model._coefs.

		:param model: a scikit-learn linear_model object. Can be either linear or logistic
					  regression. Must support .fir and ._coef.
		:param x: an array of training observations of shape [n_observations, observation_dim]
		:param y: an array of integer labels of shape [n_observations]
		:param labels2text: dict of textual name for each label
		:return: dict of accuracies of the CAVs.
		"""
		X_train, X_test, y_train, y_test = train_test_split(
			x, y, test_size=0.33, stratify=y)
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		n_classes = max(y) + 1
		acc = {}
		n_correct = 0
		for class_id in range(n_classes):
			idx = (y_test == class_id)
			acc[labels2text[class_id]] = metrics.accuracy_score(
				y_pred[idx], y_test[idx])
			n_correct += (sum(idx) * acc[labels2text[class_id]])
		acc['overall'] = n_correct / len(y_test)
		tf.logging.info('acc per class %s' % (str(acc)))
		return acc

def get_or_train_cav(concepts, bottleneck, acts,
					 cav_dir=None, cav_hparams=None,
					 overwrite=False):
	"""
	Get, create and train if necessary, the specified CAV.
	Assumes the activations already exists.

	:param concepts: set of concepts used for CAV
	:param bottleneck: the bottleneck used for CAV
	:param acts: a dict contains activations of concepts in each bottleneck
	:param cav_dir: a dir to store the results
	:param cav_hparams: a parameter used to learn CAV
	:param overwrite: if set to true, overwrite any saved CAV files
	:return: a CAV instance
	"""
	if cav_hparams is None:
		cav_hparams = CAV.default_hparams()

	cav_path = None
	if cav_dir is not None:
		utils.mkdir(cav_dir)
		cav_path = os.path.join(
			cav_dir,
			CAV.cav_key(concepts, bottleneck, cav_hparams.model_type,
						cav_hparams.alpha).replace('/', '.') + '.pkl')

		if not overwrite and os.path.join(cav_path):
			tf.logging.info('CAV already exists: {}'.format(cav_path))
			cav_instance = CAV.load_cav(cav_path)
			return cav_instance

	tf.logging.info('Training CAV {} - {} alpha {}'.format(
					concepts, bottleneck, cav_hparams.alpha))
	cav_instance = CAV(concepts, bottleneck, cav_hparams, cav_path)
	cav_instance.train({concept: acts[concepts] for concept in concepts})
	return cav_instance