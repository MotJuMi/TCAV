from multiprocessing import dummy as multiprocessing
import time
from cav import CAV, get_or_train_cav
import run_params
import tensorflow as tf
import utils

class TCAV():
	"""
	Run TCAV for one target and a set of concepts.
	"""
	@staticmethod
	def get_dir_derivative_sign(black_box, act, cav, concept, class_id):
		"""
		Get the sign of directional derivative.

		:param black_box: a model class instance
		:param act: activations of one bottleneck to get gradient w.r.t.
		:param cav: an instance of cav
		:param concept: a concept
		:param class_id: index of the class of interest (target) in logit layer
		:return: sign of the directional derivative
		"""
		grad = np.reshape(black_box.get_gradient(act, [class_id], cav.bottleneck), -1)
		dot_prod = np.dot(grad, cav.get_direction(concept))
		return dot_prod < 0

	@staticmethod
	def compute_tcav_score(black_box, target_class, concept, cav,
						   class_acts, run_parallel=True, num_workers=20):
		"""
		Compute TCAV score.

		:param black_box: a model class instance
		:param target_class: one target class
		:param concept: a concept
		:param cav: an instance of cav
		:param class_acts: activations of the images in the target class
		:param run_parallel: run this in parallel
		:param num_workers: number of workers if we run in parallel
		:return: TCAV score
		"""
		count = 0
		class_id = black_box.label_to_id(target_class)
		if run_parallel:
			pool = multiprocessing.Pool(num_workers)
			directions = pool.map(
				lambda act: TCAV.get_dir_derivative_sign(
								black_box, [act], cav,
								concept, class_id),
				class_acts)
			return sum(directions) / len(class_acts)
		else:
			for i in range(len(class_acts)):
				act = np.expand_dims(class_acts[i], 0)
				if TCAV.get_dir_derivative_sign(black_box, act, cav, concept, class_id):
					count += 1
			return count / len(class_acts)

	@staticmethod
	def get_dir_derivative(black_box, target_class, concept, cav, class_acts):
		"""
		Return the list of values of directional derivatives.

		:param black_box: a model class instance 
		:param target_class: a target class 
		:param concept: a concept 
		:param cav: an instance of cav 
		:param class_acts: activations of the images in the target class
		:return: list of values of directional derivatives 
		"""
		class_id = black_box.label_to_id(target_class)
		dir_derivative_vals = []
		for i in range(len(class_acts)):
			act = np.expand_dims(class_acts[i], 0)
			grad = np.reshape(
				black_box.get_gradient(act, [class_id], cav.bottleneck), -1)
			dir_derivative_vals.append(np.dot(grad, cav.get_direction(concept)))
		return dir_derivative_vals

	def __init__(self, target, concepts, bottlenecks, model_instance,
				 activation_generator, alphas, random_counterpart,
				 cav_dir=None, num_random_exp=5):
		"""
		Initialize tcav class.

		:param target: a target class
		:param concepts: a concept
		:param bottlenecks: name of a bottleneck of interest
		:param model_instance: an instance of model class
		:param activation_generator: a function handler to return activations
		:param alphas: list of hyper-parameters to run
		:param random_counterpart: the random concept to run against
							       the concepts for statistical testing
		:param cav_dir: the path to store CAVs
		:param num_random_exp: number of random experiments to compare against
		"""
		self.target = target
		self.concepts = concepts
		self.bottlenecks = bottlenecks
		self.activation_generator = activation_generator
		self.cav_dir = cav_dir
		self.alphas = alphas
		self.random_counterpart = random_counterpart
		self.black_box = model_instance
		self.model_to_rum = self.black_box.model_name

		self._process_what_to_run_expand(num_random_exp=num_random_exp)
		self.params = self.get_params()
		tf.logging.info('TCAV will %s params' % len(self.params))

	def run(self, run_parallel=False, num_workers=10):
		"""
		Run TCAV for all parameters (concept and random), write results to html

		:param run_parallel: run this in parallel
		:param num_workers: number of workers if we run in parallel
		:return results: result directory
		"""
		tf.logging.info('running %s params' % len(self.params))
		now = time.time()
		if run_parallel:
			pool = multiprocessing.Pool(num_workers)
			results = pool.map(lambda param: self._run_single_set(param), self.params)
		else:
			results = []
			for param in self.params:
				results.append(self._run_single_set(param))
		tf.logging.info('Done running %s params. Took %s seconds...' % 
						(len(self.params), time.time() - now))
		return results

	def _run_single_set(self, params):
		"""
		Run TCAV with provided for one set of (target, concepts).

		:param params: parameters to run
		:return: a dict of results (pandas dataframe)
		"""
		bottleneck = params.bottleneck
		concepts = params.concepts
		target_class = params.target_class
		activation_generator = params.activation_generator
		alpha = params.alpha
		black_box = params.black_box
		cav_dir = params.cav_dir

		tf.logging.info('running %s %s' % (target_class, concepts))

		acts = activation_generator(black_box, bottlenecks, concepts + [target_class])
		cav_hparams = CAV.default_hparams()
		cav_hparams.alpha = alpha
		cav_instance = get_or_train_cav(
			concepts, bottlenecks, acts, cav_dir=cav_dir, cav_hparams=cav_hparams)
		
		for concept in concepts:
			del acts[concept]

		a_cav_key = CAV.cav_key(concepts, bottlenecks, cav_hparams.model_type,
								cav_hparams.alpha)
		target_class_for_compute_tcav_score = target_class

		for cav_concept in concepts:
			if cav_concept is self.random_counterpart or 'random' not in cav_concept:
				i_up = self.compute_tcav_score(
					black_box, target_class_for_compute_tcav_score, cav_concept.
					cav_instance, acts[target_class][cav_instance.bottleneck])
				val_dir_derivatives = self.get_dir_derivative(
					black_box, target_class_for_compute_tcav_score, cav_concept.
					cav_instance, acts[target_class][cav_instance.bottleneck])
				result = {
					'cav_key': a_cav_key,
					'cav_concept': cav_concept,
					'target_class': target_class,
					'i_up': i_up,
					'val_dir_derivatives_abs_mean': np.mean(np.abs(val_dir_derivatives)),
					'val_dir_derivatives_mean': np.mean(val_dir_derivatives),
					'val_dir_derivatives_std': np.std(val_dir_derivatives),
					'note': 'alpha_%s' % (alpha),
					'alpha': alpha,
					'bottleneck': bottlenecks
				} 
		del acts
		return result

	def _process_what_to_run_expand(self, num_random_exp=100):
		"""
		Get tuples of parameters to run TCAV with.

		:param num_random_exp: number of random experiments to run to compare.
		"""
		target_concept_pairs = [(self.target, self.concepts)]
		all_concepts_concepts, pairs_to_run_concepts = utils.process_what_to_run_expand(
			utils.process_what_to_run_concepts(target_concept_pairs),
			self.random_counterpart, num_random_exp=num_random_exp)
		all_concepts_randoms, pairs_to_run_randoms = utils.process_what_to_run_expand(
			utils.process_what_to_run_randoms(target_concept_pairs, self.random_counterpart),
			self.random_counterpart, num_random_exp=num_random_exp)
		self.all_concepts = list(set(all_concepts_concepts + all_concepts_randoms))
		self.pairs_to_test = pairs_to_run_concepts + pairs_to_run_randoms

	def get_params(self):
		"""
		Enumerate parameters for the run function.

		:return: parameters
		"""
		params = []
		for bottleneck in self.bottlenecks:
			for target_in_test, concept_in_test in self.pairs_to_test:
				for alpha in self.alphas:
					tf.logging.info('%s %s %s %s', bottleneck, concept_in_test,
									target_in_test, alpha)
					params.append(run_params.RunParams(
						bottleneck, concept_in_test, target_in_test,
						self.activation_generator, self.cav_dir,
						alpha, self.black_box))
		return params
































































