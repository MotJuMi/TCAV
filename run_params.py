class RunParams():
	"""
	Run parameters for TCAV.
	"""
	def __init__(self, bottleneck, concepts, target_class,
				 activation_generator, cav_dir, alpha,
				 black_box, overwrite=True):
		"""
		A simple class to take care of TCAV parameters.
		
		:param bottleneck: the name of a bottleneck of interest
		:param concepts: a concept
		:param target_class: a target class
		:param activation_generator: function handler that returns activations
		:param cav_dir: the path to store CAVs
		:param alpha: cav parameter
		:param black_box: an instance of a model class
		:param overwrite: if set true, rewrite any files written in the *_dir path
		"""
		self.bottleneck = bottleneck
		self.concepts = concepts
		self.target_class = target_class
		self.activation_generator = activation_generator
		self.cav_dir = cav_dir
		self.alpha = alpha
		self.overwrite = overwrite
		self.black_box = black_box

	def get_key(self):
		return '-'.join([
			str(self.bottleneck), '_'.join(self.concepts),
			'target_' + self.target_class, 'alpha_' + str(self.alpha)
		])


