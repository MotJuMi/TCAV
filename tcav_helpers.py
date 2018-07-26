from multiprocessing import dummy as multiprocessing
import os
import re
import numpy as np
import PIL.Image
import scipy.stats as stats
import tensorflow as tf
import json

def make_key(d, key):
	"""
	Make the key in dict d, if not already exists.
	"""
	if key not in d:
		d[key] = {}

def save_np_array(array, path):
	"""
	Save an array as numpy array
	"""
	with open(path, 'w') as f:
		np.save(f, array, allow_pickle=False)

def read_np_array(path):
	"""
	Read a saved numpy array and return.
	"""
	with open(path) as f:
		data = np.load(f)
	return data

def read_file(path):
	"""
	Read a file in path.
	"""
	with open(path, 'r') as f:
		data = f.read()
	return data

def write_file(data, path, mode='w'):
	"""
	Write data to path to cns.
	"""
	with open(path, mode) as f:
		if mode == 'a':
			f.write('\n')
		f.write(data)

def load_img_from_file(filename, shape):
	"""
	Given a filename, try to open the file.

	:param filename: location of the image file
	:param shape: shape of the image file to be scaled
	:return: image or None
	:raise: exception if the image was not the right shape.
	"""
	if not os.path.exists(filename):
		tf.logging.error('Cannot find file: {}'.format(filename))
		return None
	try:
		img = np.array(PIL.Image.open(tf.gfile.Open(filename)).resize(
			shape, PIL.Image.BILINEAR))
		img = np.float32(img) / 255
		if not (len(img.shape) == 3 and img.shape[2] == 3):
			return None
		else:
			return img

	except Exception as e:
		tf.logging.info(e)
		return None
	return img

def load_imgs_from_files(filenames, max_imgs=500, return_filenames=False,
						   shuffle=True, run_parallel=True, shape=(299, 299),
						   num_workers=100):
	"""
	Return image arrays from filenames.
	
	:param filenames: location of image files
	:param max_imgs: max number of images from filenames
	:param return_filenames: return the succeeded filenames or not
	:param shuffle: before getting max_imgs files, whether to shuffle the names
	:param run_parallel: whether to get images in run_parallel
	:param shape: desired shape of the image
	:param num_worker: number of workers in parallelization
	:return: image arrays and succeeded filenames if return_filenames=True.
	"""
	imgs = []
	filenames = filenames[:]
	if shuffle:
		np.random.shuffle(filenames)
	if return_filenames:
		final_filenames = []

	if run_parallel:
		pool = multiprocessing.Pool(num_workers)
		imgs = pool.map(lambda filename: load_img_from_file(filename, shape),
						filenames[:max_imgs])
		if return_filenames:
			final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
							   if imgs[i] is not None]
		imgs = [img for img in imgs if img is not None]
	else:
		for filename in filenames:
			img = load_img_from_file(filename, shape)
			if img is not None:
				imgs.append(img)
				if return_filenames:
					final_filenames.append(filename)
			if len(imgs) >= max_imgs:
				break

	if return_filenames:
		return np.array(imgs), final_filenames
	else:
		return np.array(imgs)

def get_acts_from_images(imgs, model, bottleneck_name):
	"""
	Run images in the model to get the activations.

	:param imgs: a list of images
	:param model: a model instance
	:param bottleneck_name: bottleneck name to get the activation from
	:return: array of activations.
	"""
	return np.asarray(model.run_imgs(imgs, bottleneck_name)).squeeze()

def get_imgs_and_acts_save(model, bottleneck_name, img_paths, acts_path,
						   img_shape, max_images=500):
	"""
	Get images from files, process acts and saves them.

	:param model: a model instance 
	:param bottleneck_name: name of the bottleneck that activations are from 
	:param img_paths: where image lives 
	:param acts_path: where to store activations 
	:param img_shape: shape of the image 
	:param max_images: max number of images to save acts_path
	:return: success or not. 
	"""
	imgs = load_imgs_from_files(img_paths, max_images, shape=img_shape)

	tf.logging.info('got %s imgs' % (len(imgs)))
	acts = get_acts_from_images(imgs, model, bottleneck_name)
	tf.logging.info('Writing acts to {}'.format(acts_path))
	with open(acts_path, 'w') as f:
		np.save(f, acts, allow_pickle=False)
	del acts
	del imgs
	return True

def process_and_load_activations(model, bottleneck_names, concepts, source_dir, 
							     acts_dir, acts=None, max_images=500):
	"""
	If activations does not already exists, make one, and returns them.

	:param model: a model instance
	:param bottleneck_names: list of bottlenecks we want to process
	:param concepts: list of concepts of interest
	:param source_dir: dir containing the concept images
	:param acts_dir: activations dir to save
	:param acts: a dict of activations if there were any pre-loaded
	:param max_images: maximum number of images for each concept
	:return acts: dict of activations
	"""
	if not os.path.exists(acts_dir):
		os.makedirs(acts_dir)
	if acts is None:
		acts = {}

	for concept in concepts:
		concept_dir = os.path.join(source_dir, concept)
		if concept not in acts:
			acts[concept] = {}
		if not os.path.exists(concept_dir):
			tf.logging.fatal('Image directory does not exists: {}'.format(concept_dir))
			raise ValueError('Image directory does not exists: {}'.format(concept_dir))

		for bottleneck_name in bottleneck_names:
			acts_path = os.path.join(acts_dir, 'acts_{}_{}'.format(
				concept, bottleneck_name))
			if not os.path.exists(acts_path):
				tf.logging.info('{} does not exist, making one...'.format(acts_path))
				img_paths = [os.path.join(concept_dir, d) for d in os.listdir(
					concept_dir)]
				get_imgs_and_acts_save(model, bottleneck_name, img_paths, acts_path,
									   model.get_image_shape()[:2],
									   max_images=max_images)
			if bottleneck_name not in acts[concept].keys():
				with open(acts_path) as f:
					acts[concept][bottleneck_name] = np.load(f).squeeze()
					tf.logging.info('Loaded {} shape {}'.format(
						acts_path,
						acts[concept][bottleneck_name].shape))
			else:
				tf.logging.info('%s, %s already exists in acts. Skipping...' % (
					concept, bottleneck_name))

	return acts 









































































