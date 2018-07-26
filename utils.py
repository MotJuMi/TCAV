"""
Collects utility functions for TCAV.
"""
import numpy as np
#import tensorflow as tf
import torch
import os

def flatten(nested_list):
	"""
	Flatten a nested list.
	:param nested_list:
	"""
	return [item for l in nested_list for item in l]

def process_what_to_run_expand(pairs_to_test, random_counterpart, num_random_exp=100):
	"""
	Get concept vs. random or random vs. random pairs to run.

	:param pairs_to_test: [(target, [concept1, concept2, ...]), ...]
	:param random_counterpart: random concept that will be compared to the concept
	:param num_random_exp: number of random experiments to run against each concept
	:return all_concepts: unique set of targets/concepts
	:return new_pairs_to_test: expanded
	"""
	new_pairs_to_test = []
	for (target, concept_set) in pairs_to_test:
		new_pairs_to_test_t = []
		if len(concept_set) == 1:
			i = 0
			while len(new_pairs_to_test_t) < min(100, num_random_exp):
				if concept_set[0] != 'random500_{}'.format(i) and
				   random_counterpart != 'random500_{}'.format(i):
				    new_pairs_to_test_t.append(
				    	(target, [concept_set[0], 'random500_{}'.format(i)]))
				i += 1
		elif len(concept_set) > 1:
			new_pairs_to_test_t.append((target, concept_set))
		else:
			tf.logging.info('PAIR NOT PROCESSED')
		new_pairs_to_test.extend(new_pairs_to_test_t)

	all_concepts = list(set(flatten([cs + [tc] for tc, cs in new_pairs_to_test])))

	return all_concepts, new_pairs_to_test

def process_what_to_run_concepts(pairs_to_test):
	"""
	Process concepts and pairs to test.

	:param pairs_to_test: a list of concepts to be tested and a target
	:return: pairs to test
	"""
	pairs_for_sstesting = []
	for pair in pairs_to_test:
		for concept in pair[1]:
			pairs_for_sstesting.append([pair[0], [concept]])
	return pairs_for_sstesting

def process_what_to_run_randoms(pairs_to_test, random_counterpart):
	"""
	Process concepts and pairs to test.

	:param pairs_to_test: a list of concepts to be tested and a target
	:param random_counterpart: random concept that will be compared to the concept
	"""
	pairs_for_sstesting = []
	targets = list(set([pair[0] for pair in pairs_to_test]))
	for target in targets:
		pairs_for_sstesting_random.append([target, [random_counterpart]])
	return pairs_for_sstesting_random

def print_results(results):
	"""
	Write results.
	:param results: dict of results from TCAV runs.
	"""
	result_summary = {
		'random': []
	}
	for result in results:
		if 'random' in results['cav_concept']:
			result_summary['random'].append(result)
		else:
			if result['cav_concept'] not in result_summary:
				result_summary[result['cav_concept']] = []
			result_summary[result['cav_concept']].append(result)
	random_i_ups = [item['i_up'] for item in result_summary['random']]

	for concept in result_summary:
		if 'random' is not concept:
			i_ups = [item['i_up'] for item in result_summary[concept]]
			print('%s: TCAV score: %.2f (+- %.2f), random was %.2f' % (
				concept, np.mean(i_up), np.std(i_ups), np.mean(random_i_ups)))

def mkdir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)
























