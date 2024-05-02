import numpy as np
from numpy import random
import argparse
import matplotlib.pyplot as plt
import copy


def flags():
	"""
	function uses argparse package to allow flags to be used in the command line to alter how the code is run 
	"""
	parser = argparse.ArgumentParser(description='Process files')
	
	parser.add_argument('-defuant', dest='defuant', action='store_true',
		help='runs the defuant model')
	parser.add_argument('-test_defuant', dest='test_defuant', action='store_true',
		help='runs tests to check code is crrectly changing opinions')
	parser.add_argument('-beta', type=float, default=0.2, dest='beta', action='store', help='beta value (coupling parameter)')
	parser.add_argument('-threshold', type=float, default=0.2, dest='threshold', action='store', help='threshold value')

	args = parser.parse_args()
	return args


def test():
	
	"""
	tests are run between 3 people with quantified opinions, difference in opinions of persons 1 and 2 are below 
	the threshold and so each get updated, this is checked against the known values they should go to 
	which have been worked out for the sake of the test. 
	opinions between persons 1 and 3 should not change as their difference in score is over the threshold
	so the final two assertions make sure they go unchanged after being fed into the 'opinion update' function.
	numbers outputted from function had to be rounded as they potentially had floating point error which 
	would make the tests falsely fail

	"""

	person_1 = 0.2
	person_2 = 0.3
	person_3 = 0.5

	threshold = 0.2
	beta = 0.2

	updated_person_1 = 0.22
	updated_person_2 = 0.28

	test_array = [person_1, person_2, person_3]
	opinion_update_1 = opinion_update(test_array, 0, 1, threshold, beta)
	opinion_update_2 = opinion_update(test_array, 0, 2, threshold, beta)

	person_1_first_calculation = opinion_update_1[1]
	person_2_calculation = opinion_update_1[3]
	person_1_second_calculation = opinion_update_2[1]
	person_3_calculation = opinion_update_2[3]


	if args.test_defuant == True:
		assert round(person_1_first_calculation, 2) == updated_person_1
		assert round(person_2_calculation, 2) == updated_person_2
		assert round(person_1_second_calculation, 2) == person_1
		assert round(person_3_calculation, 2) == person_3

		return print('Tests passed')


def selector(array):

	"""
	function that randomly selects individual from the population and a random one of its two neighbours
	if decider is 0 function compares opinion to neighbour on the left
	and if it is 1 it takes the neighbour to the right

	"""

	number_of_indicies = len(array) - 1
	person_index = random.randint(number_of_indicies)
	decider = random.randint(2)

	if decider == 0 and person_index == 0:
		neighbour_index = number_of_indicies
	elif decider == 0:
		neighbour_index = person_index - 1
	elif person_index == number_of_indicies:
		neighbour_index = 0
	else:
		neighbour_index = person_index + 1

	results = [person_index, neighbour_index]

	return results


def opinion_update(array, person_index, neighbour_index, threshold, beta):
	"""
	function applies formula to the two randomly selected people's opinions providing they lie
	close enough together to satisfy the threshold limit
	"""

	person_value = array[person_index]
	neighbour_value = array[neighbour_index]

	if person_value >= neighbour_value:
		difference = person_value - neighbour_value	
	else:
		difference = neighbour_value - person_value
	
	if difference < threshold:		
		person_new_value = person_value + (beta * (neighbour_value - person_value))
		neighbour_new_value = neighbour_value + (beta * (person_value - neighbour_value))
	else:
		person_new_value = person_value
		neighbour_new_value = neighbour_value

	results = [person_index, person_new_value, neighbour_index, neighbour_new_value]

	return results


def iterate_population(threshold, beta):
	"""
	a population size of 150 was defaulty chosen and for the code to select neighbours 10000 times since there 
	was no instruction for what those numbers should be nor that it should become a flagged input to 
	enter in the command line
	"""
	population size = 150
	array = np.random.rand(population_size)
	time_duration = 10000
	list_of_arrays = [array]

	for t in range(time_duration):
		selector_results = selector(array)
		person_index = selector_results[0]
		neighbour_index = selector_results[1]

		opinion_update_results = opinion_update(array, person_index, neighbour_index, threshold, beta)
		array[person_index] = opinion_update_results[1]
		array[neighbour_index] = opinion_update_results[3]

		list_of_arrays.append(copy.deepcopy(array))

	results = [array, list_of_arrays]

	return results


def plot_first_graph(array):
	"""
	plots histogram of final opinions
	"""
	plt.hist(array)
	plt.xlabel('Opinion') 
	plt.show()



def plot_second_graph(list_of_arrays):
	"""
	plots scatter graph of how the opinions change/converge over time
	NOTE: this graph takes slightly longer than expected to be displayed, from 30-60 seconds roughly
	"""
	plt.figure()

	for index, array in enumerate(list_of_arrays):
		index_repeated_list = np.ones(len(array)) * index
		plt.scatter(index_repeated_list, array, color='red')

	plt.xlabel('Time/Number of Iterations')
	plt.ylabel('Opinion')
	plt.show()


def defuant_main(args, threshold, beta):
	"""
	main function organises the code and calls the correct functions to produce what 
	is called for by the flags input on the command line
	"""

	if args.test_defuant:
		test()

	print('Threshold =', threshold)
	print('Beta (coupling parameter) =', beta)

	if args.defuant:

		iteration_results = iterate_population(threshold, beta)
		plot_first_graph(iteration_results[0])
		plot_second_graph(iteration_results[1])

	else:
		print('Please provide "-defaunt" flag if you want the code to run')


if __name__ == '__main__':
	"""
	boiler plate code begins code and initialises threshold and beta values to be fed into main
	"""
	args = flags()
	threshold = args.threshold
	beta = args.beta

	defuant_main(args, threshold, beta)






