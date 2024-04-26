import numpy as np
from numpy import random
import argparse
import matplotlib.pyplot as plt
import copy

# function uses argparse package to allow flags to be used in the command line to alter how
# the code is run 
def flags():
	parser = argparse.ArgumentParser(description='Process files')
	
	parser.add_argument('-defuant', dest='defuant', action='store_true',
		help='runs the defuant model')
	parser.add_argument('-test_defuant', dest='test_defuant', action='store_true',
		help='runs tests to check code is crrectly changing opinions')
	parser.add_argument('-beta', dest='beta', action='store', help='beta value (coupling parameter)')
	parser.add_argument('-threshold', dest='threshold', action='store', help='threshold value')

	args = parser.parse_args()
	return args


# class created to apply the function to the two opinions
# and either bring them closer together or further apart in value
# if they are close enough according to the threshold
class Individual:
	def __init__(self, initial_value, neighbour_value):
		self.person = initial_value
		self.neighbour = neighbour_value

	def update(self, threshold, beta):
		if self.person >= self.neighbour:
			difference = self.person - self.neighbour
			
		elif self.neighbour > self.person:
			difference = self.neighbour - self.person
			
		person_new_value = self.person + (beta * (self.neighbour - self.person))
		neighbour_new_value = self.neighbour + (beta * (self.person - self.neighbour))

		if difference < threshold:
			self.person = person_new_value
			self.neighbour = neighbour_new_value


# tests are run between 3 people with scored opinions, opinions of persons 1 and 2 are below 
# the threshold and so each get updated, this is checked against the knoiwn values they should go to 
# which have been worked out for the sake of the test. 
# opinions between persons 1 and 3 should not change as their difference in score is over the threshold
# so the final two assertions make sure they go uchanged after being fed into the class
# numbers outputted from class had to be rounded as they potentially had floating point error which 
# would make it falsely fail the test
def test(flags_results):
	args = flags_results
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

	first_person_1_new_calculation = opinion_update_1[1]
	person_2_new_calculation = opinion_update_1[3]
	second_person_1_new_calculation = opinion_update_2[1]
	person_3_new_calculation = opinion_update_2[3]


	if args.test_defuant == True:
		assert round(first_person_1_new_calculation, 2) == updated_person_1
		assert round(person_2_new_calculation, 2) == updated_person_2
		assert round(second_person_1_new_calculation, 2) == person_1
		assert round(person_3_new_calculation, 2) == person_3

		return print('Tests passed')


# function that randomly selects idividual of the population
# and a random one of its two neighbours
# if decider is 0 function compares opinion to neighbour on the left
# and if it is 1 it takes the neighbour to the right
def selector(array):
	number_of_indecies = len(array) - 1
	person_index = random.randint(number_of_indecies)
	decider = random.randint(2)
	if decider == 0 and person_index == 0:
		neighbour_index = number_of_indecies
	elif decider == 0:
		neighbour_index = person_index - 1
	elif person_index == number_of_indecies:
		neighbour_index = 0
	else:
		neighbour_index = person_index + 1
	results = [decider, person_index, neighbour_index]
	return results


# function applies formula to the two randomly selected people's opinions providing they lie
# close enough together to satisfy the threshold limit
def opinion_update(array, person_index, neighbour_index, threshold, beta):

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


def plot_second_graph(list_of_arrays):
	plt.figure()
	for index, array in enumerate(list_of_arrays):
		
		index_repeated_list = np.ones(len(array)) * index
		
		plt.scatter(index_repeated_list, array, color='red')
	plt.show()



def main(flags_results, threshold, beta):
	args = flags_results
	if args.test_defuant == True:
		test(args)
	print('Threshold =', threshold)
	print('Beta (coupling parameter) =', beta)
	if args.defuant == True:
		array = np.random.rand(150)
		print(array)
		time_duration = 10000
		list_of_arrays = [array]
		for i in range(time_duration):
			selector_results = selector(array)
			
			person_index = selector_results[1]
			neighbour_index = selector_results[2]
			opinion_update_results = opinion_update(array, person_index, neighbour_index, threshold, beta)
			print(person_index, neighbour_index)
			array[person_index] = opinion_update_results[1]
			array[neighbour_index] = opinion_update_results[3]
			list_of_arrays.append(copy.deepcopy(array))

		
		plt.hist(array)
		plt.xlabel('Opinion') 
		plt.show()
		plot_second_graph(list_of_arrays)
		print(list_of_arrays[0])
		print(list_of_arrays[1])
		print(list_of_arrays[2])

	else:
		print('Please provide "-defaunt" flag if you want the code to run')



if __name__ == '__main__':
	flags_results = flags()
	if flags_results.threshold == None:
		flags_results.threshold = 0.2
	if flags_results.beta == None:
		flags_results.beta = 0.2
	threshold = float(flags_results.threshold)
	beta = float(flags_results.beta)
	main(flags_results, threshold, beta)



