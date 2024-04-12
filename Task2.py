import numpy as np
from numpy import random
import argparse


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

	test_person_1_and_2 = Individual(person_1, person_2)
	test_person_1_and_3 = Individual(person_1, person_3)
	test_person_1_and_2.update(threshold, beta)


	if args.test_defuant == True:
		assert round(test_person_1_and_2.person, 2) == updated_person_1
		assert round(test_person_1_and_2.neighbour, 2) == updated_person_2
		assert round(test_person_1_and_3.person, 2) == person_1
		assert round(test_person_1_and_3.neighbour, 2) == person_3

		return print('Tests passed')


# function that randomly selects idividual of the population
# and a random one of its two neighbours
# if decider is 0 function compares opinion to neighbour on the left
# and if it is 1 it takes the neighbour to the right
def selector():
	person_index = random.randint(9)
	decider = random.randint(2)
	if decider == 0 and person_index == 0:
		neighbour_index = 9
	elif decider == 0:
		neighbour_index = person_index - 1
	elif person_index == 9:
		neighbour_index = 0
	else:
		neighbour_index = person_index + 1
	results = [decider, person_index, neighbour_index]
	return results


def opinion_update(array, person_index, neighbour_index, threshold, beta):

	person_value = array[person_index]
	neighbour_value = array[neighbour_index]

	person = Individual(person_value, neighbour_value)
	person.update(threshold, beta)
	person_new_value = person.person
	neighbour_new_value = person.neighbour

	results = [person_index, person_new_value, neighbour_index, neighbour_new_value]

	return results


def main(flags_results, threshold, beta):
	args = flags_results
	if args.test_defuant == True:
		test(args)
	print('Threshold =', threshold)
	print('Beta (coupling parameter) =', beta)
	if args.defuant == True:
		array = np.random.rand(10)
		print(array)
		for i in range(1000):
			selector_results = selector()
			
			person_index = selector_results[1]
			neighbour_index = selector_results[2]
			opinion_update_results = opinion_update(array, person_index, neighbour_index, threshold, beta)
			
			array[person_index] = opinion_update_results[1]
			array[neighbour_index] = opinion_update_results[3]
		print(array)
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




