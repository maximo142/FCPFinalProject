import numpy as np
from numpy import random

threshold = 0.2
beta = 0.35

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


def main(threshold, beta):
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
	


main(0.2, 0.35)




