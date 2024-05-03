import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import argparse
from numpy import random
import math
from math import exp 
import copy

# function uses argparse package to allow flags to be used in the command line to alter how
# the code is run 
def flags():
	parser = argparse.ArgumentParser(description='Process files')
	
	parser.add_argument('-test_ising', dest='test_ising', action='store_true',
                help='Runs the test function for the ising model')
	parser.add_argument('-ising_model', dest='ising_model', action='store_true',
                help='runs the ising model')
	parser.add_argument('-alpha', type=float, default=1, dest='alpha', action='store', help='Choose the value for Alpha')
	parser.add_argument('-external', type=float, default=0, dest='external', action='store', help='Choose value for H')
	
	parser.add_argument('-defuant', dest='defuant', action='store_true',
		help='runs the defuant model')
	parser.add_argument('-test_defuant', dest='test_defuant', action='store_true',
		help='runs tests to check code is crrectly changing opinions')

	parser.add_argument('-use_network', type=int, dest='use_network', action='store', 
		help='uses a random network to run the model instead of a linear array with <N> number of nodes/people')
	parser.add_argument('-beta', type=float, default=0.2, dest='beta', action='store', help='beta value (coupling parameter)')
	parser.add_argument('-threshold', type=float, default=0.2, dest='threshold', action='store', help='threshold value')

	parser.add_argument('-test_network', dest='test_network', action='store_true',
                help='runs tests to check if network runs correctly')
	parser.add_argument('-network', type=int, dest='network', action='store',
                help='Creates a random network with N number of nodes')

	parser.add_argument('-ring_network', type=int, dest='ring_network', action='store', help='create a ring network')
	parser.add_argument('-small_world', type=int, dest='small_world', action='store', help='create a small-worlds network with default parameters')
	parser.add_argument('-re_wire', type=float, dest='re_wire', action='store', 
		help='create a small worlds network with the value given being the re-wiring probability')

	args = parser.parse_args()
	return args

class Node:

	def __init__(self, value, number,connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

	def __init__(self, nodes=None):
		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes
	# TASK 3
	def breadth_first_search(self, start_node_index):
		"""
		Performs breadth-first search starting from the given start node index.

		Parameters:
		start_node_index (int): The index of the start node for the BFS.

		Returns:
		list: A list containing the distances of each node from the start node.
		"""

		visited = [False] * len(self.nodes) # List to track visited nodes
		distance = [0] * len(self.nodes) # List to store distances from the start node

		queue = [] # Queue for BFS traversal
		queue.append(start_node_index) # Enqueue the start node
		visited[start_node_index] = True # Mark start node as visited

		while queue:
			current_node_index = queue.pop(0) # Dequeue a node from the queue

			# Iterate over neighbors of the current node
			for neighbor_index, connected in enumerate(self.nodes[current_node_index].connections):
				if connected == 1 and not visited[neighbor_index]:
					queue.append(neighbor_index) # Enqueue unvisited neighbor
					visited[neighbor_index] = True # Mark neighbor as visited
					distance[neighbor_index] = distance[current_node_index] + 1 # Update distance to neighbor

		return distance

	def get_mean_degree(self):
		#Your code  for task 3 goes here
		degree = 0 #Create a variable that adds all connections
		for node in range(len(self.nodes)): #Iterate through all nodes
			for links in self.nodes[node].connections: #Iterate through the connections of the node
				degree += links #Add the links
		return degree/len(self.nodes)  #Divide the degree with the number of nodes to get the mean

	def get_mean_clustering(self):
		#Your code for task 3 goes here
		poscnc2 = 0
		total_cluster_coefficient = 0
		Break = False
		for idx in range(len(self.nodes)): #Iterate through nodes
			for cnc in range(len(self.nodes[idx].connections)): #Iterate through connections
				if cnc == idx: #If the index of connection is the same as node ignore and continue
					continue
				if self.nodes[idx].connections[cnc] == 0: #If there the network is not fully connected change Break to true
					Break = True
					break
			if Break == True:
				break
		if Break == False:  #If Break remained False then return 1.0 as mean cluster
			mean_cluster_coefficient = 1.0
			return mean_cluster_coefficient
		for index in range(len(self.nodes)): #Iterate through nodes
			adjnodes = 0
			ones = []
			start = 1
			poscnc = 0
			for cnc in range(len(self.nodes[index].connections)): #Iterate through the connections of the node
				if self.nodes[index].connections[cnc] == 0: #If there is not a connection then continue 
					continue
				else: #If there is a connection add the index of the node to a list called ones and increase the value of the adjacent nodes
					adjnodes += 1
					ones.append(cnc)
			# Calculate the clustering coefficient for the current node
			num_ones = len(ones)
			if num_ones >= 2:  # Only calculate if the node has at least 2 neighbors
				for n in range(num_ones):
					for cnc2 in range(start, num_ones):
						# Count the number of possible connections forming triangles
						if self.nodes[ones[n]].connections[ones[cnc2]] == 1:
							poscnc += 1

				# Calculate the clustering coefficient using the formula
				clustering_coefficient = poscnc / (num_ones * (num_ones - 1))
				total_cluster_coefficient += clustering_coefficient

			start += 1  # Increment start index for the next iteration
        
		# Calculate the mean clustering coefficient
		if total_cluster_coefficient == 0:  # Check if there are no nodes with at least 2 neighbors
			return 0  # Return 0 to avoid division by zero
		mean_cluster_coefficient = total_cluster_coefficient / len(self.nodes)
    
		return mean_cluster_coefficient

	def get_mean_path_length(self):
		"""
		Calculates the mean path length of the network using breadth-first search.

		Returns:
		float: The mean path length of the network.
		"""
		#Your code for task 3 goes here 
		total_path_length = 0

		# Iterate over each node in the network
		for start_node_index in range(len(self.nodes)):
			distances = self.breadth_first_search(start_node_index) # Perform BFS from current node
			total_path_length += sum(distances) # Add up distances to other nodes from current node

		# Calculate mean path length by dividing total path length by the number of paths and rounding the result to 15 decimal places
		return round(total_path_length / (len(self.nodes) * (len(self.nodes) - 1)), 15)


	def make_random_network(self,N,connection_probability=0.5):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1
		# END OF TASK 3

	# TASK 4
	def make_ring_network(self, N, neighbour_range=1):
		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes): # Iterate over each node
			cursor = index

			for forward in range(neighbour_range): # Increment the cursor to point at forward adjacent nodes and enable the connection between them
				cursor += 1
				if cursor == N:
					cursor = 0

				node.connections[cursor] = 1
				self.nodes[cursor].connections[index] = 1

			cursor = index # Reset the cursor

			for backward in range(neighbour_range): # Decrement the cursor to point at backward adjacent nodes and enable the connection between them
				cursor -= 1
				if cursor == -1:
					cursor = N - 1

				node.connections[cursor] = 1
				self.nodes[cursor].connections[index] = 1


	def make_small_world_network(self, N, re_wire_prob=0.2):
		self.make_ring_network(N, 2) # Start with a ring network

		for (index, node) in enumerate(self.nodes):
			for (edge_index, edge) in enumerate(node.connections): # Go over each edge and attempt to rewire
				if edge == 1: # If the edge is already connected to something
					if np.random.random() < re_wire_prob:
						while True: # Keep attempting this block of code until a valid condition is met
							new_connection_index = np.random.randint(0, N)

							# If the new connection index is not equal to the current node (no self-connections allowed) and not equal to the connection
	   						# index it's already connected to (no repeat connections allowed), and it's not an already existing connection (no repeat
		  					# connections allowed), then proceed
							if new_connection_index != index and new_connection_index != edge_index and \
								node.connections[new_connection_index] != 1 and \
								self.nodes[new_connection_index].connections[
								index] != 1:
								node.connections[new_connection_index] = 1 # Complete the rewire to the new node and unwire from the old node
								self.nodes[new_connection_index].connections[index] = 1

								node.connections[edge_index] = 0
								self.nodes[edge_index].connections[index] = 0
								break


	def plot_task4(self):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.5*network_radius, 1.5*network_radius])
		ax.set_ylim([-1.5*network_radius, 1.5*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 30, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
	# END OF TASK 4

	def plot(self):

		colour = [["pink","violet","hotpink"],["mediumorchid","darkviolet","indigo"]]
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes			
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)
			if self.nodes[i].value >= 0.5:
				color = colour[0]
				if self.nodes[i].value >= 0.85:
					color = color[0]
				elif self.nodes[i].value >= 0.65:
					color = color[1]
				else:
					color = color[2]
			else:
				color = colour[1]
				if self.nodes[i].value <= 0.15:
					color = color[2]
				elif self.nodes[i].value <= 0.35:
					color = color[1]
				else:
					color = color[0]
			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=color)
			value = round(self.nodes[i].value, 4)
			if node_x < -50:
				plt.text(node_x-25,node_y-15, value)
			elif node_y < 0:
				plt.text(node_x-10, node_y-20,value)
			else:
				plt.text(node_x+5, node_y+10,value)
			ax.add_patch(circle)
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_mean_clustering()==0), network.get_mean_clustering()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==5), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()

	print("All tests passed")



# TASK 1
def create_array(rows, cols):
	'''
	Generate a 2D grid for the Ising model with significant size.
	Each cell in the grid is randomly set to either -1 or 1
	Returns:
		numpy.ndarray: A 2D array with randomly assigned spins of -1 or 1.
	'''
	if not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0:  # make sure the inputs are
		# positive intgers
		raise ValueError("Rows and columns must be positive integers.")
	return np.random.choice([-1, 1], size=(rows, cols))


a1 = create_array(100, 100)  # creat 100 by 100 array called a1

def calculate_agreement(population, row, col, external=0):
	'''
	This function should return the extent to which a cell agrees with its neighbours.
	Inputs:
		population (numpy array)
		row (int)
		col (int)
		H (float): external magnetic field acting as a bias
	Returns:
		agreement (float)
	'''
	n_rows, n_cols = population.shape  # Get the dimensions of the array

	# Indices of the direct neighbors
	neighbors_indices = [
		((row - 1) % n_rows, col),  # Up
		(row, (col - 1) % n_cols),  # Left
		(row, (col + 1) % n_cols),  # Right
		((row + 1) % n_rows, col)  # Down
	]
	neighbors_values = [population[r, c] for r, c in
		neighbors_indices]  # takeout the neighbors values and put in a list
	cell_current_value = population[row, col]

	# agreement as the sum of the products of neighboring values and the cell's value
	agreement = sum(cell_current_value * value for value in neighbors_values) + (cell_current_value * external)

	return agreement


def ising_step(population, alpha, external):
	'''
	This function will perform a single update of the Ising model
	Inputs:
		population (numpy array)
		H (float): external magnetic field
		alpha (float): inverse temperature, controls stochastic flipping
	'''

	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col = np.random.randint(0, n_cols)

	agreement = calculate_agreement(population, row, col, external)

	if agreement < 0:
		population[row, col] *= -1
	if alpha != 0: # make sure we did not get mathmatic erorr
		probability = math.exp(-agreement / alpha)  # use the probalitity formula from Task 1 (condition 2)
	else:
		probability = 0
	random_float = random.random()  # generate a random float from 0 to 1 to take the correct porbability (condition 2)

	# condition 2: even when agreement is  positive, this condition we choose to accept flips that reduce agreement with probability
	if agreement > 0 and probability > random_float:
		population[row, col] *= -1


def plot_ising(im, population):
	'''
	Display a plot of the Ising model by updating the plot image with the new population state.

	Args:
	im (matplotlib.image.AxesImage): The image object returned by a previous imshow call.
	population (numpy array): The grid of the Ising model where cells are either -1 or 1.
	'''
	# Map Ising model values to color values: 255 for -1, and 0 for 1
	new_im = np.array([[0 if val == 1 else 255 for val in row] for row in population], dtype=np.uint8)
	im.set_data(new_im)
	plt.draw()
	plt.pause(0.1)  # pause to update the plot


def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''
	print("Testing ising model calculations")
	population = -np.ones((3, 3))
	assert calculate_agreement(population, 1, 1) == 4, "Test 1"

	population[1, 1] = 1.
	assert calculate_agreement(population, 1, 1) == -4, "Test 2"

	population[0, 1] = 1.
	assert calculate_agreement(population, 1, 1) == -2, "Test 3"

	population[1, 0] = 1.
	assert calculate_agreement(population, 1, 1) == 0, "Test 4"

	population[2, 1] = 1.
	assert calculate_agreement(population, 1, 1) == 2, "Test 5"

	population[1, 2] = 1.
	assert calculate_agreement(population, 1, 1) == 4, "Test 6"

	# Testing external pull (H)
	population = -np.ones((3, 3))
	assert calculate_agreement(population, 1, 1, 1) == 3, "Test 7"
	assert calculate_agreement(population, 1, 1, -1) == 5, "Test 8"
	assert calculate_agreement(population, 1, 1, 10) == -6, "Test 9"
	assert calculate_agreement(population, 1, 1, -10) == 14, "Test 10"

	print("Tests passed")


def ising_model(population=a1, alpha=1, external=0):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_axis_off()
	im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

	# Iterating an update 100 times
	for frame in range(100):
		# Iterating single steps 1000 times to form an update
		for step in range(1000):
			ising_step(population, alpha, external)
		print('Step:', frame, end='\r')
		plot_ising(im, population)
# TASK 2
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
	population_size = 100
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

	if args.defuant:

		print('Threshold =', threshold)
		print('Beta (coupling parameter) =', beta)
		iteration_results = iterate_population(threshold, beta)
		plot_first_graph(iteration_results[0])
		plot_second_graph(iteration_results[1])


# TASK 4 continued

def task4plot(network):
	network.plot_task4()
	plt.show()


def apply_ring_network_method(args):
	network = Network()
	network_size = args.ring_network
	network.make_ring_network(network_size)

	task4plot(network)


def apply_small_world_method(args):
	network = Network()
	network_size = args.small_world
	re_wire_prob = args.re_wire
	network.make_small_world_network(network_size, re_wire_prob)

	task4plot(network)



# TASK 5
def apply_network_method(args):
	network = Network()
	network_size = args.use_network
	network.make_random_network(network_size, connection_probability=0.5)

	return network



def selector_network(network):
	total_number_of_indices = len(network.nodes) - 1
	person_index = random.randint(total_number_of_indices)

	person_attributes = network.nodes[person_index]
	person_connections_list = person_attributes.connections
	person_value = person_attributes.value


	dict_connections = {}
	for index, connection in enumerate(person_connections_list):
		if index != person_index:
			if connection == 1:
				key = index
				value = network.nodes[index].value
				dict_connections[key] = value

	
	connections_number_of_indices = len(dict_connections)
	neighbour_index = random.randint(connections_number_of_indices)
	for i, real_index in enumerate(dict_connections):
		if i == neighbour_index:
			applied_neighbour_index = real_index
	

	neighbour_value = dict_connections[applied_neighbour_index]
	

	results = [person_index, person_value, neighbour_index, neighbour_value]

	return results

	


# function applies formula to the two randomly selected people's opinions providing they lie
# close enough together to satisfy the threshold limit
def network_opinion_update(person_value, neighbour_value, threshold, beta):

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

	results = [person_new_value, neighbour_new_value]

	return results



def update_network(network, person_index, new_person_value, neighbour_index, new_neighbour_value):
	network.nodes[person_index].value = new_person_value
	network.nodes[neighbour_index].value = new_neighbour_value

	return network



# I have chosen a population size of 150 and for the code to select neighbours 10000 times since there 
# was no instruction for what those numbers should be nor that it should become a flagged input to 
# enter in the command line
def network_iterate_population(network, threshold, beta):
	
	time_duration = 100
	
	list_of_network_arrays = []
	for t in range(time_duration):
		selector_results = selector_network(network)
		person_index = selector_results[0]
		person_value = selector_results[1]
		neighbour_index = selector_results[2]
		neighbour_value = selector_results[3]

		opinion_update_results = network_opinion_update(person_value, neighbour_value, threshold, beta)

		new_person_value = opinion_update_results[0]
		new_neighbour_value = opinion_update_results[1]

		update_network(network, person_index, new_person_value, neighbour_index, new_neighbour_value)
		value_list = []
		network.plot()
		plt.pause(0.2)
		plt.close()
		plt.show()
		for index in range(len(network.nodes)):

			value_list.append(network.nodes[index].value)
		list_of_network_arrays.append(value_list)

	return list_of_network_arrays



def network_defuant_main(network, args):
	
	threshold = args.threshold
	beta = args.beta
	print('Threshold =', threshold)
	print('Beta (coupling parameter) =', beta)

	list_of_network_arrays = network_iterate_population(network, threshold, beta)
	final_iteration = list_of_network_arrays[-1]



# boiler plate code begins code and processing of flags
if __name__ == '__main__':
	args = flags()
	#task 1
	if args.test_ising:
		test_ising()

	if args.ising_model:
		print(args.alpha, args.external)
		# Run the ising_model function with provided alpha and external field values
		ising_model(population=a1, alpha=args.alpha, external=args.external)

			
	#task 5
	if args.defuant and args.use_network != None:
		network = apply_network_method(args)
		network_defuant_main(network, args)

	#task 2
	elif args.use_network == None:
		threshold = args.threshold
		beta = args.beta
		defuant_main(args, threshold, beta)
	#task 3	
	if args.test_network:
		test_networks()
	if args.network != None:
		tsk3 = Network() #Create a network
		tsk3.make_random_network(args.network)
		tsk3.plot()
		plt.show()
		print("Mean degree:", tsk3.get_mean_degree(),"\nAverage path length:", tsk3.get_mean_path_length(),"\nClustering co-efficient:", tsk3.get_mean_clustering())

	#task 4
	if args.ring_network != None:
		apply_ring_network_method(args)

	if args.small_world != None and args.re_wire != None:
		apply_small_world_method(args)

	elif args.small_world != None:
		args.re_wire = 0.2
		apply_small_world_method(args)

	


