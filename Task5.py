import numpy as np
from numpy import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# function uses argparse package to allow flags to be used in the command line to alter how
# the code is run 
def flags():
	parser = argparse.ArgumentParser(description='Process files')
	
	parser.add_argument('-defuant', dest='defuant', action='store_true',
		help='runs the defuant model')
	parser.add_argument('-test_defuant', dest='test_defuant', action='store_true',
		help='runs tests to check code is crrectly changing opinions')
	parser.add_argument('-use_network', type=int, dest='use_network', action='store', 
		help='uses a random network to run the model instead of a linear array with <N> number of nodes/people')
	parser.add_argument('-beta', type=float, default=0.2, dest='beta', action='store', help='beta value (coupling parameter)')
	parser.add_argument('-threshold', type=float, default=0.2, dest='threshold', action='store', help='threshold value')

	args = parser.parse_args()
	return args


class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

	def __init__(self, nodes=None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 
 


	def make_random_network(self, N, connection_probability=0.5):
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



	def plot(self):

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

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
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
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

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
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
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
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")



def apply_network_method(args):
	network = Network()
	network_size = int(args.use_network)
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


def network_plot_first_graph(last_iteration):
	
	plt.hist(last_iteration)
	plt.xlabel('Opinion') 
	plt.show()


def network_plot_second_graph(list_of_network_arrays):
	plt.figure()

	for index, array in enumerate(list_of_network_arrays):
		index_repeated_list = np.ones(len(array)) * index
		plt.scatter(index_repeated_list, array, color='red')

	plt.xlabel('Time/Number of Iterations')
	plt.ylabel('Opinion')
	plt.show()


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

	network_plot_first_graph(final_iteration)
	network_plot_second_graph(list_of_network_arrays)





# boiler plate code begins code and initialises threshold and beta values to be fed into main
if __name__ == '__main__':
	args = flags()
	if args.use_network != None:
		network = apply_network_method(args)
		network_defuant_main(network, args)
	



