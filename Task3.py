import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
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
		#print(self.nodes)

##	def make_ring_network(self, N, neighbour_range=1):
##		#Your code  for task 4 goes here
##
##	def make_small_world_network(self, N, re_wire_prob=0.2):
##		#Your code for task 4 goes here

	def plot(self):
		colour = ['gold','red','blue','black','cyan','orange','green']
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
			if i >= 7:
				circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color="purple")
			else:
				circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=colour[i])
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
N = 0
for element in sys.argv: #Iterate through the flags
	if element == "-test_network":
		test_networks()
	if element == "-network":
		num_nodes = int(sys.argv[N+1]) #Find the number of nodes written with the flag
		network = Network() #Create a network
		network.make_random_network(num_nodes)  #Create a random network with N number of nodes
		print("Mean degree:", network.get_mean_degree(),"\nAverage path length:", network.get_mean_path_length(),"\nClustering co-efficient:", network.get_mean_clustering())
	N += 1
