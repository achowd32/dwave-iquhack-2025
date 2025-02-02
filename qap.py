import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

class QAP:
    def __init__(self, flow: np.ndarray, dist: np.ndarray, num_closets = None, given_sampler = None):
        """
        Initialise the QAP class at time t = 0. Sampler, if given, must be able to handle QUBO instances.
        If considering the closets/departments distinction, an integer must be passed into num_closets.
        The first num_closets rows of the flow matrix will be designated closets.
        """
        #ensure flow or distance are Numpy arrays
        if not isinstance(flow, np.ndarray):
            raise TypeError("Flow input must be a NumPy array.")
        if not isinstance(dist, np.ndarray):
            raise TypeError("Dist input must be a NumPy array.")  

        #ensure both arrays are 2D
        if flow.ndim != 2 or dist.ndim != 2:
            raise ValueError("Both arrays must be 2D.")
        
        #ensure both dimensions are equal for each array
        if flow.shape[0] != flow.shape[1] or dist.shape[0] != dist.shape[1]:
            raise ValueError("Each array must be a square matrix (rows = columns).")
        
        #ensure both arrays have the same shape
        if flow.shape != dist.shape:
            raise ValueError("Both arrays must have the same dimensions.")
        
        #define flow, dist, and size objects
        self.flow = flow
        self.dist = dist
        self.size = flow.shape[0]

        #define time and location (state) objects
        self.time = 0
        self.prev_state = None
        self.cur_state = None

        #define num_closets
        if not isinstance(num_closets, int) and num_closets is not None:
            raise TypeError("Num_closets must either be None or integer")
        
        if isinstance(num_closets, int) and num_closets >= self.size:
            raise ValueError("If closets are specified, there must be fewer closets than total facilities")

        self.num_closets = num_closets

        #define qubo and row/column penalty; set to none and defined properly by generate_qubo()
        self.qubo = None
        self.row_col_penalty = None
        self.generate_qubo()

        #define default sampler, set to a simulator if no sampler specified
        if given_sampler is None:
            self.default_sampler = SimulatedAnnealingSampler()
        else:
            self.default_sampler = given_sampler
        
    def generate_qubo(self, penalty = None):
        """
        Create a QUBO matrix from the flow and distance matrices of the class.
        Does NOT account for transition penalties or facility-type penalties.
        """
        #tensor product (specifically Kronecker product) of flow and distance matrices
        Q = np.kron(self.flow, self.dist)
        self.row_col_penalty = np.sum(Q)

        #initialise penalty if not given
        if penalty is None:
            penalty = self.row_col_penalty

        #define row and column constraints to ensure no facility is assigned
        #more than one location, and no location has more than one facility
        constraint_groups = []
        N = self.size
        for i in range(N):
            constraint_groups.append([i * N + m for m in range(N)])
        for m in range(N):
            constraint_groups.append([m + N * i for i in range(N)])

        #apply the row/column penalty
        for group in constraint_groups:
            for i in range(len(group)):
                for j in range(i, len(group)):  # Upper triangular terms
                    var_i, var_j = group[i], group[j]
                    if i == j:
                        Q[var_i, var_j] -= penalty # Linear penalty term
                    else:
                        Q[var_i, var_j] += penalty  # Quadratic interaction term
                        Q[var_j, var_i] += penalty  # Ensure symmetry
        
        #apply the closet/department penalty
        if self.num_closets is not None:
            for i in range(N):
                if i < self.num_closets:
                    for m in range(self.num_closets, N):
                        Q[i * N + m, i * N + m] += self.row_col_penalty
                else:
                    for m in range(self.num_closets):
                        Q[i * N + m, i * N + m] += self.row_col_penalty

        self.qubo = Q
        return Q
    
    def time_init(self):
        """
        Initialises locations, allows for time evolution
        """
        #error if time and system state have already been initialised
        if self.cur_state is not None:
            raise RuntimeError("Initialisation has already occurred, use time_evolve() instead")
        
        #sample with the default_sampler
        resp = self.sample_qap(shots = 5000).first.sample
        resp_arr = np.array(list(resp.values())).reshape((self.size, self.size))

        #set current state to the best state from sample
        self.cur_state = resp_arr
        return resp_arr
    
    def time_evolve(self, new_flow: np.ndarray, penalty = None):
        """
        Evolve system according to new flow matrix. Default time step of 1, default qubo penalty of 100.
        """
        #error if time and system state have not been initialised
        if self.cur_state is None:
            raise RuntimeError("Initialisation has not occurred, use time_init() first")
        
        #error if new_flow is not a numpy array
        if not isinstance(new_flow, np.ndarray):
            raise TypeError("Flow input must be a NumPy array.")

        #ensure new flow array is 2D
        if new_flow.ndim != 2:
            raise ValueError("Flow array must be 2D.")
        
        #ensure both dimensions are equal for new flow array
        if new_flow.shape[0] != new_flow.shape[1]:
            raise ValueError("Flow array must be a square matrix (rows = columns).")
        
        #ensure both arrays have the same shape
        if new_flow.shape != self.dist.shape:
            raise ValueError("Flow array must have the same dimensions as existing distance array.")

        #initialise penalty if not given
        if penalty is None:
            penalty = self.row_col_penalty

        #set flow object to the new flow matrix
        self.flow = new_flow
        self.generate_qubo(penalty)

        #set new previous location to current location
        new_prev = self.cur_state
        self.prev_state = new_prev

        #add transition penalties
        previous_locations = np.argmax(self.prev_state, axis=1) #get the index of the 1 in each row
        move_penalty = 10
        N = self.size
        for i in range(N):
            fac_prev_state = previous_locations[i]  #previous location of facility i
            for m in range(N):
                move_penalty = move_penalty * self.dist[fac_prev_state, m]
                self.qubo[i * N + m, i * N + m] += move_penalty

        #generate new locations by sampling
        resp = self.sample_qap(shots = 5000).first.sample
        resp_arr = np.array(list(resp.values())).reshape((self.size, self.size))
        self.cur_state = resp_arr
        self.time += 1
        return resp_arr

    def sample_qap(self, sampler = None, shots: int = 100, penalty = None):
        """
        Optimize using current qubo object.
        Optional parameters for the sampler, number of reads, and QAP penalty.
        """
        #initialise sampler if not given
        if sampler is None:
            sampler = self.default_sampler

        #initialise penalty if not given
        if penalty is None:
            penalty = self.row_col_penalty
        
        #sample and return
        response = sampler.sample_qubo(self.qubo, num_reads = shots)
        return response
    
    def show_state_graph(self):
        """
        Convert matrix representation to graph representation and display
        """
        #initialise variables, raise errors if necessary
        graph = nx.DiGraph()
        flow = self.flow
        distance = self.dist
        boolean = self.cur_state
        if boolean is None:
            raise RuntimeError("Current state has not been initialised. Use time_init()")

        #find node locations
        locations = np.zeros(len(boolean), dtype=int)
        for row in range(len(boolean)):
            for column in range(len(boolean[row])):
                if boolean[row, column] == 1:
                    locations[row] = column
        
        #define nodes
        nodes = {
            0: "Loc. 1, Dept. " + str(locations[0] + 1),
            1: "Loc. 1, Dept. " + str(locations[1] + 1),
            2: "Loc. 3, Dept. " + str(locations[2] + 1),
        }

        #add nodes
        graph.add_nodes_from(nodes.values())

        #add weighted edges
        edges = [
            (nodes[0], nodes[1], distance[0, 1], "Distance"),
            (nodes[0], nodes[1], flow[0, 1], "Flow"),


            (nodes[0], nodes[2], distance[0, 2], "Distance"),
            (nodes[0], nodes[2], flow[0, 2], "Flow"),


            (nodes[1], nodes[2], distance[1, 2], "Distance"),
            (nodes[1], nodes[2], flow[1, 2], "Flow")
            ]
        
        for u, v, weight, label in edges:
            graph.add_edge(u, v, weight=weight, label=label)
        
        #make layout
        pos = nx.spring_layout(graph, k=1.5)

        #make layout
        pos = nx.spring_layout(graph, k=1.5)
        offset = 0.1 
        pos_offset = {k: (v[0] + offset, v[1] + offset) for k, v in pos.items()}  # Shift node labels

        #create a figure before drawing
        plt.figure(figsize=(6, 6)) 

        #initial drawing
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="black", node_size=1000, font_size=8, arrows=False)

        # prepare edge labels
        edge_labels_distance = {(u, v): f"{weight} (Distance)" for u, v, weight, label in edges if label == "Distance"}
        edge_labels_flow = {(u, v): f"{weight} (Flow)" for u, v, weight, label in edges if label == "Flow"}

        #draw edge labels
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels_distance, font_size=8, label_pos=0.3, bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"))  # Distance Label
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels_flow, font_size=8, label_pos=0.7, bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"))  # Flow Label

        #display graph
        plt.show()