import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

class time_QAP:
    def __init__(self, flow: np.ndarray, dist: np.ndarray, given_sampler = None):
        """
        Initialise the QAP class at time t = 0.
        Sampler, if given, must be able to handle QUBO instances
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

        #define qubo, which is first generated with default penalty
        self.qubo = None
        self.generate_qubo()

        #define default sampler, set to a simulator if no sampler specified
        if given_sampler is None:
            self.default_sampler = SimulatedAnnealingSampler()
        else:
            self.default_sampler = given_sampler
    
    def generate_qubo(self, penalty: int = 10000):
        """
        Create a QUBO matrix from the flow and distance matrices of the class.
        Does NOT account for transition penalties or facility-type penalties.
        """
        #tensor product (specifically Kronecker product) of flow and distance matrices
        Q = np.kron(self.flow, self.dist)

        #define row and column constraints to ensure no facility is assigned
        #more than one location, and no location has more than one facility
        constraint_groups = []
        N = self.size
        for i in range(N):
            constraint_groups.append([i * N + m for m in range(N)])
        for m in range(N):
            constraint_groups.append([m + N * i for i in range(N)])

        #apply the penalty in the correct form
        for group in constraint_groups:
            for i in range(len(group)):
                for j in range(i, len(group)):  # Upper triangular terms
                    var_i, var_j = group[i], group[j]
                    if i == j:
                        Q[var_i, var_j] -= penalty # Linear penalty term
                    else:
                        Q[var_i, var_j] += penalty  # Quadratic interaction term
                        Q[var_j, var_i] += penalty  # Ensure symmetry
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
    
    def time_evolve(self, new_flow: np.ndarray, penalty: int = 100000):
        """
        Evolve system according to new flow matrix. Default time step of 1, default qubo penalty of 100.
        """
        #error if time and system state have not been initialised
        if self.cur_state is None:
            raise RuntimeError("Initialisation has not occurred, use time_init() first")

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

    def sample_qap(self, sampler = None, shots: int = 100, penalty: int = 100000):
        """
        Optimize using current qubo object.
        Optional parameters for the sampler, number of reads, and QAP penalty.
        """
        if sampler is None:
            sampler = self.default_sampler
        response = sampler.sample_qubo(self.qubo, num_reads = shots)
        return response