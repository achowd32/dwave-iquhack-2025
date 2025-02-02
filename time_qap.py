import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

class time_QAP:
    def __init__(self, flow: np.ndarray, dist: np.ndarray):
        """Initialise the QAP class at time t = 0"""
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
        
        #define class objects
        self.flow = flow
        self.dist = dist
        self.size = flow.shape[0]
        self.default_sampler = SimulatedAnnealingSampler()
        self.time = 0
        self.prev_loc = None
        self.cur_loc = None
        self.qubo = None
        self.generate_qubo()
    
    def generate_qubo(self, penalty: int = 400):
        """Create a QUBO matrix from the flow and distance matrices of the class.
        Default penalty of 100."""
        #tensor product of flow and distance matrices
        Q = np.kron(self.flow, self.dist)

        # Define row and column constraints
        constraint_groups = []
        N = self.size
        for i in range(N):
            constraint_groups.append([i * N + m for m in range(N)])
        for m in range(N):
            constraint_groups.append([m + N * i for i in range(N)])

        # Apply the penalty in the correct form
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
        """Initialises locations, allows for time evolution"""
        if self.cur_loc is not None:
            raise ValueError("Initialisation has already occurred, use time_evolve() instead") #fix this
        resp = self.sample_qap(shots = 1000).first.sample
        resp_arr = np.array(list(resp.values())).reshape((self.size, self.size))
        self.cur_loc = resp_arr
        return resp_arr
    
    def time_evolve(self, new_flow: np.ndarray, penalty: int = 400):
        """Evolve system according to new flow matrix. Default time step of 1, default qubo penalty of 100."""
        if self.cur_loc is None:
            raise ValueError("Initialisation has not occurred, use time_init() first")
        self.flow = new_flow
        self.generate_qubo(penalty)

        #set new previous location to current location
        new_prev = self.cur_loc
        self.prev_loc = new_prev

        #add transition penalties
        previous_locations = np.argmax(self.prev_loc, axis=1) #Get the index of the 1 in each row
        move_penalty = 10
        N = self.size
        for i in range(N):
            fac_prev_loc = previous_locations[i]  # Previous location of facility i
            for m in range(N):
                move_penalty = move_penalty * d[fac_prev_loc, m]
                self.qubo[i * N + m, i * N + m] += move_penalty

        #generate new locations by sampling
        resp = self.sample_qap(shots = 1000).first.sample
        resp_arr = np.array(list(resp.values())).reshape((self.size, self.size))
        self.cur_loc = resp_arr
        return resp_arr

    def sample_qap(self, sampler = None, shots: int = 100, penalty: int = 100):
        """Optimize using QUBO generated from the generate_qubo() method.
        Optional parameters for the sampler, number of reads, and QAP penalty."""
        if sampler is None:
            sampler = self.default_sampler
        response = sampler.sample_qubo(self.qubo, num_reads = shots)
        return response

# Flow matrix (f)
f = np.array([[0, 5, 2], [5, 0, 30], [2, 30, 0]])

# Distance matrix (d)
d = np.array([[0, 105, 15],
        [8, 0, 13],
        [15,13, 0]])
d2 = np.array([[0, 20, 3],
            [20, 0, 7],
            [3, 7, 0]])

test_qap = time_QAP(f, d)
print(test_qap.time_init())
print(test_qap.time_evolve(d2))