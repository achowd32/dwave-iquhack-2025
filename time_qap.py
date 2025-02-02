import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

class time_QAP:
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

qc_sampler = EmbeddingComposite(DWaveSampler())

flow = np.array([[0, 3, 2], [3, 0, 4], [2, 4, 0]])
dist = np.array([[0, 5, 1], [5, 0, 3], [1, 3, 0]])
test_3 = time_QAP(flow, dist, 2, qc_sampler)
print(test_3.time_init())