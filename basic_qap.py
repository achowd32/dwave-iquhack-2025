import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

class QAP:
    def __init__(self, flow: np.ndarray, dist: np.ndarray):
        """Initialise the QAP class"""
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
    
    def generate_qubo(self, penalty: int = 100):
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

        return Q
    
    def sample_optimized(self, sampler = None, shots: int = 100, penalty: int = 100):
        """Optimize using QUBO generated from the generate_qubo() method.
        Optional parameters for the sampler, number of reads, and QAP penalty."""
        if sampler is None:
            sampler = self.default_sampler
        Q = self.generate_qubo(penalty)
        response = sampler.sample_qubo(Q, num_reads = shots)
        return response

# Flow matrix (f)
# f = np.array([[0, 5, 2], 5, 0, 3], [2, 3, 0]])

# Distance matrix (d)
# d = np.array([[0, 8, 15], [8, 0, 13], [15,13, 0]])