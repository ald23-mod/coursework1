"""Function to calculate the transition probabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def transition_probability(fitness_C_comm, fitness_M_comm, S):

    transition_probability_C = np.zeros((N,N),dtype=float);
    transition_probability_M = np.zeros((N,N),dtype=float);
    total_fitness = fitness_C_comm + fitness_M_comm;
    probability_criteria = np.random.rand(N,N);
    for a in range(0,N):
        for b in range(0,N):
            transition_probability_M[a,b] = fitness_C_comm[a,b]/total_fitness[a,b]
            transition_probability_C[a,b] = fitness_M_comm[a,b]/total_fitness[a,b]
            if transition_probability_M < probability_criteria[a,b]:
                S[a,b] = 1
            else:
                S[a,b] = 0

return S
