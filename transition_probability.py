"""Function to calculate the transition probabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def transition_probability(fitness_C_comm, fitness_M_comm):

    transition_probability = np.zeros((N,N),dtype=float);
    total_fitness = fitness_C_comm + fitness_M_comm;
    
    for a in range(0,N):
        for b in range(0,N):
            if S[a,b] == 1:
                transition_probability[a,b] = fitness_C_comm[a,b]/total_fitness[a,b]
            else:
                transition_probability[a,b] = fitness_M_comm[a,b]/total_fitness[a,b]
