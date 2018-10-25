
"""M3C 2018 Homework 1

Anas Lasri Doukkali, CID:01209387
"""
import numpy as np
import matplotlib.pyplot as plt


def simulate1(N, Nt, b, e):
    """Simulate C vs. M competition on N x N grid over.

    Nt generations. b and e are model parameters
    to be used in fitness calculations
    Output: S: Status of each gridpoint at end of simulation, 0=M, 1=C
    fc: fraction of villages which are C at all Nt+1 times
    Do not modify input or return statement without instructor's permission.
    """

    # Set initial condition
    S = np.ones((N, N), dtype=int)  # Status of each gridpoint: 0=M, 1=C
    j = int((N-1)/2)
    S[j-1:j+2, j-1:j+2] = 0
    fc = np.zeros(Nt+1)  # Fraction of points which are C
    fc[0] = S.sum()/(N*N)
    score = np.zeros((N, N), dtype=float)
    Number_Neighbours = np.zeros((N, N), dtype=int)
    fitness_M_comm = np.zeros((N,N),dtype=float)
    fitness_C_comm = np.zeros((N,N),dtype=float)
    transition_probability_C = np.zeros((N,N),dtype=float)
    transition_probability_M = np.zeros((N,N),dtype=object)
    fitness_village = np.zeros((N,N),dtype=object)
    probability_criteria = np.random.rand(N,N)

    for t in range(1, Nt+1):
        for a in range(0, N):
            for b in range(0, N):
                if S[a, b] == 0:
                    if a == 0 and b == 0:
                        Number_Neighbours[a, b] = 3
                        for c in range(a, a+2):
                            for d in range(b, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + b
                                else:
                                    score[a, b] = score[a, b] + e
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == N-1 and b == N-1:
                        Number_Neighbours[a, b] = 3
                        for c in range(a-1, a+1):
                            for d in range(b-1, b+1):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + b
                                else:
                                    score[a, b] = score[a, b] + e
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == 0 and b == N-1:
                        Number_Neighbours[a, b] = 3
                        for c in range(a, a+2):
                            for d in range(b-1, b+1):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + b
                                else:
                                    score[a, b] = score[a, b] + e
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == N-1 and b == 0:
                        Number_Neighbours[a, b] = 3
                        for c in range(a-1, a+1):
                            for d in range(b, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + b
                                else:
                                    score[a, b] = score[a, b] + e
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif 0 < a < N-1 and b == 0:
                        Number_Neighbours[a, b] = 5
                        for c in range(a, a+2):
                            for d in range(b-1, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + b
                                else:
                                    score[a, b] = score[a, b] + e
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif 0 < a < N-1 and b == N-1:
                        Number_Neighbours[a, b] = 5
                        for c in range(a, a+2):
                            for d in range(b-1, b+1):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + b
                                else:
                                    score[a, b] = score[a, b] + e
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == 0 and 0 < b < N-1:
                        Number_Neighbours[a, b] = 5
                        for c in range(a-1, a+2):
                            for d in range(b, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + b
                                else:
                                    score[a, b] = score[a, b] + e
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == N-1 and 0 < b < N-1:
                        Number_Neighbours[a, b] = 5
                        for c in range(a-1, a+1):
                            for d in range(b-1, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + b
                                else:
                                    score[a, b] = score[a, b] + e
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    else:
                        Number_Neighbours[a, b] = 5
                        for c in range(a-1, a+2):
                            for d in range(b-1, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + b
                                else:
                                    score[a, b] = score[a, b] + e
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                elif S[a, b] == 1:
                    if a == 0 and b == 0:
                        Number_Neighbours[a, b] = 3
                        for c in range(a, a+2):
                            for d in range(b, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == 0 and b == N-1:
                        Number_Neighbours[a, b] = 3
                        for c in range(a, a+2):
                            for d in range(b-1, b+1):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == N-1 and b == N-1:
                        Number_Neighbours[a, b] = 3
                        for c in range(a-1, a+1):
                            for d in range(b-1, b+1):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == N-1 and b == N-1:
                        Number_Neighbours[a, b] = 3
                        for c in range(a, a+2):
                            for d in range(b-1, b+1):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == N-1 and b == 0:
                        Number_Neighbours[a, b] = 3
                        for c in range(a-1, a+1):
                            for d in range(b, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif 0 < a < N-1 and b == 0:
                        Number_Neighbours[a, b] = 5
                        for c in range(a, a+2):
                            for d in range(b-1, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif 0 < a < N-1 and b == N-1:
                        Number_Neighbours[a, b] = 5
                        for c in range(a, a+2):
                            for d in range(b-1, b+1):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == 0 and 0 < b < N-1:
                        Number_Neighbours[a, b] = 5
                        for c in range(a, a+2):
                            for d in range(b-1, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif a == N-1 and 0 < b < N-1:
                        Number_Neighbours[a, b] = 5
                        for c in range(a-1, a+1):
                            for d in range(b-1, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]
                    elif 0 < a < N and 0 < b < N:
                        Number_Neighbours[a, b] = 8
                        for c in range(a-1, a+2):
                            for d in range(b-1, b+2):
                                if S[c, d] == 1:
                                    score[a, b] = score[a, b] + 1
                        fitness_village[a, b] = score[a,b]/Number_Neighbours[a,b]


    """    for a in range(0,N):
            for b in range(0,N):
                transition_probability_M[a,b] = fitness_C_comm[a,b]/total_fitness[a,b]
                transition_probability_C[a,b] = fitness_M_comm[a,b]/total_fitness[a,b]
                if transition_probability_M[a,b] < probability_criteria[a,b]:
                    S[a,b] = 1
                else:
                    S[a,b] = 0
        for j in range(1,Nt):
            fc[j] = S.sum()/(N*N)
"""
    return S, fc, fitness_village


def plot_S(S):
    """Simple function to create plot from input S matrix"""

    ind_s0 = np.where(S == 0)  # C locations
    ind_s1 = np.where(S == 1)  # M locations
    plt.plot(ind_s0[1], ind_s0[0], 'rs')
    plt.hold(True)
    plt.plot(ind_s1[1], ind_s1[0], 'bs')
    plt.hold(False)
    plt.show()
    plt.pause(0.05)
    return None


def simulate2(N, Nt, b, e):
    """Simulation code for Part 2, add input variables as needed
    """


def analyze():
    """ Add input variables as needed
    """

if __name__ == '__main__':


    # The code here should call analyze and generate the
    # figures that you are submitting with your code
    output = analyze()
