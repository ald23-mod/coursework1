import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

class gridshape:
#Set initial condition
S  = np.ones((N,N),dtype=int) #Status of each gridpoint: 0=M, 1=C
j = int((N-1)/2)
S[j-1:j+2,j-1:j+2] = 0
fc = np.zeros(Nt+1) #Fraction of points which are C
fc[0] = S.sum()/(N*N)
score = np.zeros((N,N),dtype=float);
Number_Neighbours = np.zeros((N,N),dtype=int);

    def __init__(self, N, Nt, b, e):
        self.N = N
        self.Nt = Nt
        self.b = b
        self.e = e

    def simulation_transition(self,N,Nt,b,e):
        probability_criteria = np.random.rand(N,N);
        for a in range(0,N):
            for b in range(0,N):
                if S[a,b] == 0:
                    if transition_probability_C < probability_criteria[a,b]:
                        S[a,b] = 1
                    else:
                        S[a,b] = 0
                else:
                    if transition_probability_M < probability_criteria[a,b]:
                        S[a,b] = 0
                    else:
                        S[a,b] = 1
        return S 

    def village_fitness(self):
        for t in range (1,Nt+1):
            for a in range(0,N):
                for b in range(0,N):
                    if S[a,b] == 0:
                        if a == 0 and b == 0:
                            Number_Neighbours[a,b] = 3;
                            for c in range(a,a+2):
                                for d in range(b,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + b
                                    else:
                                        score[a,b] = score[a,b] + e
                        elif a == N-1 and b == N-1:
                            Number_Neighbours[a,b] = 3;
                            for c in range(a-1,a+1):
                                for d in range(b-1,b+1):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + b
                                    else:
                                        score[a,b] = score[a,b] + e
                        elif a == 0 and b == N-1:
                            Number_Neighbours[a,b] = 3;
                            for c in range(a,a+2):
                                for d in range(b-1,b+1):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + b
                                    else:
                                        score[a,b] = score[a,b] + e
                        elif a == N-1 and b == 0:
                            Number_Neighbours[a,b] = 3;
                            for c in range(a-1,a+1):
                                for d in range(b,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + b
                                    else:
                                        score[a,b] = score[a,b] + e
                        elif 0 < a < N-1  and b == 0:
                            Number_Neighbours[a,b] = 5;
                            for c in range(a,a+2):
                                for d in range(b-1,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + b
                                    else:
                                        score[a,b] = score[a,b] + e
                        elif 0 < a < N-1  and b == N-1:
                            Number_Neighbours[a,b] = 5;
                            for c in range(a,a+2):
                                for d in range(b-1,b+1):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + b
                                    else:
                                        score[a,b] = score[a,b] + e
                        elif a == 0 and 0 < b < N-1:
                            Number_Neighbours[a,b] = 5;
                            for c in range(a-1,a+2):
                                for d in range(b,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + b
                                    else:
                                        score[a,b] = score[a,b] + e
                        elif a == N-1 and 0 < b < N-1:
                            Number_Neighbours[a,b] = 5;
                            for c in range(a-1,a+1):
                                for d in range(b-1,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + b
                                    else:
                                        score[a,b] = score[a,b] + e
                        else:
                            Number_Neighbours[a,b] = 5;
                            for c in range(a-1,a+2):
                                for d in range(b-1,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + b
                                    else:
                                        score[a,b] = score[a,b] + e
                    elif S[a,b] == 1:
                        if a == 0 and b == 0:
                            Number_Neighbours[a,b] = 3;
                            for c in range(a,a+2):
                                for d in range(b,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1
                        elif a == 0 and b == N-1:
                            Number_Neighbours[a,b] = 3;
                            for c in range(a,a+2):
                                for d in range(b-1,b+1):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1

                        elif a == N-1 and b == N-1:
                            Number_Neighbours[a,b] = 3;
                            for c in range(a-1,a+1):
                                for d in range(b-1,b+1):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1
                        elif a == N-1 and b == N-1:
                            Number_Neighbours[a,b] = 3;
                            for c in range(a,a+2):
                                for d in range(b-1,b+1):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1
                        elif a == N-1 and b == 0:
                            Number_Neighbours[a,b] = 3;
                            for c in range(a-1,a+1):
                                for d in range(b,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1
                        elif 0 < a < N-1  and b == 0:
                            Number_Neighbours[a,b] = 5;
                            for c in range(a,a+2):
                                for d in range(b-1,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1
                        elif 0 < a < N-1  and b == N-1:
                            Number_Neighbours[a,b] = 5;
                            for c in range(a,a+2):
                                for d in range(b-1,b+1):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1
                        elif a == 0 and 0 < b < N-1:
                            Number_Neighbours[a,b] = 5;
                            for c in range(a,a+2):
                                for d in range(b-1,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1
                        elif a == N-1 and 0 < b < N-1:
                            Number_Neighbours[a,b] = 5;
                            for c in range(a-1,a+1):
                                for d in range(b-1,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1
                        elif 0 < a < N and 0 < b < N:
                            Number_Neighbours[a,b] = 8;
                            for c in range(a-1,a+2):
                                for d in range(b-1,b+2):
                                    if S[c,d] == 1:
                                        score[a,b] = score[a,b] + 1
                        else:
                            print(f"fucked up at ({a},{b})")
            for i in range(0,N):
                for j in range(0,N):
                    if S[i,j] == 1:
                        score[i,j] = score[i,j] - 1;
                    else:
                        score[i,j] = score[i,j] - e;
            fitness_village = np.divide(score, Number_Neighbours)
        return fitness_village

    def fitness_C_comm(self):
        fitness_C_comm = np.zeros((N,N),dtype=float);
        for a in range(0,N):
            for b in range(0,N):
                if S[a,b] == 1:
                    if a == 0 and b == 0:
                        for c in range(a,a+2):
                            for d in range(b,b+2):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    elif a == 0 and b == N-1:
                        for c in range(a,a+2):
                            for d in range(b-1,b+1):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    elif a == N-1 and b == N-1:
                        for c in range(a-1,a+1):
                            for d in range(b-1,b+1):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    elif a == N-1 and b == N-1:
                        for c in range(a,a+2):
                            for d in range(b-1,b+1):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    elif a == N-1 and b == 0:
                        for c in range(a-1,a+1):
                            for d in range(b,b+2):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    elif 0 < a < N-1  and b == 0:
                        for c in range(a,a+2):
                            for d in range(b-1,b+2):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    elif 0 < a < N-1  and b == N-1:
                        for c in range(a,a+2):
                            for d in range(b-1,b+1):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    elif a == 0 and 0 < b < N-1:
                        for c in range(a,a+2):
                            for d in range(b-1,b+2):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    elif a == N-1 and 0 < b < N-1:
                        for c in range(a-1,a+1):
                            for d in range(b-1,b+2):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    elif 0 < a < N and 0 < b < N:
                        for c in range(a-1,a+2):
                            for d in range(b-1,b+2):
                                if S[c,d] == 1:
                                    fitness_C_comm[a,b] = fitness_C_comm[a,b] + fitness_village[c,d]
                    else:
                        print(f"fucked up at ({a},{b})")

        return fitness_C_comm


    def fitness_M_comm(self):
        fitness_M_comm = np.zeros((N,N),dtype=float);
        for a in range(0,N):
            for b in range(0,N):
                if S[a,b] == 0:
                    if a == 0 and b == 0:
                        for c in range(a,a+2):
                            for d in range(b,b+2):
                                if S[c,d] == 0:
                                    fitness_M_comm[a,b] = fitness_M_comm[a,b] + fitness_village[c,d]
                    elif a == N-1 and b == N-1:
                        for c in range(a-1,a+1):
                            for d in range(b-1,b+1):
                                if S[c,d] == 0:
                                    fitness_M_comm[a,b] = fitness_M_comm[a,b] + fitness_village[c,d]
                    elif a == 0 and b == N-1:
                        for c in range(a,a+2):
                            for d in range(b-1,b+1):
                                if S[c,d] == 0:
                                    fitness_M_comm[a,b] = fitness_M_comm[a,b] + fitness_village[c,d]
                    elif a == N-1 and b == 0:
                        for c in range(a-1,a+1):
                            for d in range(b,b+2):
                                if S[c,d] == 0:
                                    fitness_M_comm[a,b] = fitness_M_comm[a,b] + fitness_village[c,d]
                    elif 0 < a < N-1  and b == 0:
                        for c in range(a,a+2):
                            for d in range(b-1,b+2):
                                if S[c,d] == 0:
                                    fitness_M_comm[a,b] = fitness_M_comm[a,b] + fitness_village[c,d]
                    elif 0 < a < N-1  and b == N-1:
                        for c in range(a,a+2):
                            for d in range(b-1,b+1):
                                if S[c,d] == 0:
                                    fitness_M_comm[a,b] = fitness_M_comm[a,b] + fitness_village[c,d]
                    elif a == 0 and 0 < b < N-1:
                        for c in range(a-1,a+2):
                            for d in range(b,b+2):
                                if S[c,d] == 0:
                                    fitness_M_comm[a,b] = fitness_M_comm[a,b] + fitness_village[c,d]
                    elif a == N-1 and 0 < b < N-1:
                        for c in range(a-1,a+1):
                            for d in range(b-1,b+2):
                                if S[c,d] == 0:
                                    fitness_M_comm[a,b] = fitness_M_comm[a,b] + fitness_village[c,d]
                    else:
                        for c in range(a-1,a+2):
                            for d in range(b-1,b+2):
                                if S[c,d] == 0:
                                    fitness_M_comm[a,b] = fitness_M_comm[a,b] + fitness_village[c,d]
         return fitness_M_comm


    def total_fitness(self):
        total_fitness = fitness_C_comm + fitness_M_comm
        return total_fitness


    def transition_probability(self):
        for a in range(0,N):
            for b in range(0,N):
                transition_probability_M[a,b] = fitness_C_comm[a,b]/total_fitness[a,b]
                transition_probability_C[a,b] = fitness_M_comm[a,b]/total_fitness[a,b]
        return transition_probability_C, transition_probability_M
