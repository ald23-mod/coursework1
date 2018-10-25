""" This function will calculate the total fitness
of M villages in the community
"""
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def fitness_M_total(fitness_village, N):
    fitness_M_comm = np.zeros((N,N),dtype=float);
    for a in range(0,N):
        for b in range(0,N):
            if S[a,b] == 0:
                if a == 0 and b == 0:
                    for c in range(a,a+2):
                        for d in range(b,b+2):
                            if S[c,d] == 0:
                                fitness_M_comm[a,b] = fitness_M_comm[a,b] + score[c,d]
                elif a == N-1 and b == N-1:
                    for c in range(a-1,a+1):
                        for d in range(b-1,b+1):
                            if S[c,d] == 0:
                                fitness_M_comm[a,b] = fitness_M_comm[a,b] + score[c,d]
                elif a == 0 and b == N-1:
                    for c in range(a,a+2):
                        for d in range(b-1,b+1):
                            if S[c,d] == 0:
                                fitness_M_comm[a,b] = fitness_M_comm[a,b] + score[c,d]
                elif a == N-1 and b == 0:
                    for c in range(a-1,a+1):
                        for d in range(b,b+2):
                            if S[c,d] == 0:
                                fitness_M_comm[a,b] = fitness_M_comm[a,b] + score[c,d]
                elif 0 < a < N-1  and b == 0:
                    for c in range(a,a+2):
                        for d in range(b-1,b+2):
                            if S[c,d] == 0:
                                fitness_M_comm[a,b] = fitness_M_comm[a,b] + score[c,d]
                elif 0 < a < N-1  and b == N-1:
                    for c in range(a,a+2):
                        for d in range(b-1,b+1):
                            if S[c,d] == 0:
                                fitness_M_comm[a,b] = fitness_M_comm[a,b] + score[c,d]
                elif a == 0 and 0 < b < N-1:
                    for c in range(a-1,a+2):
                        for d in range(b,b+2):
                            if S[c,d] == 0:
                                fitness_M_comm[a,b] = fitness_M_comm[a,b] + score[c,d]
                elif a == N-1 and 0 < b < N-1:
                    for c in range(a-1,a+1):
                        for d in range(b-1,b+2):
                            if S[c,d] == 0:
                                fitness_M_comm[a,b] = fitness_M_comm[a,b] + score[c,d]
                else:
                    for c in range(a-1,a+2):
                        for d in range(b-1,b+2):
                            if S[c,d] == 0:
                                fitness_M_comm[a,b] = fitness_M_comm[a,b] + score[c,d]
