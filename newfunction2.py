""" This function will calculate the total fitness
of C villages in the community
"""
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def fitness_C_total(fitness_village):
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
