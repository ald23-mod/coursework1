"""M3C 2018 Homework 1
Anas Lasri Doukkali, CID:01209387
"""
import numpy as np
import matplotlib.pyplot as plt


def simulate1(N,Nt,b,e):
    """Simulate C vs. M competition on N x N grid over
    Nt generations. b and e are model parameters
    to be used in fitness calculations
    Output: S: Status of each gridpoint at end of simulation, 0=M, 1=C
    fc: fraction of villages which are C at all Nt+1 times
    Do not modify input or return statement without instructor's permission.
    """
    #Set initial condition
    S  = np.ones((N,N),dtype=int) #Status of each gridpoint: 0=M, 1=C
    j = int((N-1)/2)
    S[j-1:j+2,j-1:j+2] = 0

    fc = np.zeros(Nt+1) #Fraction of points which are C
    fc[0] = S.sum()/(N*N)
    score_M = np.zeros((N,N),dtype=float);
    score_C = np.zeros((N,N),dtype=float);
    Number_Neighbours = np.zeros((N,N),dtype=int);
    for a in range(0,N):
        for b in range(0,N):
            if S[a,b] == 0:
                if a == 0 and b == 0:
                    Number_Neighbours[a,b] = 3;
                    for c in range(a,a+2):
                        for d in range(b,b+2):
                            if S[c,d] == 1:
                                score_M[a,b] = score_M[a,b] + b
                            else:
                                score_M[a,b] = score_M[a,b] + e

                elif a == N-1 and b == N-1:
                    Number_Neighbours[a,b] = 3;
                    for c in range(a-1,a+1):
                        for d in range(b-1,b+1):
                            if S[c,d] == 1:
                                score_M[a,b] = score_M[a,b] + b
                            else:
                                score_M[a,b] = score_M[a,b] + e
                elif a == 0 and b == N-1:
                    Number_Neighbours[a,b] = 3;
                    for c in range(a,a+2):
                        for d in range(b-1,b+1):
                            if S[c,d] == 1:
                                score_M[a,b] = score_M[a,b] + b
                            else:
                                score_M[a,b] = score_M[a,b] + e
                elif a == N-1 and b == 0:
                    Number_Neighbours[a,b] = 3;
                    for c in range(a-1,a+1):
                        for d in range(b,b+2):
                            if S[c,d] == 1:
                                score_M[a,b] = score_M[a,b] + b
                            else:
                                score_M[a,b] = score_M[a,b] + e
                elif 0 < a < N-1  and b == 0:
                    Number_Neighbours[a,b] = 5;
                    for c in range(a,a+2):
                        for d in range(b-1,b+2):
                            if S[c,d] == 1:
                                score_M[a,b] = score_M[a,b] + b
                            else:
                                score_M[a,b] = score_M[a,b] + e
                elif 0 < a < N-1  and b == N-1:
                    Number_Neighbours[a,b] = 5;
                    for c in range(a,a+2):
                        for d in range(b-1,b+1):
                            if S[c,d] == 1:
                                score_M[a,b] = score_M[a,b] + b
                            else:
                                score_M[a,b] = score_M[a,b] + e
                elif a == 0 and 0 < b < N-1:
                    Number_Neighbours[a,b] = 5;
                    for c in range(a-1,a+2):
                        for d in range(b,b+2):
                            if S[c,d] == 1:
                                score_M[a,b] = score_M[a,b] + b
                            else:
                                score_M[a,b] = score_M[a,b] + e
                elif a == N-1 and 0 < b < N-1:
                    Number_Neighbours[a,b] = 5;
                    for c in range(a-1,a+1):
                        for d in range(b-1,b+2):
                            if S[c,d] == 1:
                                score_M[a,b] = score_M[a,b] + b
                            else:
                                score_M[a,b] = score_M[a,b] + e
                else:
                    Number_Neighbours[a,b] = 5;
                    for c in range(a-1,a+2):
                        for d in range(b-1,b+2):
                            if S[c,d] == 1:
                                score_M[a,b] = score_M[a,b] + b
                            else:
                                score_M[a,b] = score_M[a,b] + e
            elif S[a,b] == 1:
                if a == 0 and b == 0:
                    Number_Neighbours[a,b] = 3;
                    for c in range(a,a+2):
                        for d in range(b,b+2):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1
                elif a == 0 and b == N-1:
                    Number_Neighbours[a,b] = 3;
                    for c in range(a,a+2):
                        for d in range(b-1,b+1):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1

                elif a == N-1 and b == N-1:
                    Number_Neighbours[a,b] = 3;
                    for c in range(a-1,a+1):
                        for d in range(b-1,b+1):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1
                elif a == N-1 and b == N-1:
                    Number_Neighbours[a,b] = 3;
                    for c in range(a,a+2):
                        for d in range(b-1,b+1):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1
                elif a == N-1 and b == 0:
                    Number_Neighbours[a,b] = 3;
                    for c in range(a-1,a+1):
                        for d in range(b,b+2):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1
                elif 0 < a < N-1  and b == 0:
                    Number_Neighbours[a,b] = 5;
                    for c in range(a,a+2):
                        for d in range(b-1,b+2):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1
                elif 0 < a < N-1  and b == N-1:
                    Number_Neighbours[a,b] = 5;
                    for c in range(a,a+2):
                        for d in range(b-1,b+1):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1
                elif a == 0 and 0 < b < N-1:
                    Number_Neighbours[a,b] = 5;
                    for c in range(a,a+2):
                        for d in range(b-1,b+2):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1
                elif a == N-1 and 0 < b < N-1:
                    Number_Neighbours[a,b] = 5;
                    for c in range(a-1,a+1):
                        for d in range(b-1,b+2):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1
                elif 0 < a < N and 0 < b < N:
                    Number_Neighbours[a,b] = 8;
                    for c in range(a-1,a+2):
                        for d in range(b-1,b+2):
                            if S[c,d] == 1:
                                score_C[a,b] = score_C[a,b] + 1
                else:
                    print(f"fucked up at ({a},{b})")




    return S,fc,score_C,score_M

def plot_S(S):
    """Simple function to create plot from input S matrix
    """
    ind_s0 = np.where(S==0) #C locations
    ind_s1 = np.where(S==1) #M locations
    plt.plot(ind_s0[1],ind_s0[0],'rs')
    plt.hold(True)
    plt.plot(ind_s1[1],ind_s1[0],'bs')
    plt.hold(False)
    plt.show()
    plt.pause(0.05)
    return None


def simulate2(N,Nt,b,e):
    """Simulation code for Part 2, add input variables as needed
    """


def analyze():
    """ Add input variables as needed
    """



if __name__ == '__main__':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code
    output = analyze()
