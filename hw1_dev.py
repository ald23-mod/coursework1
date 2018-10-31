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
    A = 8*np.ones((N,N), dtype=object)
    B = np.pad(np.zeros((N-2,N-2), dtype=object), pad_width=1, mode='constant', constant_values=3)
    C = np.zeros((N,N))
    C[0,0] = 2
    C[0,C.shape[1]-1] = 2
    C[C.shape[0]-1,0] = 2
    C[C.shape[0]-1,C.shape[1]-1]=2
    Number_Neighbours = A - B - C
    Number_Neighbours = np.pad(Number_Neighbours, pad_width = 1, mode='constant', constant_values=2)
    S  = np.pad(S, pad_width=1, mode='constant', constant_values=2)
    for t in range(0,Nt):
        score = np.zeros((N+2,N+2), dtype=object)
        fitness_matrix = np.zeros((N+2,N+2), dtype=object)
        fitness_C_comm = np.zeros((N+2,N+2), dtype=object)
        fitness_M_comm = np.zeros((N+2,N+2), dtype=object)
        transitionprob_to_M = np.zeros((N+2,N+2), dtype=object)
        transitionprob_to_C = np.zeros((N+2,N+2), dtype=object)
        for i in range(1,N+1):
            for j in range(1,N+1):
                if S[i,j] == 1:
                    for c in range(i-1,i+2):
                        for d in range(j-1,j+2):
                            if S[c,d] == 1:
                                score[i,j] = score[i,j] + 1
                            else:
                                score[i,j] = score[i,j]
                    score[i,j] = score[i,j] - 1
                elif S[i,j] == 0:
                    for c in range(i-1,i+2):
                        for d in range(j-1,j+2):
                            if S[c,d] == 1:
                                score[i,j] = score[i,j] + b
                            elif S[c,d] == 0:
                                score[i,j] = score[i,j] + e
                            else:
                                score[i,j] = score[i,j]
                    score[i,j] = score[i,j] - e
                else:
                    score[i,j] = 0
        fitness_matrix = np.divide(score, Number_Neighbours)
        for i in range(1,N+1):
            for j in range(1,N+1):
                for c in range(i-1,i+2):
                    for d in range(j-1,j+2):
                        if S[c,d] == 0:
                            fitness_M_comm[i,j] = fitness_M_comm[i,j] + fitness_matrix[c,d]
                            fitness_C_comm[i,j] = fitness_C_comm[i,j]
                        elif S[c,d] == 1:
                            fitness_C_comm[i,j] = fitness_C_comm[i,j] + fitness_matrix[c,d]
                            fitness_M_comm[i,j] = fitness_M_comm[i,j]

                        else:
                            fitness_C_comm[i,j] = fitness_C_comm[i,j]
                            fitness_M_comm[i,j] = fitness_M_comm[i,j]
        for i in range(1,N+1):
            for j in range(1,N+1):
                if S[i,j] == 1:
                    transitionprob_to_M[i,j] = fitness_M_comm[i,j]/(fitness_C_comm[i,j] + fitness_M_comm[i,j])
                elif S[i,j] == 0:
                    transitionprob_to_C[i,j] = fitness_C_comm[i,j]/(fitness_M_comm[i,j] + fitness_C_comm[i,j])
        R = np.random.rand(N+2,N+2)
        for i in range(1,N+1):
            for j in range(1,N+1):
                if S[i,j] == 1:
                    if R[i,j] <= transitionprob_to_M[i,j]:
                        S[i,j] = 0
                    else:
                        S[i,j] = 1
                else:
                    if R[i,j] <= transitionprob_to_C[i,j]:
                        S[i,j] = 1
                    else:
                        S[i,j] = 0
        fc[t+1] = (S.sum()-8*N-8)/(N*N)
        """
        S = np.delete(S,0,1)
        S = np.delete(S,N,1)
        S = np.delete(S,0,0)
        S = np.delete(S,N,0)
        #fc[t] = S.sum()/(N*N)
        """

        plot_S(S)
    return S, fc, score, fitness_matrix



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


def analyze(display=False):
    """ Add input variables as needed
    """
    bvalues = [1.5, 5.0, 20, 100]
    e = 0.01
    N = 21
    Nt = [0,21]
    for b in enumerate(bvalues):
        _,fc = simulate1(N,Nt,b,e)
        print(fc)
    if display:
        plt.figure()
        plt.plot(fc[:],Nt,'x--')






if __name__ == '__main__':


    # The code here should call analyze and generate the
    # figures that you are submitting with your code
    output = analyze()
