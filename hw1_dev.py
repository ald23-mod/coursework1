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
    fc[0] = S.sum()/(N*N) # First proportion of C's in the entire community
    ###########################################################################
    """ In these next 10 lines of code I will generate the Number of Neighbours
     matrix to avoid using loops to calculate it and hence vectorize an important
     part of the excercise.This is done by generating three matrices and using
     simple matrix substraction to acheive the Number of Neighbours matrix for
     any given N.
    """
    A = 8*np.ones((N,N), dtype=object) # Matrix of 8's
    B = np.pad(np.zeros((N-2,N-2), dtype=object), pad_width=1, mode='constant', constant_values=3) # Matrix of 0's with 3's in the border
    C = np.zeros((N,N)) # Matrix of zeros, with 2's in all four corners
    C[0,0] = 2
    C[0,C.shape[1]-1] = 2
    C[C.shape[0]-1,0] = 2
    C[C.shape[0]-1,C.shape[1]-1]=2
    Number_Neighbours = A - B - C # This gives us the Number of Neighbours N*N Matrix
    ###########################################################################
    """ Again, to avoid excesive "if" statements, I create a border around the
     matrices with constant value "2". This way, the loop to calculate the score
     will be the same for all elements as all elements will have 8 "Neighbours"
     however as we have the Number of Neighbours matrix already set, the extra
     "2" neighbours do not affect our algorithm
    """
    S  = np.pad(S, pad_width=1, mode='constant', constant_values=2)
    Number_Neighbours = np.pad(Number_Neighbours, pad_width = 1, mode='constant', constant_values=2)
    ###########################################################################
    for t in range(0,Nt):
        # Initializing all the matrices that will be used
        score = np.zeros((N+2,N+2), dtype=object)
        fitness_matrix = np.zeros((N+2,N+2), dtype=object)
        fitness_C_comm = np.zeros((N+2,N+2), dtype=object)
        fitness_M_comm = np.zeros((N+2,N+2), dtype=object)
        transitionprob_to_M = np.zeros((N+2,N+2), dtype=object)
        transitionprob_to_C = np.zeros((N+2,N+2), dtype=object)

    ###########################################################################
        # Calculating the score of each village.
        for i in range(1,N+1):
            for j in range(1,N+1):
                if S[i,j] == 1:
                    for c in range(i-1,i+2):
                        for d in range(j-1,j+2):
                            if S[c,d] == 1:
                                score[i,j] = score[i,j] + 1
                            elif S[c,d] == 0:
                                score[i,j] = score[i,j]
                    score[i,j] = score[i,j] - 1 # As the element is a C we substract 1 that was added since the loop takes into account the element as well
                elif S[i,j] == 0:
                    for c in range(i-1,i+2):
                        for d in range(j-1,j+2):
                            if S[c,d] == 1:
                                score[i,j] = score[i,j] + b
                            elif S[c,d] == 0:
                                score[i,j] = score[i,j] + e
                    score[i,j] = score[i,j] - e # As the element is an M we substract e that was added since the loop takes into account the element as well

        # Use np.divide to divide element by element and avoid loops to
        # calculate the fitness of each village
        fitness_matrix = np.divide(score, Number_Neighbours)

        #######################################################################
        # Setting up the loops to calculate the fitness of the "Communities"
        for i in range(1,N+1):
            for j in range(1,N+1):
                for c in range(i-1,i+2):
                    for d in range(j-1,j+2):
                        if S[c,d] == 0:
                            fitness_M_comm[i,j] = fitness_M_comm[i,j] + fitness_matrix[c,d]
                        elif S[c,d] == 1:
                            fitness_C_comm[i,j] = fitness_C_comm[i,j] + fitness_matrix[c,d]

        #######################################################################
        # Setting up the loops to calculate the transition probabilities of each village
        for i in range(1,N+1):
            for j in range(1,N+1):
                if S[i,j] == 1:
                    transitionprob_to_M[i,j] = fitness_M_comm[i,j]/(fitness_C_comm[i,j] + fitness_M_comm[i,j])
                elif S[i,j] == 0:
                    transitionprob_to_C[i,j] = fitness_C_comm[i,j]/(fitness_M_comm[i,j] + fitness_C_comm[i,j])

        # Generating a random matrix with values between 0 and 1
        R = np.random.rand(N+1,N+1)
        #######################################################################
        # This is the part where the randomization happens. If the number
        # that was randomly generated is less than the transition probability
        # we then transition from our current state to the next one
        for i in range(1,N+1):
            for j in range(1,N+1):
                if S[i,j] == 1:
                    if R[i,j] <= transitionprob_to_M[i,j]:
                        S[i,j] = 0
                else:
                    if R[i,j] <= transitionprob_to_C[i,j]:
                        S[i,j] = 1
        # As my matrix has 2's in the border I have to introduce a correction
        # to the proportion of C's fc vector so that it calculates the correct
        # quantity
        fc[t+1] = (S.sum()-8*(N+1))/(N*N)
        # Can use the next line without '#' to plot the figures
        #plot_S(S)
    # Here I use slice to return S to its main form with no added borders
    slice_val = 1
    S = S[slice_val:-slice_val, slice_val:-slice_val]
    return S, fc

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

def simulate2(b , Nt, N=21, e=0.01):
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
    fc[0] = S.sum()/(N*N) # First proportion of C's in the entire community
    ###########################################################################
    """ In these next 10 lines of code I will generate the Number of Neighbours
     matrix to avoid using loops to calculate it and hence vectorize an important
     part of the excercise.This is done by generating three matrices and using
     simple matrix substraction to acheive the Number of Neighbours matrix for
     any given N.
    """
    A = 8*np.ones((N,N), dtype=object) # Matrix of 8's
    B = np.pad(np.zeros((N-2,N-2), dtype=object), pad_width=1, mode='constant', constant_values=3) # Matrix of 0's with 3's in the border
    C = np.zeros((N,N)) # Matrix of zeros, with 2's in all four corners
    C[0,0] = 2
    C[0,C.shape[1]-1] = 2
    C[C.shape[0]-1,0] = 2
    C[C.shape[0]-1,C.shape[1]-1]=2
    Number_Neighbours = A - B - C # This gives us the Number of Neighbours N*N Matrix
    ###########################################################################
    """ Again, to avoid excesive "if" statements, I create a border around the
     matrices with constant value "2". This way, the loop to calculate the score
     will be the same for all elements as all elements will have 8 "Neighbours"
     however as we have the Number of Neighbours matrix already set, the extra
     "2" neighbours do not affect our algorithm
    """
    S  = np.pad(S, pad_width=1, mode='constant', constant_values=2)
    Number_Neighbours = np.pad(Number_Neighbours, pad_width = 1, mode='constant', constant_values=2)
    ###########################################################################
    for t in range(0,Nt):
        # Initializing all the matrices that will be used
        score = np.zeros((N+2,N+2), dtype=object)
        fitness_matrix = np.zeros((N+2,N+2), dtype=object)
        fitness_C_comm = np.zeros((N+2,N+2), dtype=object)
        fitness_M_comm = np.zeros((N+2,N+2), dtype=object)
        transitionprob_to_M = np.zeros((N+2,N+2), dtype=object)
        transitionprob_to_C = np.zeros((N+2,N+2), dtype=object)

    ###########################################################################
        # Calculating the score of each village.
        for i in range(1,N+1):
            for j in range(1,N+1):
                if S[i,j] == 1:
                    for c in range(i-1,i+2):
                        for d in range(j-1,j+2):
                            if S[c,d] == 1:
                                score[i,j] = score[i,j] + 1
                            elif S[c,d] == 0:
                                score[i,j] = score[i,j]
                    score[i,j] = score[i,j] - 1 # As the element is a C we substract 1 that was added since the loop takes into account the element as well
                elif S[i,j] == 0:
                    for c in range(i-1,i+2):
                        for d in range(j-1,j+2):
                            if S[c,d] == 1:
                                score[i,j] = score[i,j] + b
                            elif S[c,d] == 0:
                                score[i,j] = score[i,j] + e
                    score[i,j] = score[i,j] - e # As the element is an M we substract e that was added since the loop takes into account the element as well

        # Use np.divide to divide element by element and avoid loops to
        # calculate the fitness of each village
        fitness_matrix = np.divide(score, Number_Neighbours)

    ###########################################################################
        # Setting up the loops to calculate the fitness of the "Communities"
        for i in range(1,N+1):
            for j in range(1,N+1):
                for c in range(i-1,i+2):
                    for d in range(j-1,j+2):
                        if S[c,d] == 0:
                            fitness_M_comm[i,j] = fitness_M_comm[i,j] + fitness_matrix[c,d]
                        elif S[c,d] == 1:
                            fitness_C_comm[i,j] = fitness_C_comm[i,j] + fitness_matrix[c,d]

    ###########################################################################
        # Setting up the loops to calculate the transition probabilities of each village
        for i in range(1,N+1):
            for j in range(1,N+1):
                if S[i,j] == 1:
                    transitionprob_to_M[i,j] = fitness_M_comm[i,j]/(fitness_C_comm[i,j] + fitness_M_comm[i,j])
                elif S[i,j] == 0:
                    transitionprob_to_C[i,j] = fitness_C_comm[i,j]/(fitness_M_comm[i,j] + fitness_C_comm[i,j])

        # Generating a random matrix with values between 0 and 1
        R = np.random.rand(N+1,N+1)
    ############################################################################
        # This is the part where the randomization happens. If the number
        # that was randomly generated is less than the transition probability
        # we then transition from our current state to the next one
        for i in range(1,N+1):
            for j in range(1,N+1):
                if S[i,j] == 1:
                    if R[i,j] <= transitionprob_to_M[i,j]:
                        S[i,j] = 0
                else:
                    if R[i,j] <= transitionprob_to_C[i,j]:
                        S[i,j] = 1
        # As my matrix has 2's in the border I have to introduce a correction
        # to the proportion of C's fc vector so that it calculates the correct
        # quantity
        fc[t+1] = (S.sum()-8*(N+1))/(N*N)
        # Can use the next line without '#' to plot the figures
        #plot_S(S)
    # Here I use slice to return S to its main form with no added borders
    slice_val = 1
    S = S[slice_val:-slice_val, slice_val:-slice_val]
    return S, fc
def analyze(display=False):
    """ In this part of the coursework I decided to start by analysing how the
    proportion of C villages changes for different values of b. As we will see
    from the plots and values below I identified three main patterns. The first
    pattern is when 1.5 =< b < 2, for these values, M villages expand however
    they do not take over completely as the values of fc oscilate between 0.95
    and 0.6. The second pattern is for 2 =< b < 10, this is when fc decreases
    drastically and the M villages take over in  all random simulations, leaving
    fc, hovering around 0 however not yet acheiving it. The third case is when
    b > 10, this is when fc does go to 0 and the Mercenary villages do take over
    completely. Thus b should be looked at as an Mercenary village agressivity
    parameter that increases the likelihood or odds of these so mentioned M
    villages of taking over.
    Note that I also introduced a fourth and last plot in which I used b values
    from the last three ranges to give the reader a clearer view of the overal
    variance of fc.
    """
    # I will use only these b values.
    b_values = [[1.5,1.55,1.60,1.65,1.70],[2.0, 2.3, 2.5, 5, 8],[10, 20, 30, 40, 50],[1.6, 2, 5, 8, 16]]
    b_values= np.array(b_values)
    # Initialize some needed values
    fc_array = np.zeros((4,5))
    #starting a loop for the plots to make the calculations easier
    for t in range(4):
        for i,b in enumerate(b_values[t,:]):
            Nt = 15
            _,fc = simulate2(b, Nt)
            fc_array[t,i] =fc[-1]
    if display:
        fig = plt.figure()
        for t in range(4):
            plt.subplot(2,2,t+1)
            plt.plot(b_values[t,:],fc_array[t,:],'x--')
            plt.xlabel('b values')
            plt.ylabel('fc values')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle('Anas Lasri from Analyze: fc variation.\n\n', fontsize=12)
        plt.savefig('hw11.png', dpi=300)
        plt.draw()
        plt.pause(0.001)
    b_value = [1.5, 2.0, 5.0, 20]
    b_value = np.array(b_value)
    fc1 = np.zeros((4,21))
    for i,b in enumerate(b_value):
        _,fc = simulate2(b_value[i],20)
        fc1[i,:] = fc
    if display:
        input("Press [enter] to continue")
        fig = plt.figure()
        plt.plot(np.linspace(1,15,21), fc1[0,:], 'g-',np.linspace(1,15,21), fc1[1,:], 'r-', np.linspace(1,15,21), fc1[2,:], 'b-', np.linspace(1,15,21), fc1[3,:], 'y-')
        plt.xlabel('time in years')
        plt.ylabel('fc')
        plt.legend(('b = 1.5','b = 2.0','b = 5.0','b = 20.0'), loc='best')
        plt.suptitle('Anas Lasri from Analyze \n Multiple fc visualizations against time')
        plt.savefig('hw12.png', dpi=300)
        plt.draw()
        plt.pause(0.001)
    return
if __name__ == '__main__':
    """ I added user interactivity with the code. It will ask the user to press
    enter
    """
    # The code here should call analyze and generate the
    # figures that you are submitting with your code
    output = analyze(True)
