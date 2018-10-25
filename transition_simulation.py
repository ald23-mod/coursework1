""" This function will simulate the transition of villages.
"""

""" The way the simulation will be handled is, I will generate an n*n random
matrix of values between 0 and 1. If the probability of transition of a
village to a state C, say, is 0.90. I will compare this with the randomly
generate number from the matrix and hence determine whether the transition
will happen or not
"""
probability_criteria = np.random.rand(N,N);
