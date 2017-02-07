from numpy.random import laplace, normal, exponential, random_integers
from numpy import sqrt, exp
import numpy as np
import sys

# <!-- constants and variables -->
N = 943; M = 1682; K = 10;
MAX_RATING = 5; MIN_RATING = 1;
alpha = 0.001; beta = 0.01; epi = 1;

p = 2147483647

R = np.zeros((N, M))
P = np.random.rand(N,K) 
Q = np.random.rand(M,K)
testList = {}

ftrain = sys.argv[1]
ftest = sys.argv[2]
foutput = sys.argv[3]

# <!-- train data -->
with open(ftrain) as f:
	for line in f:
		numbers_str = line.split()
		numbers = [int(x) for x in numbers_str]
		u = numbers[0] - 1
		i = numbers[1] - 1 
		r = numbers[2]
		R[u][i] = r

# <!-- test data -->
with open(ftest) as f:
	for line in f:
		numbers_str = line.split()
		numbers = [int(x) for x in numbers_str]
		testList[(numbers[0]-1, numbers[1]-1)] = numbers[2]

# <!-- pre-mf -->
def preMF(R, P, Q, K, steps=100):
	Q = Q.T
	for step in xrange(steps):
	    for i in xrange(N):
	        for j in xrange(M):
	            if R[i][j] > 0:
	                eij = R[i][j] - np.dot(P[i,:],Q[:,j])
	                for k in xrange(K):
	                    P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-
	                        beta*P[i][k])
	                    Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-
	                        beta*Q[k][j]) 
	    eR = np.dot(P,Q)
	    e = 0
	    for i in xrange(len(R)):
	        for j in xrange(len(R[0])):
	            if R[i][j] > 0:
	                e = e+pow(R[i][j]-np.dot(P[i,:],Q[:,j]),2)           
	                for k in xrange(K):
	                    e = e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2))
	    print '%d: %f' % (step, e)
 	
 	return P
#
nP = preMF(R, P, Q, K)



