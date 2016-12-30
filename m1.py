from numpy.random import laplace, normal
from numpy import sqrt
import numpy as np
import sys

N = 943  # users
M = 1682 # items
K = 20   # latent features
maxr = 5.0
minr = 1.0

##### the epsilon value
epi = 0.1

Ratings = {}
Users = np.random.rand(N,K)
Items = np.random.rand(M,K)

Test = {}

def MF(R, P, Q, K, steps=100, alpha=0.002, beta=0.07):
    Q=Q.T
    for step in xrange(steps):
        for i in xrange(N):
            for j in xrange(M):
            	key = (i,j)
            	if key in R:
            		eij = R[key] - np.dot(P[i,:],Q[:,j])
            		for k in xrange(K):
				        P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])
				        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])             			
        # eR = np.dot(P,Q)
        e = 0	
        for i in xrange(N):
            for j in xrange(M):
            	key = (i,j)
            	if key in R:
                    e = e+pow(R[key]-np.dot(P[i,:],Q[:,j]),2)           
                    for k in xrange(K):
                        e = e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2))
    	print 'MF %d: %f' % (step, e)
        if e < 0.001:
            break
    return P, Q.T
def pred(file, R, P, Q, T):
	fw = open(file, 'w')
	Q = Q.T
	count = 0
	e = 0   # calculate rmse
	for key in T:
		i = key[0]
		j = key[1]
		r = np.dot(P[i,:],Q[:,j])
		fw.write('%d %d %f\n' % (i+1,j+1,r))
		if r  > 5:
			r = 5
		e +=  (r - T[key])**2
		count+=1
	fw.close()
	print (e/count)**0.5
# === read train file ===
def read1(file):
	with open(file) as f:
		for line in f:
			numbers_str = line.split()
			numbers = [int(x) for x in numbers_str]
			Ratings[(numbers[0]-1, numbers[1]-1)] = numbers[2]
# === read test file ===
def read2(file):
	with open(file) as f:
		for line in f:
			numbers_str = line.split()
			numbers = [int(x) for x in numbers_str]
			Test[(numbers[0]-1, numbers[1]-1)] = numbers[2]

### generate random noise array
def noise():
	f = laplace(0, 2*(maxr-minr)*sqrt(K)/epi,M*K)
	Noise = np.reshape(f, (M,K))
	return Noise

### Noise MF
def NoiseMF(R, P, Q, K, Noise, steps, alpha=0.002, beta=0.05):
    Q=Q.T
    for step in xrange(steps):
        for i in xrange(N):
            for j in xrange(M):
            	key = (i,j)
            	if key in R:
            		eij = R[key] - np.dot(P[i,:],Q[:,j])
            		for k in xrange(K):
						# P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])
				        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j]-Noise[j][k])           			
        # eR = np.dot(P,Q)
        e = 0	
        for i in xrange(N):
            for j in xrange(M):
            	key = (i,j)
            	if key in R:
                    e = e+pow(R[key]-np.dot(P[i,:],Q[:,j]),2)+np.dot(Noise[j,:],Q[:,j])  
                    for k in xrange(K):
                        e = e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2))
    	print 'Noise MF %d: %f' % (step, e)
        #if e < 0.001:
        #    break
    return Q.T




### ====  main ====
file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
read1(file1)
read2(file2)

### do origin matrix factorization first
nP, nQ = MF(Ratings, Users, Items, K)
pred(file3, Ratings, nP, nQ, Test)

### intial Noise array
Noise = noise()

### ADD NOISE to V
nnQ = NoiseMF(Ratings, nP, Items, K, Noise,20)
pred(file3, Ratings, nP, nnQ, Test)
nnQ = NoiseMF(Ratings, nP, Items, K, Noise,40)
pred(file3, Ratings, nP, nnQ, Test)
#nnQ = NoiseMF(Ratings, nP, nQ, K, Noise,8000)
#pred(file3, Ratings, nP, nnQ, Test)
#nnQ = NoiseMF(Ratings, nP, nQ, K, Noise,10000)
#pred(file3, Ratings, nP, nnQ, Test)



















