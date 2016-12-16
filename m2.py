from numpy.random import laplace, normal, exponential
from numpy import sqrt
import numpy as np
import sys

N = 943  # users
M = 1682 # items
K = 20   # latent features, between 20 to 100
maxr = 5
minr = 1

epi = 1
Ratings = np.zeros((943,1682))
Test = {}

#####
# def noise():
# 	f = laplace(0, 2*(maxr-minr)*sqrt(K)/epi,M*K)
# 	Noise = np.reshape(f, (M,K))
# 	return Noise

# def random_number(C, H):
# 	h = exponential(1)
# 	C = normal(0,1)
# 	return C, h

# def MF(R, P, Q, K, steps=100, alpha=0.002, beta=0.05):
#     Q=Q.T
#     for step in xrange(steps):
#         for i in xrange(N):
#             for j in xrange(M):
#             	key = (i,j)
#             	if key in R:
#             		eij = R[key] - np.dot(P[i,:],Q[:,j])
#             		for k in xrange(K):
# 						P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-
# 						    beta*P[i][k])
# 						Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-
# 						    beta*Q[k][j])             			
#         # eR = np.dot(P,Q)
#         e = 0	
#         for i in xrange(N):
#             for j in xrange(M):
#             	key = (i,j)
#             	if key in R:
#                     e = e+pow(R[key]-np.dot(P[i,:],Q[:,j]),2)           
#                     for k in xrange(K):
#                         e = e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2))
#     	print '%d: %f' % (step, e)
#         if e < 0.001:
#             break
#     return P, Q.T
#####

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
			u = numbers[0]-1
			it = numbers[1]-1
			rate = numbers[2]
			Ratings[u][it] = rate

# === read test file ===
def read2(file):
	with open(file) as f:
		for line in f:
			numbers_str = line.split()
			numbers = [int(x) for x in numbers_str]
			Test[(numbers[0]-1, numbers[1]-1)] = numbers[2]


### intial functions 
def uArray():
	return np.random.rand(1, K)

def sArray():
	return np.random.rand(M,K)

def ratings(uid):
	# Ratings
	r = np.reshape(Ratings[uid], (1, len(Ratings[uid])))
	return r

def Noise():
	f = laplace(0, 2*(maxr-minr)*sqrt(K)/epi,1*M)
	noise = np.reshape(f, (1,M))
	return noise

def initGrad():
	return np.zeros((1,K))


# Users = np.random.rand(N,K)
# Items = np.random.rand(M,K)

class Server:
	def __init__(self):
		self.V = sArray()

class Client:
	# p = []  user preference, confidential
	# noise = []  perturbation array
	# noise2 = []  pertubation array 2
	t = 0  # the tth iteration
	count = 0
	def __init__(self, uid):
		self.uid = uid
		self.p = uArray()
		self.R = ratings(uid)
		self.grad = initGrad()
		self.noise = Noise()
		# self.noise2 = noise2()
		Client.count+=1
	def gradient(self, Q):
	    for j in xrange(M):
	    	key = (uid,j)
	    	if key in R:
	    		eij = self.R[key] - np.dot(self.p[0,:],Q[:,j])
	    		for k in xrange(K):
					grad[0][j]=Q[k][j]+alpha*(2*eij*self.p[0][k]-beta*Q[k][j])+self.noise[0][k]
		return uid, grad

### ====  main ====
file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
read1(file1)
read2(file2)

U = [Client(i) for i in xrange(N)]
server = Server()


