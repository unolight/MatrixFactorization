from numpy.random import laplace, normal, exponential, random_integers
from numpy import sqrt
import numpy as np
import sys

N = 943  # users
M = 1682 # items
K = 20   # latent features, between 20 to 100
P = 2147483647
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
def initUserArray():
	return np.random.rand(1, K)

def initItemArray():
	return np.random.rand(M,K)

def parseUserRatings(uid):
	# Ratings
	r = np.reshape(Ratings[uid], (1, len(Ratings[uid])))
	return r

def Noise():
	f = laplace(0, 2*(maxr-minr)*sqrt(K)/epi,k*M)
	noise = np.reshape(f, (M,k))
	return noise

def initGrad():
	return np.zeros((M,K))


# Users = np.random.rand(N,K)
# Items = np.random.rand(M,K)

class Server:
	def __init__(self):
		self.V = initItemArray()
	def randomNoise():
		return random_integers(0,P)

class Client:
	# p = []  user preference, confidential
	# noise = []  perturbation array
	t = 0  # the tth iteration
	count = 0
	def __init__(self, uid):
		self.uid = uid
		self.p = initUserArray()
		self.R = parseUserRatings(uid)
		self.noise = Noise()
		self.noise2 = Noise()
		Client.count+=1
	def gradient(self, Q):
		self.noise2= Noise()
		# print self.noise2
		grad = initGrad()
	    for j in xrange(M):
	    	if R[0][j] != 0:
	    		eij = self.R[0][j] - np.dot(self.p[0,:],Q[:,j])
	    		for k in xrange(K):
					grad[j][k]=Q[k][j]+alpha*(2*eij*self.p[0][k]-beta*Q[k][j]) \
					+self.noise[j][k]+self.noise2[j][k]
		return uid, grad

class semiServer:
	def __init__(self):
		gradCount = np.zeors((N,M))



### ====  main ====
steps = 1000

train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
read1(train)
read2(test)

U = [Client(i) for i in xrange(N)]
# print U[0].p.shape, U[0].R.shape, U[0].grad.shape, U[0].noise.shape
server = Server()
semiServer = SemiServer()

for step in xrange(steps):
	for ui in U:






