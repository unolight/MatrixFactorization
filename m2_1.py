from numpy.random import laplace, normal, exponential, random_integers
from numpy import sqrt, exp
import numpy as np
import sys

# <!-- constants and variables -->
N = 943; M = 1682; K = 10;
MAX_RATING = 5; MIN_RATING = 1;
alpha = 0.00001; beta = 0.01; epi = 1;
p = 2147483647

R = np.zeros((N, M))
P = np.random.rand(N,K) 
Q = np.random.rand(M,K)
testList = {}

ftrain = sys.argv[1]
ftest = sys.argv[2]
# foutput = sys.argv[3]

# <!-- train data -->
print('read train')
with open(ftrain) as f:
	for line in f:
		numbers_str = line.split()
		numbers = [int(x) for x in numbers_str]
		u = numbers[0] - 1
		i = numbers[1] - 1 
		r = numbers[2]
		R[u][i] = r

# <!-- test data -->
print('read test')
with open(ftest) as f:
	for line in f:
		numbers_str = line.split()
		numbers = [int(x) for x in numbers_str]
		testList[(numbers[0]-1, numbers[1]-1)] = numbers[2]

# <!-- pre-mf -->
print ('start pre-MF')

def preMF(R, P, Q, K, alpha=0.02, beta=0.1, steps=1):
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
nP = preMF(R, P, Q, K)

# <!-- class define -->
## server
class Server:
	def __init__(self):
		self._q = np.random.rand(M,K)
	def descent(self, grad):
		self._q = self._q - grad
## client
class Client:
	_userNum = 0
	def __init__(self, uid, r, p):
			self._id = uid
			self._p = p
			self._r = r
			Client._userNum += 1
	# gradient descent
	def gradient(self, q):
		grad = np.zeros((M,K))
		for j in xrange(M):
			if self._r[j] != 0:
				eij = self._r[j]-np.dot(self._p, q[:,j])
				for k in xrange(K):
					grad[j][k]=-alpha*(2*eij*self._p[k]-beta*q[k][j]) 
		return grad
	def error(self, q):
		e = 0
		for j in xrange(M):
		    if u._r[j] > 0:
		        e = e+pow(self._r[j]-np.dot(self._p, q[:,j]),2)           
		        for k in xrange(K):
		            e = e+(beta/2)*(pow(self._p[k],2)+pow(q[k][j],2))
		return e
## semi-server
class SemiServer:
	def __init__(self):
		self._grad = np.zeros((M,K))
	def clear(self):
		self._grad = np.zeros((M,K))
	def add(self, grad):
		self._grad = self._grad+grad
	def send(self,server):
		server.descent(self._grad)
		self.clear()
##
U = [Client(i, R[i], nP[i]) for i in xrange(N)]
server = Server()
semiServer = SemiServer()

steps = 10
for step in xrange(steps):
	print 'step %d: ' % (step) 
	Q = server._q
	Q = Q.T
	for u in U:	
		grad = u.gradient(Q)
		semiServer.add(grad)
	semiServer.send(server)
	
	## error value
	e = 0
	Q = server._q
	Q = Q.T
	for u in U:
	   e = e + u.error(Q)
	print e

## prediction
print ('testing')
def pred(U, S, tList):
	Q = S._q
	Q = Q.T
	e = 0
	for key in tList:
		i = key[0]; j = key[1]; tr = tList[key];
		r = np.dot(U[i]._p, Q[:,j])
		if r >= 5:
			r = 5
		if r <= 1:
			r = 1
		e = e+(r-tr)**2
	rmse = (e/len(tList))**0.5
	print rmse	
pred(U, server, testList)



