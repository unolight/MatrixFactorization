from numpy.random import laplace, normal, exponential, random_integers
from numpy import sqrt, exp
import numpy as np
import sys

N = 943  # users
M = 1682 # items
K = 8  # latent features, between 20 to 100
P = 1e9
maxr = 5 # max rating score
minr = 1 # min rating score

alpha=0.01
beta=0.001
epi = 1e9

Ratings = np.zeros((943,1682))
test_list = {}

def pred(file, U, Q, test_list):
    fw = open(file, 'w')
    Q = Q.T
    count = 0
    e = 0   # calculate rmse
    for key in test_list:
        i = key[0]
        j = key[1]
        r = np.dot(U[i].p[0,:],Q[:,j])
        fw.write('%d %d %f\n' % (i+1,j+1,r))
        if r  > 5:
            r = 5
        e =  (e+(r - test_list[key])**2)%P
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
            test_list[(numbers[0]-1, numbers[1]-1)] = numbers[2]


### intial functions 
def parseUserRatings(uid):
    return Ratings[uid]

class Server:
    def __init__(self):
        self.Q = np.random.rand(M,K)
        self.grad = np.zeros((M,K))
        self.ratingCount = np.zeros((M,))
        self.uphi = np.zeros((M,K))
    def randomNoiseVector(self, it):
        n=random_integers(0,P,K)
        self.add_phi(it,n)
        return n
    def randomNumberVector(self):
        return exponential(1, K)
    def init_grad(self):
        self.grad = np.zeros((M,K))
    def init_phi(self):
        self.uphi = np.zeros((M,K))
    def updateQ(self, semi_grad):
        self.grad = self.cal_grad(semi_grad)
	self.Q += self.grad
    def add_phi(self, it, n):
        self.uphi[it] = (self.uphi[it] + n) % P
    def cal_grad(self, ss_grad):
	return (ss_grad - self.uphi)
class Client:
    t = 0  # the tth iteration
    count = 0
    def __init__(self, uid):
        self.uid = uid
        self.p = np.random.rand(1,K)
        self.R = parseUserRatings(uid)
        self.C = np.zeros((M, K))
        self.eta = np.zeros((M,K))
        self.rho = np.zeros((M,K)) # iteration random noise
        self.phi = np.zeros((M,K))
        Client.count+=1
    def initGrad(self):
        return np.zeros((M,K))
    def randomNormalVector(self, j, count):
        self.C[j] = normal(0, 1.0/count, K)
    def genNoise(self, j, H):
        self.eta[j] = 2*(maxr-minr)*sqrt(K)/epi*sqrt(2*H)*self.C[j]
    def genNoise2(self, j, H):
        self.rho[j] = 2*(maxr-minr)*sqrt(K)/epi*sqrt(2*H)*self.C[j]
    def gradient(self, Q):
        grad = self.initGrad()
        for j in xrange(M):
            if self.R[j] != 0:
                eij = self.R[j] - np.dot(self.p[0,:],Q[:,j])
                for k in xrange(K):
                    grad[j][k]=(alpha*(2*eij*self.p[0][k]-beta*Q[k][j])+self.eta[j][k]+self.rho[j][k]+self.phi[j][k])%P
        return grad
    def error(self, Q):
        e = 0
        for j in xrange(M):
            if  self.R[j] > 0:
                e = (e+pow(self.R[j]-np.dot(self.p[0,:],Q[:,j]),2))%P      
                for k in xrange(K):
                    e = (e+(beta/2)*(pow(self.p[0][k],2)+pow(Q[k][j],2)))%P
        return e


class SemiServer:
    def __init__(self):
        self.grad = np.zeros((M,K))
    def initgrad(self):
	self.grad = np.zeros((M,K))	
    def add_grad(self, user_grad):
	self.grad = (self.grad+user_grad) % P

### ====  main ====
train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
read1(train)  # read Ratings array
read2(test)

U = [Client(i) for i in xrange(N)]
# print U[0].p.shape, U[0].R.shape, U[0].grad.shape, U[0].noise.shape
server = Server()
semiServer = SemiServer()

### produce ratecount array: how many users have rated item
for i in xrange(N):
    for j in xrange(M):
        if Ratings[i][j] > 0:
            server.ratingCount[j] += 1

#####   set Noise  #####
for j in xrange(M):
    H = server.randomNumberVector()
    count = server.ratingCount[j]
    for i in xrange(N):
        if U[i].R[j] > 0:
            U[i].randomNormalVector(j, count)
            U[i].genNoise(j, H)

steps = 10
for step in xrange(steps):
    print 'step%d:' % (step)
    Q = server.Q
    Q = Q.T
    server.init_grad()
    server.init_phi()
    semiServer.initgrad()
    ### all user's noise 2 and phi
    for j in xrange(M):
        H = server.randomNumberVector()
        count = server.ratingCount[j]
        for i in xrange(N):
            if U[i].R[j] > 0:
                U[i].randomNormalVector(j, count)
                U[i].genNoise2(j, H)
                U[i].phi[j]=server.randomNoiseVector(j)
    
    for ui in U:
        semiServer.add_grad(ui.gradient(Q))
    server.updateQ(semiServer.grad)

    nQ = server.Q
    nQ = nQ.T
    err = 0
    for ui in U:
        err = (err + ui.error(nQ))%P
    print('err: %f' % (err))
pred(output, U, server.Q, test_list)









