import numpy as np
from scipy.optimize import minimize 
import math 
from gain import channel_gain 
import utils 
from options import args_parser 

#Size of the model
S = 100000
#Total Bandwidth = 1Mhz
B = 1000000
#inverse of N0 
N10 = 4*10**15
#8 bits per pixel, images in mnist have a size of 28 x 28 
Nbr_bits_pic = 6272 
#coefficients for tradeoff
roT = 1/4
roE = 1/4
roI = 1/2

def initialize(args, numEpochs=2, xmax = 500, ymax = 500):
	#bandwidth ratio
	args.num_users = 100
	alpha = [1/int(args.num_users) for i in range(int(args.num_users))]
	#CPU FREQUENCIES BETWEEN 1 AND 3 GhZ  
	CPUFreq = np.random.uniform(low=1, high=3, size=(args.num_users,))
	#transmission power
	Power = np.random.uniform(low=1, high=5, size=(args.num_users,))
	#PROCESSING DENSITY OF LEARNING TASKS (cycle/bit) 
	ProcDens = np.random.randint(low=10, high=30, size=(args.num_users,))
	#UEs and BS COORDINATES
	xyUEs = np.vstack((np.random.uniform(low=0.0, high=xmax, size=args.num_users),np.random.uniform(low=0.0, high=ymax, size=args.num_users))) 
	xyBSs = np.vstack((250,250))
	#CHANNEL GAIN
	Gains = channel_gain(args.num_users,1,1, xyUEs, xyBSs,0)
	#generate the datasets depending on args ( iid, num_users, balance )
	train_dataset, test_dataset, user_groups = utils.get_dataset(args)
	#the training set size in bits 
	TrainSize = [ numEpochs*Nbr_bits_pic*len(user_groups[i]) for i in range(args.num_users)]
	#print(TrainSize)
	#Calculate necessary time for training
	Ttrain = np.longdouble(np.zeros(args.num_users,))
	#Ttrain = size * processing density / cpu frequency ( it in GhZ, should be transformed to Hz by multiplying by 10**9 )
	for i in range(args.num_users):	
		Ttrain[i] = TrainSize[i]/1000000000
		Ttrain[i] *= ProcDens[i]/CPUFreq[i]
	gains=[]
	for i in range(args.num_users):
		gains.append(Gains[0,i,0])

		
	return Ttrain, alpha , gains , Power, train_dataset, test_dataset, user_groups


def newround(args):
	xyUEs = np.vstack((np.random.uniform(low=0.0, high=500, size=args.num_users),np.random.uniform(low=0.0, high=500, size=args.num_users)))
	xyBSs = np.vstack((250,250))
	Gains = channel_gain(args.num_users,1,1, xyUEs, xyBSs,0)
	gains=[]
	for i in range(args.num_users):
		gains.append(Gains[0,i,0])
	alpha = [1/int(args.num_users) for i in range(int(args.num_users))]
	return alpha, gains

#Calculate time for upload t_up
'''
def tup(alpha,gain):
	tup=[]
	for i in range(len(alpha)):
		tup.append(S/(alpha[i]*B*(math.log2(1+Power[i]*gain[0,i,0]*N10/(alpha[i]*B)))))
	return tup	
'''



if __name__ == '__main__':

	try:
		import cPickle as pickle
	except ImportError:  # python 3.x
		import pickle
	import os
	args = args_parser()
	direxp = 'simulations_'+str(args.num_users)+'u_'+str(args.local_ep)+'e_'+str(S)+'s'
	os.mkdir(direxp)
	nbrofrounds = 15
	for i in range(50):		
		Ttrain, alpha , gains , Power, train_dataset, test_dataset, user_groups = initialize(args)
		d = dict(((k, eval(k)) for k in ('Ttrain', 'alpha' , 'gains' , 'Power', 'train_dataset', 'test_dataset', 'user_groups')))
		dirname = direxp+'/sim'+str(i)
		os.mkdir(dirname)
		filename = dirname+'/run1_1.p'
		with open(filename, 'wb') as fp:
			pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
		file2 = dirname+'/rounds.p'	
		dico = {}
		for i in range(nbrofrounds):
			alpha, gain = newround(args)
			d = dict(((k, eval(k)) for k in ('alpha','gain')))
			dico[str(i)] = d
		with open(file2, 'wb') as fp:
			pickle.dump(dico, fp, protocol=pickle.HIGHEST_PROTOCOL)
