import numpy as np
from scipy.optimize import minimize 
import math 
from gain import channel_gain 
import utils 
from options import args_parser 
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import os
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

numexperiment = '100u_3e_100000s'

dirsim = 'simulations_'+numexperiment
dirres = 'simulations_baseline'+numexperiment

os.mkdir(dirres)
tries = 50
nbr_rounds = 15
def load(sim):
    
    filename = dirsim + '/sim'+str(sim)+'/run1_1.p'
    with open(filename, 'rb') as f:
        x = pickle.load(f)

            
    return x['Ttrain'], x['alpha'] , x['gains'] , x['Power'], x['train_dataset'], x['test_dataset'], x['user_groups']


def newround(sim, nbr):
    filename = dirsim + '/sim'+str(sim)+'/rounds.p'
    with open(filename, 'rb') as f:
        x = pickle.load(f)
    return x[str(nbr)]



def tup(alpha,gain):
    tup=[]
    for i in range(len(alpha)):
        tup.append(S/(alpha[i]*B*(math.log2(1+Power[i]*gain[i]*N10/(alpha[i]*B)))))
    return tup  


#objective function for P2
def objective2(alpha):
    Tround = []
    Tup = tup(alpha, gains)
    for i in range(len(alpha)):
        Tround.append((Tup[i]+Ttrain[i]))
    T= max(Tround)/Tmax
    #print('T=',T)
    Ek = [Power[i]*Tup[i] for i in range(len(alpha))]
    E=sum(Ek)/Emax
    #print('E=',E)
    return roE*E+roT*T  

def results(alpha):
    Tround = []
    Tup = tup(alpha, gains)
    for i in range(len(alpha)):
        Tround.append((Tup[i]+Ttrain[i]))
    T= max(Tround)
    #print(T)
    Ek = [Power[i]*Tup[i] for i in range(len(alpha))]
    E=sum(Ek)
    #print(E)
    return E,T  

def constraint2(alpha):
    return 1-sum(alpha)

def constraint3(alpha):
    return sum(alpha)-0.95

def constraint1(x):
    return sum(x)-2

if __name__ == '__main__':
    args = args_parser()
    
    for nbr in range(tries):

        Ttrain, alpha , gains , Power, train_dataset, test_dataset, user_groups = load(nbr)
        A = np.zeros(args.num_users)
        size = [len(user_groups[idx]) for idx in range(args.num_users)]
        #Index= utils.get_gini(user_groups,train_dataset)
        #Importance = utils.get_importance(Index,size,A,15) 
        Tup=tup(alpha,gains)
        Tround=[]
        for i in range(len(alpha)):
            Tround.append((Tup[i]+Ttrain[i]))
        Tmax = max(Tround)
        Emax = sum([Power[i]*Tup[i] for i in range(len(alpha))])
        con2 ={'type':'eq','fun': constraint2}
        con3 ={'type':'ineq','fun': constraint3}
        cons=[con2,con3]
        indexes = list(range(100))
        alpha0=[1/len(indexes) for i in range(len(indexes))]
        bnds=[]
        b1=(0.0,1.0)
        for k in range(len(alpha0)):
            bnds.append(b1)
        bnds2=tuple(bnds)
        sol2 = minimize(objective2, alpha0, method='SLSQP', bounds=bnds2, constraints=cons)             
        
        alpha = sol2.x
        dico = {}
        dico['E'], dico['T'] = results(alpha)
        resultsfile = dirres+'/sim'+str(nbr)+'_results.p'
        dico1={'1': dico }
        
        with open(resultsfile, 'wb') as fp:
            pickle.dump(dico1, fp, protocol=pickle.HIGHEST_PROTOCOL)

        for rnd in range(nbr_rounds):
            d = newround(nbr,rnd)

            alpha,gains = d['alpha'],d['gain']
            #Index= utils.get_entropy(user_groups,train_dataset)
            #Index = np.ones(args.num_users) 
            Tup=tup(alpha,gains)
            alpha0=[1/len(indexes) for i in range(len(indexes))]
            bnds=[]
            b1=(0.0,1.0)
            for k in range(len(alpha0)):
                bnds.append(b1)
            bnds2=tuple(bnds)
            sol2 = minimize(objective2, alpha0, method='SLSQP', bounds=bnds2, constraints=cons) 
            alpha = sol2.x

            dico = {}
            dico['E'], dico['T'] = results(alpha)
            resultsfile = dirres + '/sim'+str(nbr)+'_results'+str(rnd)+'.p'
            dico1={str(rnd+1): dico }
            with open(resultsfile, 'wb') as fp:
                pickle.dump(dico1, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print( 'rnd',rnd,'is done' )




