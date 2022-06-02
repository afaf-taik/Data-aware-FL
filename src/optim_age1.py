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

numexperiment = '100u_2e_100000s'
dirres = 'simulations_age_'+numexperiment
dirsim = 'simulations_'+numexperiment
os.mkdir(dirres)

def load(sim):
    
    filename = dirsim+'/sim'+str(sim)+'/run1_1.p'
    with open(filename, 'rb') as f:
        x = pickle.load(f)

            
    return x['Ttrain'], x['alpha'] , x['gains'] , x['Power'], x['train_dataset'], x['test_dataset'], x['user_groups']


def newround(sim, nbr):
    filename = dirsim+'/sim'+str(sim)+'/rounds.p'
    with open(filename, 'rb') as f:
        x = pickle.load(f)
    return x[str(nbr)]



def tup(alpha,gain):
    tup=[]
    for i in range(len(alpha)):
        tup.append(S/(alpha[i]*B*(math.log2(1+Power[i]*gain[i]*N10/(alpha[i]*B)))))
    return tup  

#objective function for P1
def objective(x):
    Tround=[]
    I=0
    for i in range(len(x)):
        Tround.append((Tup[i]+Ttrain[i])*x[i])
        I += Importance[i]*x[i]/(args.num_users*math.log2(2+rnd))
    T = max(Tround)/Tmax 
    print('T=',T)
    #print(T)
    Ek = [Power[i]*Tup[i]*x[i] for i in range(len(alpha))]
    E=sum(Ek)/Emax
    print('E=',E)
    print('I=',I)
    return roE*E+roT*T-roI*I

#objective function for P2
def objective2(alpha):
    Tround = []
    Tup = tup(alpha, Gains_s)
    for i in range(len(alpha)):
        Tround.append((Tup[i]+Ttrain_s[i]))
    T= max(Tround)/Tmax
    #print('T=',T)
    Ek = [P_s[i]*Tup[i] for i in range(len(alpha))]
    E=sum(Ek)/Emax
    #print('E=',E)
    #print('I=',E)
    return roE*E+roT*T  

def results(alpha):
    Tround = []
    Tup = tup(alpha, Gains_s)
    for i in range(len(alpha)):
        Tround.append((Tup[i]+Ttrain_s[i]))
    T= max(Tround)
    #print(T)
    Ek = [P_s[i]*Tup[i] for i in range(len(alpha))]
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
    for nbr in range(50):
        rnd = 0 
        Ttrain, alpha , gains , Power, train_dataset, test_dataset, user_groups = load(nbr)
        A = np.zeros(args.num_users)
        Importance = utils.get_age(A) 
        Tup=tup(alpha,gains)
        Tround=[]
        for i in range(len(alpha)):
            Tround.append((Tup[i]+Ttrain[i]))
        Tmax = max(Tround)
        Emax = sum([Power[i]*Tup[i] for i in range(len(alpha))])
        b1=(0.0,1.0)
        bnds=[] 
        for k in range(args.num_users):
            bnds.append(b1)
        bnds1=tuple(bnds)   
        x0 = [ 0.99 for i in range(args.num_users)]
        con1 ={'type':'ineq','fun': constraint1}
        cons1=[con1]
        con2 ={'type':'eq','fun': constraint2}
        con3 ={'type':'ineq','fun': constraint3}
        cons=[con2,con3]
        sol = minimize(objective,x0,method='SLSQP',bounds=bnds1,constraints=cons1)

        x = sol.x
        print(x)
        indexes=[]
        P_s=[]
        Gains_s=[]
        Ttrain_s=[]
        for i in range(len(x)):
            if(x[i]>0.5):
                indexes.append(i)
                P_s.append(Power[i])
                Gains_s.append(gains[i])
                Ttrain_s.append(Ttrain[i])
        if(indexes==[]):
                i = np.argmax(x)
                indexes.append(i)
                P_s.append(Power[i])
                Gains_s.append(gains[i])
                Ttrain_s.append(Ttrain[i])
        print('indexes: ',indexes)
        for x in range(args.num_users):
            if x in indexes:
                A[x] = 0 
            else:
                A[x]+=1


        alpha0=[1/len(indexes) for i in range(len(indexes))]
        bnds=[]
        for k in range(len(alpha0)):
            bnds.append(b1)
        bnds2=tuple(bnds)
        sol2 = minimize(objective2, alpha0, method='SLSQP', bounds=bnds2, constraints=cons)             
        MAXITER = 5
        convergence = 0
        ITER = 0
        y = np.ones(len(indexes))
        alpha = sol2.x
        while(ITER<MAXITER and convergence == 0):
            bnds=[]
            for i in range(len(y)):
                bnds.append(b1)
            bnds1=tuple(bnds)   

            sol = minimize(objective, y, method='SLSQP', bounds=bnds1)
            comp = y == sol.x
            if(comp.all()):
                convergence = 1 
                break
            oldindexes = indexes
            x = sol.x
            indexes=[]
            P_s=[]
            Gains_s=[]
            Ttrain_s=[] 
            for i in range(len(x)):
                if(x[i]>0.5):
                    indexes.append(i)
                    P_s.append(Power[i])
                    Gains_s.append(gains[i])
                    Ttrain_s.append(Ttrain[i])
            if(indexes==[]):
                indexes = oldindexes
                P_s=[]
                Gains_s=[]
                Ttrain_s=[]
                for i in indexes:
                    P_s.append(Power[i])
                    Gains_s.append(gains[i])
                    Ttrain_s.append(Ttrain[i])
                break

            alpha0=[1/len(indexes) for i in range(len(indexes))]
            y = np.ones(len(indexes))
            bnds=[]
            for k in range(len(alpha0)):
                bnds.append(b1)
            bnds2=tuple(bnds)
            sol2 = minimize(objective2,alpha0, method='SLSQP', bounds=bnds2, constraints=cons)  
            alpha = sol2.x
            ITER+=1 

        dico = {}
        dico['x']= sol.x
        dico['indexes'] = indexes
        print(indexes)
        print(alpha)
        dico['E'], dico['T'] = results(alpha)
        resultsfile = dirres+'/sim'+str(nbr)+'_results.p'
        dico1={'1': dico }
        with open(resultsfile, 'wb') as fp:
            pickle.dump(dico1, fp, protocol=pickle.HIGHEST_PROTOCOL)
########################################################################################################
################
################################################################################


########################################################################################################
        for rnd in range(15):
            d = newround(nbr,rnd)
            alpha,gains = d['alpha'],d['gain']
            #Index= utils.get_entropy(user_groups,train_dataset)
            #Index = np.ones(args.num_users) 
            Tup=tup(alpha,gains)
            Importance = utils.get_age(A)
            b1=(0.0,1.0)
            bnds=[] 
            for k in range(args.num_users):
                bnds.append(b1)
            bnds1=tuple(bnds)   
            x0 = [ 0.99 for i in range(args.num_users)]
            con2 ={'type':'eq','fun': constraint2}
            cons=[con2]
            sol = minimize(objective,x0,method='SLSQP',bounds=bnds1,constraints = cons1)        
            x = sol.x
            oldindexes=indexes
            indexes=[]
            P_s=[]
            Gains_s=[]
            Ttrain_s=[]
            for i in range(len(x)):
                if(x[i]>0.5):
                    indexes.append(i)
                    P_s.append(Power[i])
                    Gains_s.append(gains[i])
                    Ttrain_s.append(Ttrain[i])
            if(indexes==[]):
                indexes = oldindexes
                P_s=[]
                Gains_s=[]
                Ttrain_s=[]
                for i in indexes:
                    P_s.append(Power[i])
                    Gains_s.append(gains[i])
                    Ttrain_s.append(Ttrain[i])
                break
            alpha0=[1/len(indexes) for i in range(len(indexes))]
            bnds=[]
            for k in range(len(alpha0)):
                bnds.append(b1)
            bnds2=tuple(bnds)
            sol2 = minimize(objective2, alpha0, method='SLSQP', bounds=bnds2, constraints=cons) 
            for x in range(args.num_users):
                if x in indexes:
                    A[x] = 0 
                else:
                    A[x]+=1
            MAXITER = 5
            convergence = 0
            ITER = 0
            y = np.ones(len(indexes))
            alpha = sol2.x

            
            while(ITER<MAXITER and convergence == 0):
                bnds=[]
                for i in range(len(y)):
                    bnds.append(b1)
                bnds1=tuple(bnds)   

                sol = minimize(objective, y, method='SLSQP', bounds=bnds1)
                comp = y == sol.x
                if(comp.all()):
                    convergence = 1 
                    break
                x = sol.x
                oldindexes = indexes
                indexes=[]
                P_s=[]
                Gains_s=[]
                Ttrain_s=[] 
                for i in range(len(x)):
                    if(x[i]>0.5):
                        indexes.append(i)
                        P_s.append(Power[i])
                        Gains_s.append(gains[i])
                        Ttrain_s.append(Ttrain[i])
                if(indexes==[]):
                    indexes = oldindexes
                    P_s=[]
                    Gains_s=[]
                    Ttrain_s=[]
                    for i in indexes:
                        P_s.append(Power[i])
                        Gains_s.append(gains[i])
                        Ttrain_s.append(Ttrain[i])
                    break

                alpha0=[1/len(indexes) for i in range(len(indexes))]
                y = np.ones(len(indexes))
                bnds=[]
                for k in range(len(alpha0)):
                    bnds.append(b1)
                bnds2=tuple(bnds)
                sol2 = minimize(objective2,alpha0, method='SLSQP', bounds=bnds2, constraints=cons)  
                alpha = sol2.x
                ITER+=1 


            dico = {}
            dico['x']= sol.x
            dico['indexes'] = indexes
            print(indexes)
            dico['E'], dico['T'] = results(alpha)
            resultsfile = dirres+'/sim'+str(nbr)+'_results'+str(rnd)+'.p'
            dico1={str(rnd+1): dico }
            with open(resultsfile, 'wb') as fp:
                pickle.dump(dico1, fp, protocol=pickle.HIGHEST_PROTOCOL)



