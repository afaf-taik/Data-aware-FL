import pickle 


def load_0(sim,dirname):
    
    filename = dirname+ '/sim'+str(sim)+'_results.p'
    with open(filename, 'rb') as f:
        x = pickle.load(f)            
    return x[str(1)]['indexes']

def load(sim,rnd,dirname):
    if(rnd==0):
        return load_0(sim,dirname)
    else:    
        filename = dirname+'/sim'+str(sim)+'_results'+str(rnd-1)+'.p'
        with open(filename, 'rb') as f:
            x = pickle.load(f)            
        return x[str(rnd)]['indexes']

def load_data(sim):    
    filename = 'simulations_2ep_100u/sim'+str(sim)+'/run1_1.p'
    with open(filename, 'rb') as f:
        x = pickle.load(f)            
    return  x['train_dataset'], x['test_dataset'], x['user_groups']

if __name__ == '__main__':
	for i in range (15):
		print(load(1,i,'simulations_importance5'))

