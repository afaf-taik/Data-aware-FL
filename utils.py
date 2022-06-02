# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import numpy as np

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = 'data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset == 'mnist':
            data_dir = 'data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        else:
            data_dir = 'data/fmnist/'
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                print('i am non iid unequal')
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

'''
entropy = []
for k, v in d.items():
    en=0
    counts=np.zeros(10)
    for i in v: 
        _,lbl = dataset_train[int(i)]
        counts[lbl]+=1

    for j in range (len(counts)):
        counts[j]=counts[j]/len(v)
        if(counts[j]!=0):
            en+= -counts[j]*math.log(counts[j])

    entropy.append(en)
print(entropy)
'''
import math
def get_entropy(user_groups,dataset_train):
    entropy = []
    for k, v in user_groups.items():
        en=0
        counts=np.zeros(10)
        for i in v: 
            _,lbl = dataset_train[int(i)]
            counts[lbl]+=1

        for j in range (len(counts)):
            counts[j]=counts[j]/len(v)
            if(counts[j]!=0):
                en+= -counts[j]*math.log(counts[j])
        entropy.append(en)
    return entropy

def get_gini(user_groups,dataset_train):
    entropy = []
    #dico = {}
    for k, v in user_groups.items():
        en = 0
        counts = np.zeros(10)
        for i in v: 
            _,lbl = dataset_train[int(i)]
            counts[lbl] += 1

        for j in range (len(counts)):
            counts[j] = counts[j]/len(v)
            en += counts[j]**2
        en = 1 - en
        entropy.append(en)
        #dico[str(i)] = en
    return entropy


def get_importance(entropy,size,age,rnd,roE=1/3,roD = 1/3, roA = 1/3):
    importance=[]
    totalSize = sum(size)
    for i in range(len(entropy)):
        importance.append(roE* entropy[i] + roD*size[i]/totalSize + roA*age[i]/rnd)
    return importance


def get_order(importance):
    e=[]
    for i in range(len(importance)):
        e.append(-1*importance[i])
    return np.argsort(e)


'''
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

'''

def average_weights(w,s):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    m=sum(s)
    for key in w_avg.keys():
        w_avg[key] = s[0]*w_avg[key]
        for i in range(1, len(w)):
            w_avg[key] += s[i]*w[i][key]
        w_avg[key] = torch.div(w_avg[key], m)
    return w_avg
#'''
def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def get_importance(entropy,size,age,rnd,roE=1/3,roD=1/3,roA =1/3):
    importance = []
    totalsize = sum(size)
    #print(totalsize)
    for i in range(len(entropy)):
        importance.append(roE*entropy[i] + roD * size[i]/totalsize + roA*age[i]/rnd )
    return importance


def get_age(age):
    a = []
    for i in range(len(age)):
        a.append(math.log2(1+age[i]))
    return a