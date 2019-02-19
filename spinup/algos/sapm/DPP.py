import numpy as np
from itertools import product
"""
excerpt from https://github.com/mehdidc/dpp/blob/master/dpp.py
"""

def build_similary_matrix(cov_function, items):
    """
    build the similarity matrix from a covariance function
    cov_function and a set of items. each pair of items
    is given to cov_function, which computes the similarity
    between two items.
    """
    L = np.zeros((len(items), len(items)))
    for i in range(len(items)):
        for j in range(i, len(items)):
            L[i, j] = cov_function(items[i], items[j])
            L[j, i] = L[i, j]
    
    L = L + 0.01*np.identity(np.shape(L)[0])
    return L


def exp_quadratic(sigma):
    def f(p1, p2):
        return np.exp(-(((p1 - p2)**2).sum()) / sigma**2)
    return f


def sample_k(actions, candidates, sig, k):
    
    items = np.concatenate((actions, candidates))
    L = build_similary_matrix(exp_quadratic(sigma=sig), items)
    
    L_tmp = np.ones((len(items), len(items)))
    L_tmp[:len(actions), :len(actions)] = L[:len(actions), :len(actions)]
    
    for i in range(k):
        idx_det = np.zeros(len(candidates))
        
        for j in range(len(candidates)):
            L_tmp[len(actions), :len(actions)] = L[len(actions)+j,:len(actions)]
            L_tmp[:len(actions), len(actions)] = L[:len(actions),len(actions)+j]
            
            idx_det[j] = np.linalg.det(L_tmp[:len(actions)+1, :len(actions)+1])
        
        idx_max = np.argmax(idx_det)
        L_tmp[len(actions), :len(actions)] = L[len(actions)+idx_max,:len(actions)]
        L_tmp[:len(actions), len(actions)] = L[:len(actions),len(actions)+idx_max]
        
        if idx_max != 0:
            tmp = np.array(L[len(actions),:])
            L[len(actions),:] = L[(len(actions)+idx_max),:]
            L[(len(actions)+idx_max),:] = tmp
            
            tmp = np.array(L[:,len(actions)])
            L[:,len(actions)] = L[:,(len(actions)+idx_max)]
            L[:,(len(actions)+idx_max)] = tmp[:]
            
        actions = np.append(actions, [candidates[idx_max]], axis=0)
        candidates = np.delete(candidates, idx_max, axis=0)
        
        
    return actions        
            
