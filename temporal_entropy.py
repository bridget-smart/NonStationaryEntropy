"""
Preamble for most code and jupyter notebooks
@author: bridgetsmart
@notebook date: 3 Apr 2023
"""
from numba import jit, prange
import numpy as np
import pandas as pd

from ProcessEntropy.SelfEntropy import self_entropy_rate
from tqdm.notebook import tqdm

from collections.abc import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
# rng set up
from numpy.random import default_rng
import numpy as np
from scipy.interpolate import splrep, BSpline

rng = default_rng()

# need a function to flatten irregular list of lists
def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

# set up sequence generating functions
def simulate_bernoulli(N_parallel_sims,p):
    return rng.binomial(1,p,size =N_parallel_sims)

def assess_fit(y,y_fit):
    # residual sum of squares
    ss_res = np.sum((y - y_fit) ** 2)

    # total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return ss_res, ss_tot 

def gen_seq(T, t_del, time_variance,get_val, N_parallel_sims):
    '''
    Function to generate a seqence of discrete values between 0 and T (max time)
    with time steps of size t_del. 

    These sequences are drawn from a non-homogeneous Bernoulli function which has 
    a probably of success given by time_variance.

    N_parallel_sims dictates the number of simulations run.
    '''
    seq = []
    # true_ent = []
    for t in np.arange(0,T,t_del):
        p = time_variance(t,get_val)
        seq.append(simulate_bernoulli(N_parallel_sims,p))
        # true_ent.append(-p*np.log2(p)-(1-p)*np.log2(1-p))

    seq = np.array(seq)
    return np.arange(0,T,t_del), seq

 # entropy of each subsequence
def gen_sliding_window(m, seq,N_parallel_sims):
    '''
    Returns average estimate for each time
    '''
    ent_rates = np.zeros(seq.shape)

    for iter in range(N_parallel_sims): # for each parallel sequence
        s = seq[:,iter]
        ent_rates[m//2:-m//2+1,iter] = list([self_entropy_rate(s[i:i+m]) for i in range(len(s)-m+1)]) # gets all substrings of length m # go from m//2-1 since we are only going from i to i+m-1

    return np.mean(ent_rates, axis=1), np.mean(np.var(ent_rates, axis=1))

def time_variance(t, get_val):
    '''
    Function which captures trends over time. This is user defined.
    '''
    if type(t) == type(np.array([])):
        return np.array([get_val(ti) for ti in t])
    else:
        return get_val(t)


def true_entropy(time_variance,get_val, times):
    return -time_variance(times,get_val)*np.log2(time_variance(times,get_val))-(1-time_variance(times,get_val))*np.log2(1-time_variance(times,get_val))

@jit(nopython=True, fastmath=True) 
def find_lambda_constrained_start(target, source):
    """
    Finds the longest subsequence of the target array, 
    starting from index 0, that is contained in the source array.
    Returns the length of that subsequence + 1.
    
    i.e. returns the length of the shortest subsequence starting at 0 
    that has not previously appeared.
    
    Args:
        target: NumPy array, preferable of type int.
        source: NumPy array, preferable of type int.
    
    Returns:
        Integer of the length.
        
    """
    source = np.array(source)
    target = np.array(target)
    
    source_size = source.shape[0]-1
    target_size = target.shape[0]-1
    t_max = 0
    c_max = 0

    
    if source[0] == target[0]:
        c_max = 1
        for ei in range(1,min(target_size+1, source_size +1)):
            if(source[ei] != target[ei]):
                break
            else:
                c_max = c_max+1

        if c_max > t_max:
            t_max = c_max 
                
    return t_max+1


# entropy landscape

def sliding_window_grid_search(T,t_del, N_parallel_sims,get_val,N_points=50):

    res= []
    for m in tqdm(np.arange(10,T//2,T//N_points)):

        times, seq = gen_seq(T,t_del,time_variance,get_val, N_parallel_sims)
        y = true_entropy(time_variance,get_val,times)

        y_fit, v = gen_sliding_window(m,seq,N_parallel_sims)

        ss_res, ss_tot = assess_fit(y[y_fit!=0], y_fit[y_fit!=0])
        res.append([m,ss_res,ss_tot,v,sum(y_fit!=0), len(seq)])

    return pd.DataFrame(res, columns=['m','SS_res','SS_tot','variance','N','seq length']) 

def spline_grid_search(m_best,T,t_del,get_val,N_parallel_sims,s_grid = np.arange(0,10), k_grid = np.arange(1,6)): # s - smoothness, k = degree of spline fit
    x, seq = gen_seq(T,t_del,time_variance,get_val, N_parallel_sims)
    y_fit, _ = gen_sliding_window(m_best, seq, N_parallel_sims)

    true_e = true_entropy(time_variance,get_val,x)
    res_d2 = []
    for s_ in s_grid:
        for k_ in k_grid:
            tck = splrep(x, y_fit, s=s_,k=k_)
            y_sp = BSpline(*tck)(x)
            ss_res, ss_tot = assess_fit(true_e,y_sp)

            # print(f'For a number of knots {s_}, we have ss_res, {ss_res} and ss_tot, {ss_tot} with an average estimate variance of {v} and {sum(y_fit!=0)} points.')
            res_d2.append(['spline',s_,k_,ss_res,ss_tot])

    return pd.DataFrame(res_d2, columns=['type','s','k','SS_res','SS_tot'])


# entropy kernel
