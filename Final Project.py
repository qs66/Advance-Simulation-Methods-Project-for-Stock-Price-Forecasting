#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:28:38 2016

@author: Qingmin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline

data = pd.read_table('price_updated.txt', sep = ' ')
sp = data.ix[:,1]
date = data.ix[:,0]

log_ret = np.log(sp) - np.log(sp.shift())
log_ret = sp.pct_change()
print(log_ret)

map_10 = pd.rolling_mean(sp, window = 10)
map_30 = pd.rolling_mean(sp, window = 30)

from matplotlib.collections import LineCollection

#fig,ax = plt.subplots(figsize=(12,5))
#plt.figure(figsize=(12,5))
#plt.plot(sp)
#plt.plot(map_10, color='black')
#xx=map_10.index
#yy=map_10
#lwidths=abs((map_10-map_30).fillna(0))*1.5
#points = np.array([xx, yy]).T.reshape(-1, 1, 2)
#segments = np.concatenate([points[:-1], points[1:]], axis=1)
#lc = LineCollection(segments, linewidths=lwidths)
#fig,a = plt.subplots()
#ax.add_collection(lc)
#ax.plot(map_10)
#ax.plot(map_30)
#ax.legend(['1day','10days', '30days'])
#plt.title('Moving Average of Price')

ret_10 = pd.rolling_mean(log_ret, window = 10)
ret_30 = pd.rolling_mean(log_ret, window = 30)
plt.figure(figsize=(12,5))
plt.plot(log_ret)
plt.plot(ret_10)
plt.plot(ret_30)
plt.legend(['1day', '10days','30days'])
plt.title('Moving Average of Log_Return')

std_10 = pd.rolling_std(log_ret, window = 10)
std_30 = pd.rolling_std(log_ret, window = 30)
std_all = np.std(log_ret)
plt.figure(figsize=(12,5))
plt.plot(std_10)
plt.plot(std_30)
plt.legend(['10days','30days'])
plt.axhline(y=std_all)
plt.title('Rolling Standard Deviation')

def gmb(s0, mu, sigma):
    s1 = s0*np.exp((mu-0.5*sigma**2) + sigma*np.random.normal())
    return s1

def generate_truncate_normal(mu, sigma, v_range):
    
    assert len(v_range) == 2, 'v_range should be a vector with length 2'
    v = np.random.normal()*sigma + mu
    while not (v_range[0] <= v <= v_range[1]):
        v = np.random.normal()*sigma + mu
        
    return v

def generate_rand_mu(p1, p2, p3, mu_range_array):
    """
    mu_range_array shoule contain the mean and range of 3 senerioes.
    senario sequence is [stay, up, down]
    """
    assert p1 + p2 + p3==1, 'Sum of probability should be 1'
    info = np.array(mu_range_array)
    assert info.shape == (3, 3), 'Shape of mu_range_array doesn\'t match the number of situations.' 
    uni = np.random.uniform()
    state = 0
    if uni <= p1:
        mu = generate_truncate_normal(mu_range_array[0,0], (mu_range_array[0,2]- mu_range_array[0,0])/2., 
                                      mu_range_array[0, 1:])
        state = 1
    elif uni < p1+p2:
        mu = generate_truncate_normal(mu_range_array[1,0], (mu_range_array[1,2]- mu_range_array[1,0])/2., 
                                      mu_range_array[1, 1:])
        state = 2
    else:
        mu = generate_truncate_normal(mu_range_array[2,0], (mu_range_array[2,2]- mu_range_array[2,0])/2., 
                                      mu_range_array[2, 1:])
        state = 3
        
    return (mu, state)

def filter(p):
    flag = False
    for i in range(len(p)-2):
        if p[i]==1:
            if (p[i+1]==1)|(p[i+2]==1):
                flag=True
    return flag

def generate_rand_sigma(lam, pd, mu_range_array):
    """
    lam is the lambda from a posson distribution, we ues poisson distribution to model possible jump of sigma
    At most jump 2 times
    pd is the prob of jumping down
    mu_range_array should be in sequence [stay, down, up]
    """
    sigma_ = np.nan
    sigma = -1
    p = [min(np.random.poisson(lam), 1) for i in range(10)]
    while (sum(p)>2)|(filter(p)):
        p = [min(np.random.poisson(lam), 1) for i in range(10)]
        
    assert sum(p) in [0, 1, 2], 'jump should not be other than 0, 1, 2 times'
    
    state = 0
    
    if sum(p) == 0:
        sigma = generate_truncate_normal(mu_range_array[0,0], (mu_range_array[0,2]- mu_range_array[0,0])/2., 
                                         mu_range_array[0, 1:])
        state = 1
    if sum(p) == 1:
        uni = np.random.uniform()
        if uni < pd:
            sigma = generate_truncate_normal(mu_range_array[1,0], (mu_range_array[1,2]- mu_range_array[1,0])/2., 
                                             mu_range_array[1, 1:])
            state = 2
        else:
            sigma = generate_truncate_normal(mu_range_array[2,0], (mu_range_array[2,2]- mu_range_array[2,0])/2., 
                                         mu_range_array[2, 1:])
            state = 3
            
    if sum(p) == 2:
        uni = np.random.uniform()
        index = -1
        for i in range(len(p)):
            if p[i] == 1:
                index = i
        if uni < pd:
            sigma = generate_truncate_normal(mu_range_array[1,0], (mu_range_array[1,2]- mu_range_array[1,0])/2., 
                                             mu_range_array[1, 1:])
            sigma_ = (generate_truncate_normal(mu_range_array[2,0]+0.003, (mu_range_array[2,2]- mu_range_array[2,0])/2., 
                                               mu_range_array[2, 1:]+0.003), index)
            state = 4
        else:
            sigma = generate_truncate_normal(mu_range_array[2,0], (mu_range_array[2,2]- mu_range_array[2,0])/2., 
                                             mu_range_array[2, 1:])
            sigma_ = (generate_truncate_normal(mu_range_array[1,0]-0.003, (mu_range_array[1,2]- mu_range_array[1,0])/2., 
                                               mu_range_array[1, 1:]-0.003), index)
            state = 5
            
    return (sigma, sigma_, state)


'''
mu_info = np.array([[0.,-0.003, 0.003],
                   [0.006, 0.003, 0.01], 
                   [-0.008, -0.012, -0.003]])

mu_prob = [0.3, 0.4, 0.3]

sigma_info = np.array([[0.0225, 0.02, 0.025], 
                       [0.0175, 0.015, 0.02], 
                       [0.028, 0.025, 0.032]])
sigma_lam = 9./80.
sigma_pd = 0.6
'''

mu_info = np.array([[0.,-0.003, 0.003],
                   [0.006, 0.003, 0.01], 
                   [-0.008, -0.012, -0.003]])

mu_prob = [0.3, 0.4, 0.3]

sigma_info = np.array([[0.0225, 0.02, 0.025], 
                       [0.0175, 0.015, 0.02], 
                       [0.028, 0.025, 0.032]])

sigma_lam = 9./80.
sigma_pd = 0.4


def generate_paths(s0, n, mu_info, mu_prob, sigma_info, sigma_lam, sigma_pd):
    
    state = np.zeros(n)
    paths = np.zeros((n, 11))
    for i in range(n):
        mu_ = generate_rand_mu(mu_prob[0], mu_prob[1], mu_prob[2], mu_info)
        mu = mu_[0]
        
        sigma_ = generate_rand_sigma(sigma_lam, sigma_pd, sigma_info)
        sigma0 = sigma_[0]
        assert sigma0 != -1, 'sigma should not be -1'
        sigma1_ = sigma_[1]
        
        state[i] = 10*mu_[1]+sigma_[2]
        
        if sigma1_ is np.nan:
            paths[i, 0] = s0
            for j in range(10):
                paths[i, j+1] = gmb(paths[i, j], mu, sigma0)
        
        if sigma1_ is not np.nan:
            sigma1 = sigma1_[0]
            sigma1_idx = sigma1_[1]
            paths[i, 0] = s0
            for j in range(10):
                if j < sigma1_idx:
                    paths[i, j+1] = gmb(paths[i, j], mu, sigma0)
                else:
                    paths[i, j+1] = gmb(paths[i, j], mu, sigma1)
                    
        if i%1000 == 0:
            total = np.ceil(n/1000)
            print('{k}th out of {t}'.format(k=np.ceil(i/1000)+1, t=total))
                
    return (paths, state)

print(sp[len(sp)-1])
paths_ = generate_paths(38.34, int(2e5), mu_info, mu_prob, sigma_info, sigma_lam, sigma_pd)

paths = paths_[0]
path_states = paths_[1]

random_index = np.random.choice(100000, 100, replace = False)
'''
plt.figure(figsize=(12,5))
for i in random_index:
    if np.floor(path_states[i]/10) == 1:
        plt.plot(paths[i,:])
    elif np.floor(path_states[i]/10) == 2:
        plt.plot(paths[i,:])
    else:
        plt.plot(paths[i,:])
        
plt.title('Paths of Different Drift Scenario')

def get_quantile(path, q1, q2):
    path = np.array(path)
    ci_l = np.percentile(path, q1, axis=0)
    ci_u = np.percentile(path, q2, axis=0)
    return (ci_l, ci_u)
total_ci = get_quantile(paths, 10, 90)
def get_group_mu(paths_, group):
    index = []
    for i in range(len(paths_[0])):
        if np.floor(path_states[i]/10.) == group:
            index.append(i)
    path_gp = paths_[0][index,:]
    return np.array(path_gp)

gp_green = get_group_mu(paths_, 2)
gp_blue = get_group_mu(paths_, 1)
gp_red = get_group_mu(paths_, 3)
green_line = get_quantile(gp_green, 10, 90)
blue_line = get_quantile(gp_blue, 10, 90)
red_line = get_quantile(gp_red, 10, 90)
plt.figure(figsize=(12,5))
for i in random_index:
    if path_states[i]%10 == 1:
        plt.plot(paths[i,:])
    elif path_states[i]%10 == 2:
        plt.plot(paths[i,:])
    elif path_states[i]%10 == 3:
        plt.plot(paths[i,:])
    elif path_states[i]%10 == 4:
        plt.plot(paths[i,:])
    else:
        plt.plot(paths[i,:])
        
plt.title('Paths of Different Volatility Scenario')
'''

fig, ax = plt.subplots(figsize=(12,5))
plt.plot(green_line[0])
plt.plot(green_line[1])
plt.plot(blue_line[0])
plt.plot(blue_line[1])
plt.plot(red_line[0])
plt.plot(red_line[1])
x = [i for i in range(11)]
ax.fill_between(x, blue_line[0], blue_line[1], interpolate=False)
ax.fill_between(x, green_line[0], green_line[1], color="none", hatch="/////")
ax.fill_between(x, red_line[0], red_line[1], color="none", hatch="\\\\\\\\\\")
plt.plot(total_ci[0])
plt.plot(total_ci[1])
plt.title("80% CI of Different Drifts")

print(total_ci)

