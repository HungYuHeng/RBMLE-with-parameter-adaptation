
# coding: utf-8

# In[ ]:

import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import time
import pickle
import zipfile
import os
#------------------------------------------------------------------------------------------------------
theta = np.array([-0.3, 0.5, 0.8]) # the true parameters theta_*
excelID = 0 # the ID number to differentiate different true parameters on recorded tables
numActions = 10 # K in the paper, the total number of arms
isTimeVary = False # if the value is True, then the contexts are generated in a time-varying way, 
# i.e., changes from round to round in one trial, otherwise the contexts are the same in one trial

numExps = 5 # total number of trials
T = int(1e2) # T in the paper, the total number of rounds in one trial
seed = 46 # seed setup for reproducibility
path = "" # the directory to save the results
if np.linalg.norm(theta) >= 1: # this is to respect the assumption that ||theta_*|| <1 in the papers
    raise ValueError("The L2-norm of theta is bigger than 1!")
#------------------------------------------------------------------------------------------------------
methods = ["lin_ucb", "lin_bmle", "bayes_ucb", "gpucb", "gpucb_tuned", "kg", "kg_star", "lin_ts", "vids_sample"]
# the list of methdos that will be evaluated
numMethods = len(methods)
dictResults = {}
allRegrets = np.zeros((numMethods, numExps, T), dtype=float)
allRunningTimes = np.zeros((numMethods, numExps, T), dtype=float)
np.random.seed(seed) #set up the seed for reproducibility
rewardSigma = 1

def generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary):
    if isTimeVary:
        contexts = np.random.multivariate_normal(contextMus, contextSigma,(numExps, T, numActions))
    else:
        contexts = np.random.multivariate_normal(contextMus, contextSigma, (numExps, numActions))
    temp = np.linalg.norm(contexts, ord=2, axis=-1)
    contextsNorm = temp[..., np.newaxis]
    contexts = contexts / contextsNorm # this is to meet the assumption in the paper that ||x||<=1
    return contexts

def generate_rewards(theta, contexts, isTimeVary, T, rewardSigma):
    if isTimeVary:
        numExps, _, numActions, _ = contexts.shape
    else:
        numExps, numActions, _ = contexts.shape
    
    allRewards = np.zeros((numExps, T, numActions), dtype=float)
    meanRewards = np.zeros((numExps, T, numActions), dtype=float)
    tempSigma = np.eye(numActions) * rewardSigma
    for i in range(numExps):
        for j in range(T):
            tempContexts = contexts[i, j] if isTimeVary else contexts[i]
            tempMus = np.array([np.dot(theta, context) for context in tempContexts])
            meanRewards[i, j] = tempMus
            allRewards[i, j] = np.random.multivariate_normal(tempMus, tempSigma)
    return meanRewards, allRewards

contextMus = np.zeros(len(theta))
contextSigma = np.eye(len(theta)) * 10
allContexts = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
allMeanRewards, allRewards = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)
allRegrets = np.zeros((numMethods, numExps, T), dtype=float)
allRunningTimes = np.zeros((numMethods, numExps, T), dtype=float)
#------------------------------------------------------------------------------------------------------
def update_posterior(A, b, x, r):
    """
    Preparation for posteiror update in Bayes family of contextual bandit algorithms
    :A: matrix A in all algorihms' input
    :b: vector b in all algorithms' input
    Returns: updated A and updated b
    """
    A_new = np.add(A, x[:, np.newaxis] * x)
    b_new = np.add(b, r * x)
    return A_new, b_new

def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :vector: np.array
    Return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(a=indices, size=1, replace=False)[0]

def kgf(x):
    """
    :x: np.array
    Return: np.array, f(x) as defined in Ryzhov et al. (2010) 'The knowledge gradient algorithm for
    a general class of online learning problems'
    """
    return norm.cdf(x) * x + norm.pdf(x)

def lin_ucb(contexts, A, b, alpha):
    """
    :Implementation of Algorithm 1 in the paper: Contextual Bandits with Linear Payoff Functions
    :A: A in Algorithm 1 o fthe paper
    :b: b in Algorithm 1 of the paper
    :alpha: Line 7 in Algorithm 1
    Returns: the index of the arm with the largest index
    """
    A_inv = np.linalg.inv(A)
    thetaHat = np.dot(A_inv, b.T)
    indVals = np.zeros(contexts.shape[0])
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaHat, contexts[i]) + alpha * np.sqrt(np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T])) 
    arm = np.argmax(indVals)
    return arm

def lin_bmle(contexts, A, b, bias):
    """
    :Implementation of Algorithm 1 in the submitted paper
    :A: Vt in Algorithm 1
    :b: Rt in Line 112 in the paper
    :bias: alpha(t) in Algorithm 1
    Returns: the index of the arm with the largest index
    """
    A_inv = np.linalg.inv(A)
    thetaHat = np.dot(A_inv, b.T)
    indVals = np.zeros(contexts.shape[0])
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaHat, contexts[i]) + 0.5 * bias * np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T])
    arm = np.argmax(indVals)
    return arm


def lin_bucb(contexts, A, b, t) :
    """
    Reference: Section 4.3 of paper: http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf
    Prior is Gaussian prior, follows the setup in IDS paper
    :c: hyper parameter, suggested in the paper is 0
    :A: as above
    :b: as above
    :t: current round index
    Returns: the index of the arm with the largest index
    """
    A_inv = np.linalg.inv(A)
    thetaHat = np.dot(A_inv, b.T)
    indVals = np.zeros(contexts.shape[0])
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaHat, contexts[i]) + norm.ppf(q=1.0-1.0/(t), loc=0, scale=1) * np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T])
    arm = np.argmax(indVals)
    return arm

def lin_gpucb(contexts, A, b, t, delta):
    """
    Implementation of Algorithm 1 in the paper: Srinivas (2010) 'Gaussian Process Optimization in the Bandit Setting: No Regret and 
    Experimental Design' for Gaussian Bandit Problems with normal prior
    :A: as above
    :b: as above
    :t: current round index
    :delta: \delta in Theorem 1 of the paper
    Returns: the index of the arm with the largest index
    """
    A_inv = np.linalg.inv(A)
    thetaHat = np.dot(A_inv, b.T)
    beta_t = 2 * np.log(contexts.shape[0] * (t * np.pi) ** 2 / (6 * delta))
    indVals = np.zeros(contexts.shape[0])
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaHat, contexts[i]) + np.sqrt(beta_t * np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T])) 
    arm = np.argmax(indVals)
    return arm

def lin_gpucb_tuned(contexts, A, b, t, c):
    """
    Implementation of GPUCB-Tuned in the paper: Srinivas (2010) 'Gaussian Process Optimization in the Bandit Setting: No Regret and 
    :A: as above
    :b: as above
    :t: current round index
    :c: hyper parameter that needs to be tuned to be decided, in this paper it is found to be 0.9
    Returns: the index of the arm with the largest index
    """
    A_inv = np.linalg.inv(A)
    thetaHat = np.dot(A_inv, b.T)
    beta_t = c * np.log(t)
    indVals = np.zeros(contexts.shape[0])
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaHat, contexts[i]) + np.sqrt(beta_t * np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T])) 
    arm = np.argmax(indVals)
    return arm

def lin_kg(contexts, A, b, t, T):
    """
    Reference: page 184, eq (14) of "The Knowledge Gradient Algorithm for a General Class of Online Learning Problems"
    :A: as above
    :b: as above
    :t: current round index
    :T: total number of trials
    Returns: the index of the arm with the largest index
    """
    mus = np.zeros(contexts.shape[0])
    stds = np.zeros(contexts.shape[0])
    A_inv = np.linalg.inv(A)
    thetaHat = np.dot(A_inv, b.T)
    for i in range(contexts.shape[0]):
        mus[i] = np.dot(thetaHat, contexts[i])
        stds[i] = np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T])
        # reference: https://www.statlect.com/probability-distributions/normal-distribution-linear-combinations
    temp = [(mus[i], i) for i in range(len(mus))]
    temp.sort()
    cs = np.zeros(contexts.shape[0])
    for i in range(len(mus)):
        ind = temp[i][1]
        if i == len(mus)-1:
            cs[ind] = temp[i-1][0]
        else:
            cs[ind] = temp[-1][0]
    
    indVals = np.zeros(contexts.shape[0])
    for i in range(len(mus)):
        sigmaTuta = np.sqrt(stds[i]**2/(1.0 + 1.0 / stds[i]**2))
        z = -abs(float(mus[i]-cs[i])/sigmaTuta)
        indVals[i] = mus[i] + (T-t) * sigmaTuta * (norm.cdf(z) * z + norm.pdf(z))
    arm = np.argmax(indVals)
    return arm

def lin_kg_star(contexts, A, b, t, T):
    """
    Implementation of Optimized Knowledge Gradient algorithm for Bernoulli Bandit Problems with normal prior
    as described in Kaminski (2015) 'Refined knowledge-gradient policy for learning probabilities'
    Reference: https://github.com/DBaudry/Information_Directed_Sampling/blob/master/GaussianMAB.py
    :A: as above
    :b: as above
    :t: current round index
    :T: total number of trials
    Returns: the index of the arm with the largest index
    """
    sigma = 1
    mus = np.zeros(contexts.shape[0])
    stds = np.zeros(contexts.shape[0])
    A_inv = np.linalg.inv(A)
    thetaHat = np.dot(A_inv, b.T)
    for i in range(contexts.shape[0]):
        mus[i] = np.dot(thetaHat, contexts[i])
        stds[i] = np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T])
        
    nb_arms = len(mus)
    eta = sigma
    sigmas = np.array(stds)
    delta_t = np.array([mus[i] - np.max(list(mus)[:i] + list(mus)[i + 1:]) for i in range(nb_arms)])
    r = (delta_t / sigmas) ** 2
    m_lower = eta / (4 * sigmas ** 2) * (-1 + r + np.sqrt(1 + 6 * r + r ** 2))
    m_higher = eta / (4 * sigmas ** 2) * (1 + r + np.sqrt(1 + 10 * r + r ** 2))
    m_star = np.zeros(nb_arms)
    for arm in range(nb_arms):
        if T - t <= m_lower[arm]:
            m_star[arm] = T - t
        elif (delta_t[arm] == 0) or (m_higher[arm] <= 1):
            m_star[arm] = 1
        else:
            m_star[arm] = np.ceil(0.5 * ((m_lower + m_higher)[arm])).astype(int)  # approximation
    s_m = np.sqrt((m_star + 1) * sigmas ** 2 / ((eta / sigmas) ** 2 + m_star + 1))
    v_m = s_m * kgf(-np.absolute(delta_t / (s_m + 10e-9)))
    arm = np.argmax(mus - np.max(mus) + (T-t)*v_m)
    return arm

def lin_ts(contexts, A, b, v_lints):
    """
    Implementation of Algorithm 1 in the paper: Thompson Sampling for Contextual Bandits with Linear Payoffs
    :A: as above
    :b: as above
    :v_lints: v in Section 2.2, first paragraph i nthe paper
    Returns: the index of the arm with the largest index
    """
    A_inv = np.linalg.inv(A)
    Mean = np.dot(A_inv, b)
    Cov = v_lints**2 * A_inv
    thetaHat = np.random.multivariate_normal(Mean, Cov)
    indVals = np.zeros(contexts.shape[0])
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaHat, contexts[i])
    arm = np.argmax(indVals)
    return arm


def ids_action(delta, g, numQSampled):
    """
    Implementation of IDSAction algorithm (algorithm 3)  as defined in Russo & Van Roy, p. 21
    :delta: np.array, instantaneous regrets
    :g: np.array, information gains
    Return: int, arm to pull
    """
    nb_arms = len(g)
    Q = np.zeros((nb_arms, nb_arms))
    IR = np.ones((nb_arms, nb_arms)) * np.inf
    q = np.linspace(0, 1, numQSampled)
    for a in range(nb_arms - 1):
        for ap in range(a + 1, nb_arms):
            if g[a] < 1e-6 or g[ap] < 1e-6:
                return rd_argmax(-g)
            da, dap, ga, gap = delta[a], delta[ap], g[a], g[ap]
            qaap = q[rd_argmax(-(q * da + (1 - q) * dap) ** 2 / (q * ga + (1 - q) * gap))]
            IR[a, ap] = (qaap * (da - dap) + dap) ** 2 / (qaap * (ga - gap) + gap)
            Q[a, ap] = qaap
    amin = rd_argmax(-IR.reshape(nb_arms * nb_arms))
    a, ap = amin // nb_arms, amin % nb_arms
    b = np.random.binomial(1, Q[a, ap])
    arm = int(b * a + (1 - b) * ap)
    return arm

def vids_sample(contexts, mu_t, sigma_t, M, numQSampled):
    """
    Implementation of linearSampleVIR (algorithm 6 in Russo & Van Roy, p. 244) applied for Linear  Bandits with
    multivariate normal prior. Here integrals are approximated in sampling thetas according to their respective
    posterior distributions.
    :mu_t: np.array, posterior mean vector at time t
    :sigma_t: np.array, posterior covariance matrix at time t
    :M: int, number of samples
    Return: int, np.array, arm chose and p*
    """
    thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
    mu = np.mean(thetas, axis=0)
    theta_hat = np.argmax(np.dot(contexts, thetas.T), axis=0)
    theta_hat_ = [thetas[np.where(theta_hat==a)] for a in range(contexts.shape[0])]
    p_a = np.array([len(theta_hat_[a]) for a in range(contexts.shape[0])])/M
    mu_a = np.nan_to_num(np.array([np.mean([theta_hat_[a]], axis=1).squeeze() for a in range(contexts.shape[0])]))
    L_hat = np.sum(np.array([p_a[a]*np.outer(mu_a[a]-mu, mu_a[a]-mu) for a in range(contexts.shape[0])]), axis=0)
    rho_star = np.sum(np.array([p_a[a]*np.dot(contexts[a], mu_a[a]) for a in range(contexts.shape[0])]), axis=0)
    v = np.array([np.dot(np.dot(contexts[a], L_hat), contexts[a].T) for a in range(contexts.shape[0])])
    delta = np.array([rho_star - np.dot(contexts[a], mu) for a in range(contexts.shape[0])])
    arm = ids_action(delta, v, numQSampled)
        #arm = rd_argmax(-delta**2/v)
    return arm
#------------------------------------------------------------------------------------------------------
for expInd in tqdm(range(numExps)):
    # lin_ucb
    A_linucb = np.eye(len(theta))
    b_linucb = np.zeros(len(theta))
    alpha = 1
    
    # lin_bmle
    A_bmle = np.eye(len(theta))
    b_bmle = np.zeros(len(theta))
    
    # lin_bucb
    A_bucb = np.eye(len(theta))
    b_bucb = np.zeros(len(theta))
    
    # lin_gpucb
    A_gpucb = np.eye(len(theta))
    b_gpucb = np.zeros(len(theta))
    delta = 0.00001
    
    # lin_gpucbt
    A_gpucbt = np.eye(len(theta))
    b_gpucbt = np.zeros(len(theta))
    c = 0.9
    
    # kg
    A_kg = np.eye(len(theta))
    b_kg = np.zeros(len(theta))
    
    # kg_star
    A_kg_star = np.eye(len(theta))
    b_kg_star = np.zeros(len(theta))
    
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]
        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]
        maxMean = np.max(meanRewards)
    #------------------------------------------------------------------------------------------------
        mPos = methods.index("lin_ucb")
        startTime = time.time()
        arm = lin_ucb(contexts, A_linucb, b_linucb, alpha)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        A_linucb, b_linucb = update_posterior(A_linucb, b_linucb, contexts[arm], rewards[arm])
    #------------------------------------------------------------------------------------------------
        bias = np.sqrt(t)
        mPos = methods.index("lin_bmle")
        startTime = time.time()
        arm = lin_bmle(contexts, A_bmle, b_bmle, bias)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        A_bmle, b_bmle = update_posterior(A_bmle, b_bmle, contexts[arm], rewards[arm])
    #------------------------------------------------------------------------------------------------
        mPos = methods.index("bayes_ucb")
        startTime = time.time()
        arm = lin_bucb(contexts, A_bucb, b_bucb, t)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        A_bucb, b_bucb = update_posterior(A_bucb, b_bucb, contexts[arm], rewards[arm])
    #------------------------------------------------------------------------------------------------
        mPos = methods.index("gpucb")
        startTime = time.time()
        arm = lin_gpucb(contexts, A_gpucb, b_gpucb, t, delta)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        A_gpucb, b_gpucb = update_posterior(A_gpucb, b_gpucb, contexts[arm], rewards[arm])
    #------------------------------------------------------------------------------------------------
        mPos = methods.index("gpucb_tuned")
        startTime = time.time()
        arm = lin_gpucb_tuned(contexts, A_gpucbt, b_gpucbt, t, c)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        A_gpucbt, b_gpucbt = update_posterior(A_gpucbt, b_gpucbt, contexts[arm], rewards[arm])
    #------------------------------------------------------------------------------------------------
        mPos = methods.index("kg")
        startTime = time.time()
        arm = lin_kg(contexts, A_kg, b_kg, t, T)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        A_kg, b_kg = update_posterior(A_kg, b_kg, contexts[arm], rewards[arm])
    #------------------------------------------------------------------------------------------------
        mPos = methods.index("kg_star")
        startTime = time.time()
        arm = lin_kg_star(contexts, A_kg_star, b_kg_star, t, T)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        A_kg_star, b_kg_star = update_posterior(A_kg_star, b_kg_star, contexts[arm], rewards[arm])
#------------------------------------------------------------------------------------------------------
np.random.seed(seed) 
# re-setup the seed to ensure the sample path for random algorithms are the same with above deterministic ones
allContexts_forSeedReset = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
# re-generate the the contexts to ensure the sample path for random algorithms are the same with above deterministic ones
allMeanRewards_forSeedReset, allRewards_forSeedReset = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)
# re-generate the the rewards to ensure the sample path for random algorithms are the same with above deterministic ones
# all the rewards and contexts are only for fair comparison and will not be called later
for expInd in tqdm(range(numExps)):
    # ts
    A_lints = np.eye(len(theta))
    b_lints = np.zeros(len(theta))
    R_lints = rewardSigma
    delta_lints = 0.5
    epsilon_lints = 0.9
    v_lints = R_lints * np.sqrt(24 * len(theta) * np.log(1 / delta_lints))
    mPos = methods.index("lin_ts")
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]
        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]
        maxMean = np.max(meanRewards)
        
        startTime = time.time()
        arm = lin_ts(contexts, A_lints, b_lints, v_lints)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        A_lints, b_lints = update_posterior(A_lints, b_lints, contexts[arm], rewards[arm])
#------------------------------------------------------------------------------------------------------
np.random.seed(seed)
# re-setup the seed to ensure the sample path for random algorithms are the same with above deterministic ones
allContexts_forSeedReset = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
# re-generate the the contexts to ensure the sample path for random algorithms are the same with above deterministic ones
allMeanRewards_forSeedReset, allRewards_forSeedReset = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)
# re-generate the the rewards to ensure the sample path for random algorithms are the same with above deterministic ones
# all the rewards and contexts are only for fair comparison and will not be called later
for expInd in tqdm(range(numExps)):    
    #vids
    A_vids = np.eye(len(theta))
    b_vids = np.zeros(len(theta))
    mPos = methods.index("vids_sample")
    M = 10000
    numQSampled = 1000
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]
        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]
        maxMean = np.max(meanRewards)
        
        startTime = time.time()
        sigma_t = np.linalg.inv(A_vids)
        mu_t = np.dot(sigma_t, b_vids)
        arm = vids_sample(contexts, mu_t, sigma_t, M, numQSampled)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        A_vids, b_vids = update_posterior(A_vids, b_vids, contexts[arm], rewards[arm])
#------------------------------------------------------------------------------------------------------
cumRegrets = np.cumsum(allRegrets,axis=2)
meanRegrets = np.mean(cumRegrets,axis=1)
stdRegrets = np.std(cumRegrets,axis=1)
meanFinalRegret = meanRegrets[:,-1]
stdFinalRegret = stdRegrets[:,-1]
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
finalRegretQuantiles = np.quantile(cumRegrets[:,:,-1], q=quantiles, axis=1)

cumRunningTimes = np.cumsum(allRunningTimes,axis=2)
meanRunningTimes = np.mean(cumRunningTimes,axis=1)
stdRunningTimes = np.std(cumRunningTimes,axis=1)
meanTime = np.sum(allRunningTimes, axis=(1,2))/(T*numExps)
stdTime = np.std(allRunningTimes, axis=(1,2))
runningTimeQuantiles = np.quantile(cumRunningTimes[:,:,-1], q=quantiles, axis=1)

for i in range(len(methods)):
    method = methods[i]
    dictResults[method] = {}
    dictResults[method]["allRegrets"] = np.copy(allRegrets[i])
    dictResults[method]["cumRegrets"] = np.copy(cumRegrets[i])
    dictResults[method]["meanCumRegrets"] = np.copy(meanRegrets[i])
    dictResults[method]["stdCumRegrets"] = np.copy(stdRegrets[i])
    dictResults[method]["meanFinalRegret"] = np.copy(meanFinalRegret[i])
    dictResults[method]["stdFinalRegret"] = np.copy(stdFinalRegret[i])
    dictResults[method]["finalRegretQuantiles"] = np.copy(finalRegretQuantiles[:,i])
    
    
    dictResults[method]["allRunningTimes"] = np.copy(allRunningTimes[i])
    dictResults[method]["cumRunningTimes"] = np.copy(cumRunningTimes[i])
    dictResults[method]["meanCumRunningTimes"] = np.copy(meanRunningTimes[i])
    dictResults[method]["stdCumRunningTimes"] = np.copy(stdRunningTimes[i])
    dictResults[method]["meanTime"] = np.copy(meanTime[i])
    dictResults[method]["stdTime"] = np.copy(stdTime[i])
    dictResults[method]["runningTimeQuantiles"] = np.copy(runningTimeQuantiles[:,i])

FileName = 'ID=' + str(excelID) + '_linbandits_algs_seed_' + str(seed)
with open(path + FileName + '.pickle', 'wb') as handle:
    pickle.dump(dictResults, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print out the average cumulative regret all methods
# with open(path + FileName + '.pickle', 'rb') as handle:
#     dictResults = pickle.load(handle)
# for method in dictResults:
#     print (method, '--', dictResults[method]["meanFinalRegret"])
    
zipfile.ZipFile(path + FileName + '.zip', mode='w').write(path + FileName + '.pickle')

os.remove(path + FileName + '.pickle')
#------------------------------------------------------------------------------------------------------

