
# coding: utf-8

# In[ ]:

import numpy as np
from tqdm import tqdm
from scipy.optimize import fsolve, minimize
from scipy.stats import norm
import time
import pickle
import zipfile
import os
#------------------------------------------------------------------------------------------------------
theta = np.array([0.3, -0.5, 0.2, -0.7,-0.1]) # the true parameters theta_*
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
methods = ["ucb_glm", "laplace_ts", "lin_bmle"]
# the list of methdos that will be evaluated
numMethods = len(methods)
dictResults = {}
allRegrets = np.zeros((numMethods, numExps, T), dtype=float)
allRunningTimes = np.zeros((numMethods, numExps, T), dtype=float)
np.random.seed(seed)
rewardSigma = 1
#------------------------------------------------------------------------------------------------------
def link(x):
    return 1.0 / (1.0 + np.exp(-x))
def generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary):
    if isTimeVary:
        contexts = np.random.multivariate_normal(contextMus, contextSigma,(numExps, T, numActions))
    else:
        contexts = np.random.multivariate_normal(contextMus, contextSigma, (numExps, numActions))
    temp = np.linalg.norm(contexts, ord=2, axis=-1)
    contextsNorm = temp[..., np.newaxis]
    contexts = contexts / contextsNorm
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
            tempMus = np.array([link(np.dot(theta, context)) for context in tempContexts])
            meanRewards[i, j] = tempMus
            allRewards[i, j] = np.random.binomial(1, tempMus)
    return meanRewards, allRewards

contextMus = np.zeros(len(theta))
contextSigma = np.eye(len(theta)) * 10
allContexts = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
allMeanRewards, allRewards = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)
allRegrets = np.zeros((numMethods, numExps, T), dtype=float)
allRunningTimes = np.zeros((numMethods, numExps, T), dtype=float)
#------------------------------------------------------------------------------------------------------
def ucb_glm_fun(x, contextsChosen, rewardsObserved, t):
    """
    Equation (6) in the paper: Provably Optimal Algorithms for Generalized Linear Contextual Bandits 
    :contextsChosen: the contexts of historically chosen arms
    :rewardsObserved: the observed rewards of historically chosen arms
    :t: current round index
    Returns: a function
    """
    rst = [0] 
    for i in range(t-1):
        rst += (rewardsObserved[i] - link(np.dot(x, contextsChosen[i]))) * contextsChosen[i]
    return rst
def ucb_glm(contextsChosen, rewardsObserved, contexts, A, t, tau, alpha):
    """
    Implementation of the Algorithm 1 in the paper: Provably Optimal Algorithms for Generalized Linear Contextual Bandits 
    :contextsChosen: the contexts of historically chosen arms
    :rewardsObserved: the observed rewards of historically chosen arms
    :contexts: the contexts of all arms in current around
    :A: the V_t in 
    :t: current round index
    :tau: the \tau in Algorithm 1
    :alpha: the \alpha in Algorithm 1
    Returns: the index of the arm with the largest index
    """
    if t <= tau:
        return int(np.random.choice(contexts.shape[0], 1))
    indVals = np.zeros(contexts.shape[0])
    thetaTuta = fsolve(func=ucb_glm_fun, x0=[0]*contexts.shape[1], args=(contextsChosen, rewardsObserved, t))
    A_inv = np.linalg.inv(A)
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaTuta, contexts[i]) + alpha * np.sqrt(np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T]))
    return np.argmax(indVals)

for expInd in tqdm(range(numExps)):
    # ucb_glm
    tau = numActions
    kappa = 0.1  # for logistic function
    delta = 0.1
    alpha = rewardSigma/kappa*np.sqrt(len(theta)/2*np.log(1+2*T/len(theta))+np.log(1/delta))
    A_ucb_glm = np.eye(len(theta))
    contextsChosen = []
    rewardsObserved = []
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]
        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]
        maxMean = np.max(meanRewards)

        mPos = methods.index("ucb_glm")
        startTime = time.time()
        arm = ucb_glm(contextsChosen, rewardsObserved, contexts, A_ucb_glm, t, tau, alpha)
        A_ucb_glm = np.add(A_ucb_glm, contexts[arm][:, np.newaxis] * contexts[arm])
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        rewardsObserved.append(rewards[arm])
        contextsChosen.append(contexts[arm])
#------------------------------------------------------------------------------------------------------
def laplace_ts_fun(x, ms, sigmas, rewardsObserved, contextsChosen):
    """
    The objective function to be minimized to find w in Algorithm 3 of the paper: 
    An Empirical Evaluation of Thompson Sampling
    Returns: a function
    """
    qs = np.array([1.0 / sigma for sigma in sigmas])
    rst = 0.5 * np.linalg.norm(qs * (x - ms)) ** 2 
    for i in range(len(contextsChosen)):
        rst += np.log(1.0 + np.exp(np.float64(-rewardsObserved[i] * np.dot(x, contextsChosen[i]))))
    return rst

def update_posterior(mus, sigmas, rewardsObserved, contextsChosen):
    """
    Update posterior based observed rewards and the context of pulled arm: the last two lines of Algorithm 3
    Returns: the updated mean and variance of the weights
    """
    if contextsChosen == []:
        return mus, sigmas
    contextsChosen = np.array(contextsChosen)
    res = minimize(fun=laplace_ts_fun, x0=[0]*len(mus), args=(mus, sigmas, rewardsObserved, contextsChosen))
    ws = res.x
    ps = np.array([np.reciprocal(1.0 + link(np.dot(-ws, context))) for context in contextsChosen])
    qs = np.array([1.0/sigmas[i] + np.sum(contextsChosen[:, i]**2 * ps * (1.0-ps)) for i in range(len(mus))])
    return ws, np.array([1.0 / q for q in qs])

def laplace_ts(contexts, mus, sigmas):
    """
    Implementation of the Algorithm 3 in the paper: An Empirical Evaluation of Thompson Sampling
    :mus[i]: posterior means w_i in the paper
    :sigmas[i]: posterior variance of w_i in the paper
    Returns: the index of the arm with the largest index
    """
    indVals = np.zeros(contexts.shape[0])
    covMat = np.diag(sigmas)
    for i in range(contexts.shape[0]):
        weights = np.random.multivariate_normal(mus, covMat)
        indVals[i] = link(np.dot(weights, contexts[i]))
    return np.argmax(indVals)

np.random.seed(seed)
# re-setup the seed to ensure the sample path for random algorithms are the same with above deterministic ones
allContexts_forSeedReset = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
# re-generate the the contexts to ensure the sample path for random algorithms are the same with above deterministic ones
allMeanRewards_forSeedReset, allRewards_forSeedReset = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)
# re-generate the the rewards to ensure the sample path for random algorithms are the same with above deterministic ones
# all the rewards and contexts are only for fair comparison and will not be called later
for expInd in tqdm(range(numExps)):
    # laplace_ts
    lamda_ts = 1.0
    mus = [0] * len(theta)
    sigmas = [1.0 / lamda_ts] * len(theta)
    contextsChosen = []
    rewardsObserved = []
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]
        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]
        maxMean = np.max(meanRewards)

        mPos = methods.index("laplace_ts")
        startTime = time.time()
        arm = laplace_ts(contexts, mus, sigmas)
        mus, sigmas = update_posterior(mus, sigmas, rewardsObserved, contextsChosen)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        rewardsObserved.append(rewards[arm])
        contextsChosen.append(contexts[arm])
#------------------------------------------------------------------------------------------------------
def lin_bmle_fun(x, contextsChosen, curtContext, b, t, lamda):
    """
    Equation (15) of Algorithm 2 in the submitted paper
    Returns: a function
    """
    alpha_t = np.sqrt(t)
    rst = b - lamda * x + alpha_t * curtContext 
    for i in range(t-1):
        rst -= link(np.dot(x, contextsChosen[i])) * contextsChosen[i]
    return rst

def log_likelihood(contextsChosen, rewardsObserved, thetaTuta, t):
    """
    The \ell(F_t, \theta) in Line 140 of the submitted paper
    Returns: a scalar
    """
    L = -(t-1)*np.log(np.pi)/2.0
    for i in range(t-1):
        L -= 0.5*(np.dot(thetaTuta, contextsChosen[i]) - rewardsObserved[i])**2
    return L
        
def lin_bmle(contextsChosen, rewardsObserved, contexts, b, t, lamda, thetasGuess):
    """
    Implementation of Algorithm 2 of the submitted paper
    :contextsChosen: the contexts of historically chosen arms
    :rewardsObserved: the observed rewards of historically chosen arms
    :contexts: the contexts of all arms in current around
    :b: 
    :t: current round index
    :lamda: the \lambda in Algorithm 2 of the submitted paper
    :thetasGuess: the initial guess of the \theta_*, it can be arbitrary guess since the obejctive function is convex
    Returns: the index of the arm with the largest index
    """
    
    alpha_t = np.sqrt(t) # the \alpha_t in Algortihm 2
    eta_t = (1+np.log(t)) #\eta_t in the Algorithm 2
    indVals = np.zeros(contexts.shape[0])
    newThetasGuess = []
    for i in range(contexts.shape[0]):
        thetaTuta = fsolve(func=lin_bmle_fun, x0=thetasGuess[i], args=(contextsChosen, contexts[i], b, t, lamda))
        newThetasGuess.append(thetaTuta)
        L = log_likelihood(contextsChosen, rewardsObserved, thetaTuta, t)
        indVals[i] = L + eta_t * alpha_t * np.dot(thetaTuta, contexts[i]) - lamda * np.linalg.norm(thetaTuta) ** 2 / 2.0 
    return np.argmax(indVals), newThetasGuess

np.random.seed(seed)
# re-setup the seed to ensure the sample path for random algorithms are the same with above deterministic ones
allContexts_forSeedReset = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
# re-generate the the contexts to ensure the sample path for random algorithms are the same with above deterministic ones
allMeanRewards_forSeedReset, allRewards_forSeedReset = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)
# re-generate the the rewards to ensure the sample path for random algorithms are the same with above deterministic ones
# all the rewards and contexts are only for fair comparison and will not be called later
for expInd in tqdm(range(numExps)):
    # lin_bmle
    b_bmle = np.zeros(len(theta))
    thetasGuess = [[0] * len(theta) for _ in range(numActions)]
    lamda = 1.0
    contextsChosen = []
    rewardsObserved = []
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]
        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]
        maxMean = np.max(meanRewards)
    
        mPos = methods.index("lin_bmle")
        startTime = time.time()
        arm, thetasGuess = lin_bmle(contextsChosen, rewardsObserved, contexts, b_bmle, t, lamda, thetasGuess)
        b_bmle = np.add(b_bmle, rewards[arm] * contexts[arm])
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        rewardsObserved.append(rewards[arm])
        contextsChosen.append(contexts[arm])
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

FileName = 'ID=' + str(excelID) + '_general_linbandits_algs_seed_' + str(seed)
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

