
# coding: utf-8

# In[ ]:

import numpy as np
from tqdm import tqdm
from scipy.optimize import fsolve
import time
import pickle
import zipfile
import os
#------------------------------------------------------------------------------------------------------
theta = np.array([0.2, -0.8, -0.5, 0.1, 0.1], dtype=float) # the true parameters theta_*
excelID = 46 # the ID number to differentiate different true parameters on recorded tables
numActions = 10 # K in the paper, the total number of arms
isTimeVary = False # if the value is True, then the contexts are generated in a time-varying way, 
C = 0.1*(excelID-40) # for lazy update
numExps = 50 # total number of trials
T = int(10) # T in the paper, the total number of rounds in one trial
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

def de_link(x):
    return -np.exp(x) / (1 + np.exp(x))**2

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
    # tempSigma = np.eye(numActions) * rewardSigma
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
        
def lin_bmle(contextsChosen, rewardsObserved, contexts, b, t, lamda, A, thetas_tau, A_tau_det, context_tau, hessian, tau):

    alpha_t = np.sqrt(t) # the \alpha_t in Algortihm 2
    eta_t = (1+np.log(t)) #\eta_t in the Algorithm 2
    indVals = np.zeros(contexts.shape[0])
    newThetasGuess = []
    change = False
    if np.linalg.det(A) > (1+C) * A_tau_det:
        change = True
        A_tau_det = np.linalg.det(A)
        for i in range(contexts.shape[0]):
            thetaTuta = fsolve(func=lin_bmle_fun, x0=thetas_tau, args=(contextsChosen, contexts[i], b, t, lamda))
            newThetasGuess.append(thetaTuta)
            L = log_likelihood(contextsChosen, rewardsObserved, thetaTuta, t)
            indVals[i] = L + eta_t * alpha_t * np.dot(thetaTuta, contexts[i]) - lamda * np.linalg.norm(thetaTuta) ** 2 / 2.0 
    else:
        for i in range(contexts.shape[0]):
            thetaTuta = thetas_tau + np.dot(np.linalg.inv(hessian),(np.dot(np.sqrt(tau),contexts[i]) - context_tau))
            L = log_likelihood(contextsChosen, rewardsObserved, thetaTuta, tau)
            indVals[i] = L + eta_t * np.sqrt(tau) * np.dot(thetaTuta, contexts[i]) - lamda * np.linalg.norm(thetaTuta) ** 2 / 2.0 

    arm = np.argmax(indVals)
    if change :
        tau = t
        thetas_tau = newThetasGuess[arm]
        context_tau = np.dot(alpha_t,contexts[arm])
    return arm, thetas_tau, A_tau_det, context_tau, change, A_tau, tau

np.random.seed(seed)
allContexts_forSeedReset = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
allMeanRewards_forSeedReset, allRewards_forSeedReset = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)

for expInd in tqdm(range(numExps)):
    # lin_bmle
    A_bmle = np.eye(len(theta))
    A_tau = np.eye(len(theta))
    A_tau_det = np.linalg.det(A_bmle)
    b_bmle = np.zeros(len(theta))
    thetas_tau = [0] * len(theta)
    context_tau = [0] * len(theta)
    hessian = np.eye(len(theta))
    tau = 0
    lamda = 1.0
    contextsChosen = []
    rewardsObserved = []
    contexts_tau = [[0] * len(theta) for _ in range(numActions)]
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]
        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]
        maxMean = np.max(meanRewards)
    
        mPos = methods.index("lin_bmle")
        startTime = time.time()
        
        arm, thetas_tau, A_tau_det, contexts_tau, change, A_tau, tau = lin_bmle(contextsChosen, rewardsObserved, contexts, b_bmle, t, lamda, A_bmle, thetas_tau, A_tau_det, context_tau, A_tau, tau)
        A_bmle = np.add(A_bmle, contexts[arm][:, np.newaxis] * contexts[arm])
        b_bmle = np.add(b_bmle, rewards[arm] * contexts[arm])
        
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        rewardsObserved.append(rewards[arm])
        contextsChosen.append(contexts[arm])
        if change:
            A_tau = A_bmle
            hessian = np.eye(len(theta))
            for i in range(len(contextsChosen)):
                hessian += de_link(np.dot(thetas_tau, contextsChosen[i]))*contextsChosen[i][:, np.newaxis] * contextsChosen[i]
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

