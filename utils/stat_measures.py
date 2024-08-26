import numpy as np


#Standard Deviation
def st_dev(weights, cov_matrix):
    variance = np.dot(weights.transpose(), np.dot(cov_matrix, weights))
    return np.sqrt(variance)

#Expected Return given weights
def exp_return(weights, returns):
    return np.sum(returns.mean() * weights)

#Sharpe Ratio
def sharpe_ratio(weights, returns, cov_matrix, rf_rate):
    return (exp_return(weights, returns) - rf_rate)/st_dev(weights, cov_matrix)

def getPortfolio(weights, returns, cov_matrix, rf_rate):
    return weights, exp_return(weights, returns), st_dev(weights, cov_matrix), sharpe_ratio(weights, returns, cov_matrix, rf_rate)