import yfinance
import numpy as np
from utils.generate_portfolios import generate_random_portfolios
import pandas as pd
import heapq
from utils.stat_measures import getPortfolio
from utils.plotting import plot_correlation, plot_efficient_frontier, plot_portfolio
import datetime as dt
from IPython.utils import io 

def optimalPortfolio(tickers, num_portfolios, startTime, endTime, risk_free_rate, min_allocations, max_allocations, n_averages, short):
   # Obtain Assets
   assets = []
   if endTime == 'max':
      data = yfinance.download(tickers, period = endTime)['Close']
      assets.append(data)
   else:
      for ticker in tickers:
         data = yfinance.download(ticker, startTime, endTime)['Close']
         assets.append(data)

   assets = pd.concat(assets, axis = 1, keys = tickers)
   assets = assets.fillna(method='ffill').fillna(method='bfill')
   if "CASH" not in tickers:
      assets["CASH"] = 1
      tickers.append("CASH")

   #Obtain Returns
   returns = np.log(assets / assets.shift(1)) *252
   returns = returns.dropna()
   covariance = returns.cov()/252
   
   #Obtain portfolios
   portfolios = generate_random_portfolios(returns, covariance, num_portfolios, risk_free_rate, min_allocations, max_allocations, short)
   best_portfolios = heapq.nlargest(n_averages, portfolios, key=lambda x: x[3])

   #Get the best portfolio   
   average_weights = average_weights = np.mean([portfolio[0] for portfolio in best_portfolios], axis=0)
   max_sharpe_portfolio = getPortfolio(average_weights, returns, covariance, risk_free_rate)

   return portfolios, max_sharpe_portfolio

def get_portfolio(tickers, min_allocations, max_allocations, time_horizon = 90, num_portfolios = 10000, 
                  risk_free_rate = 0.035, n_averages = 10, short = False, include_cash = False):
    
    #Deal with cash
    min_allocations.append(0)
    if include_cash:
      max_allocations.append(1)
    else: 
       max_allocations.append(0)

    #Deal with shorting
    if(short):
        min_allocations = [-1]*(len(tickers)+1)
        min_allocations[len(tickers)] = 0
        max_allocations = [1]*(len(tickers)+1)
        max_allocations[len(tickers)] = 0

    # Interval Parameters
    if time_horizon != 'max':
        endTime = dt.datetime.now()
        startTime = endTime - dt.timedelta(days = time_horizon)
    else:
        endTime = 'max'
        startTime = None

    plot_correlation(tickers, startTime, endTime)

    #Perform model
    with io.capture_output() as captured:
        portfolios, max_sharpe_portfolio = optimalPortfolio(tickers, num_portfolios, startTime, endTime, risk_free_rate, 
                                                                min_allocations, max_allocations, n_averages, short)

    # Plot efficient frontier and portfolio Statistics
    plot_efficient_frontier(portfolios, max_sharpe_portfolio)
    plot_portfolio(max_sharpe_portfolio, tickers, short)
