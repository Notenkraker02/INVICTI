import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

import matplotlib.pyplot as plt
import pandas as pd
from IPython.utils import io
import seaborn
import yfinance


# Plot the Efficient Frontier
def plot_efficient_frontier(portfolios, max_sharpe_portfolio):
    returns = [portfolio[1] for portfolio in portfolios]
    volatilities = [portfolio[2] for portfolio in portfolios]
    sharpe_ratios = [portfolio[3] for portfolio in portfolios]


    # Customizing color mapping to reflect full range of Sharpe ratios
    cmap = plt.get_cmap('viridis')
    colors = [cmap(sharpe_ratio) for sharpe_ratio in sharpe_ratios]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap=cmap, marker='o')
    plt.scatter(max_sharpe_portfolio[2], max_sharpe_portfolio[1], color='red', marker='*', s=200, label='Max Sharpe Ratio Portfolio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')

    # Create color bar
    cbar = plt.colorbar(scatter, label='Sharpe Ratio')
    cbar.set_label('Sharpe Ratio')

    plt.legend()
    plt.grid(True)
    plt.show()

def plot_portfolio(portfolio, tickers, short):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    weights, returns, volatility, sharpe_ratio = portfolio

    # Plot pie chart for asset allocation with customized style
    if(not short):
        ax[0].pie(weights, labels=tickers, autopct='%1.2f%%', startangle=140)
        ax[0].set_title('Optimal Portfolio Weights')

    # Create a DataFrame for asset allocation
    portfolio_df = pd.DataFrame({'Asset': tickers, 'Weight': weights})
    portfolio_df['Weight'] = portfolio_df['Weight'].map('{:.2%}'.format)

    # Print portfolio statistics
    ax[1].text(0, 1, "Portfolio Overview:", fontsize=14, fontweight='bold', transform=ax[1].transAxes)
    ax[1].text(0, 0.9, f"Expected Annual Return: {returns:.2%}", fontsize=12, transform=ax[1].transAxes)
    ax[1].text(0, 0.8, f"Volatility: {volatility:.2%}", fontsize=12, transform=ax[1].transAxes)
    ax[1].text(0, 0.7, f"Sharpe Ratio: {sharpe_ratio:.2f}", fontsize=12, transform=ax[1].transAxes)
    ax[1].text(0, 0.6, "Asset Allocation:", fontsize=12, fontweight='bold', transform=ax[1].transAxes)
    for i, (asset, weight) in enumerate(zip(tickers, weights)):
        ax[1].text(0, 0.55 - 0.05 * i, f"{asset}: {weight:.2%}", fontsize=10, transform=ax[1].transAxes)
    ax[1].axis('off')

    plt.show()

def plot_correlation(tickers, startTime, endTime):
    with io.capture_output() as captured:
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

    cor_matrix = assets.corr()
    plt.figure(figsize=(10, 8))
    seaborn.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt=".4f")
    plt.title('Covariance Matrix Heatmap')
    plt.xlabel('Assets')
    plt.ylabel('Assets')
    plt.show()
