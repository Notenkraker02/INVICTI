from utils.stat_measures import exp_return, st_dev, sharpe_ratio
import numpy as np

# Generate portfolios with random weights
def generate_random_portfolios(returns, covariance, num_portfolios, risk_free_rate, min_allocations, max_allocations, short):
    num_assets = returns.shape[1]
    portfolios = []

    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        if(short):
            weights = weights * num_assets - 1

        # Ensure that specified assets have weights greater than or equal to the specified minimum allocations
        # And have weights less than or equal to the specified maximum allocations
        iteration = 0
        while iteration < 100:
            updated = False
            sumFree = 0
            difference = 0
            temp_min_allocations = min_allocations.copy()
            temp_max_allocations = max_allocations.copy()
            for i in range(num_assets):
                initialWeight = weights[i]
                upper_bound = min(weights[i], max_allocations[i])
                weights[i] = max(min_allocations[i], upper_bound)
                difference += weights[i] - initialWeight
                if (min_allocations[i] == 0 and max_allocations[i] == 1) or weights[i] == initialWeight:
                    temp_min_allocations[i] = 0
                    temp_max_allocations[i] = 1
                    sumFree += weights[i]

            if difference != 0:
                updated = True
                iteration += 1

            if not updated:
                break

            for i in range(num_assets):
                if temp_min_allocations[i] == 0 and temp_max_allocations[i] == 1:
                    weights[i] *= (sumFree-difference)/sumFree

        returns_portfolio = exp_return(weights, returns)
        volatility_portfolio = st_dev(weights, covariance)
        sharpe_ratio_portfolio = sharpe_ratio(weights, returns, covariance, risk_free_rate)
        if(sharpe_ratio_portfolio < 100 and sharpe_ratio_portfolio > -100):
           portfolios.append((weights, returns_portfolio, volatility_portfolio, sharpe_ratio_portfolio))

    return portfolios