"""
This program compares the results of the numerical approximation of Black-Scholes Model using the method of lines and RK4 to the analytical BSM solution.
It includes the following functions:
    1. read_options_data: reads option data from a txt file to collect values for parameters needed.
    2. compute_bs_call_price: computes the analytical Black-Scholes price for a European call option at the current stock price S0.
    3. exact_bs_call_price_on_grid: computes the exact Black-Scholes price on a grid to compare to the numerical solution.
    4. MOL_RK4_bs_call: computes the numerical price using the MOL and RK4 method.
    5. select_representative_strikes: selects 5 option options contracts based on strike.
    6. print_price_table: prints a table comparing the four numerical prices from grid sizes, analytical prices, market price, and their errors.
    7. plot_price_comparison: Plots numerical vs analytical vs market prices for a given grid size. (4 graphs)
    8. plot_error: plots the error between numerical and analytical prices and numerical and market prices for each option contract. (4 graphs)
    9. plot_convergence: plots the convergence of the numerical RK4 solution at S0 for each option contract the spatial grid size M increases. (5 graphs)

The main function reads parameters from a txt file, computes analytical and numerical prices for 4 grid sizes, and generates plots and table for analysis.
It uses the following libraries:
    - numpy for numerical computations
    - pandas for data handling
    - scipy.stats for it's norm function
    - matplotlib for plotting graphs
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

import os
import zipfile

def read_options_data(filepath):
    """
    Reads options chain data from a txt file and converts the needed data into a list of usable variables for the black scholes model.
    inputs: 
        filepath: path to the txt file containing options data
    outputs:
        options: list containing option parameters for each option contract (S0 - current stock price, K - strike price, T - time to maturity, r - risk-free rate, sigma - volatility, market_price - market price of the option)
    """
    df = pd.read_csv(filepath, low_memory=False) # read the txt file into a pandas DataFrame

    df.columns = df.columns.str.strip().str.upper() # remove leading/trailing whitespace from column names
    df['[QUOTE_DATE]'] = pd.to_datetime(df['[QUOTE_DATE]'], errors='coerce') # convert the date column to datetime format
    df = df[df['[QUOTE_DATE]'].dt.year == 2022]  # only filter for the year 2022 -- to match the risk-free rate assumption

    columns = ['[UNDERLYING_LAST]', '[STRIKE]', '[DTE]', '[C_IV]', '[C_LAST]'] # specify the columns needed from txt file (from left to right: current stockprice, strike price, days to expiry, implied volatility, actual market price of option)
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convert columns to numeric, setting errors to NaN
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in txt file.") # check if the column actually exists in the file
    
    df = df[
        (df['[C_LAST]'] > 0) &
        (df['[C_IV]'] > 0) &
        (df['[DTE]'] > 0)
    ]  # filter out rows that have non-positive market prices, implied volatilities, or days to expiry

    options = [] # list to hold options parameters

    for _, row in df.iterrows():
        S0 = row['[UNDERLYING_LAST]'] # current stock price
        K = row['[STRIKE]'] # the price you want to buy/sell the asset
        T = row['[DTE]'] / 252  # convert days to expiry to years (assuming 252 trading days in a year)
        r = 0.02  # assume a constant risk-free rate of 2% based on 2022 3-month risk-free rates (corresponding to options from 2022)
        sigma = row['[C_IV]'] # implied volatility for the option
        market_price = row['[C_LAST]'] # market price of the call option

        option = {
            'S0': S0,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'market_price': market_price
        }
        options.append(option)

    return options

def compute_bs_call_price_exact(S0, K, T, r, sigma):
    """
    Computes the exact Black-Scholes price for a European call option.
    inputs:
        S0: current stock price
        K: strike price
        T: time to maturity (in years)
        r: risk-free interest rate ~ using 3 month treasury yield from 2022 (time of data from csv)
        sigma: implied volatility of the stock
    outputs:
        call_price (float): the theoretical exact Black-Scholes fair price of the European call option
    """
    if T == 0:
        return max(0, S0 - K)  # option has expired, return original value
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def exact_bs_call_price_on_grid(S, K, T, r, sigma):
    """
    Computes the exact Black-Scholes price on a grid of stock prices S.
    inputs:
        S: array of stock prices
        K: strike price
        T: time to maturity (in years)
        r: risk-free interest rate ~ using 3 month treasury yield from 2022 (time of data from csv)
        sigma: implied volatility of the stock
    outputs:
        call_prices: array of theoretical exact Black-Scholes fair prices of the European call option corresponding to each stock price in S
    """
    S = np.array(S, dtype=float)
    S_safe = np.maximum(S, 1e-10)  # avoid log(0) by making sure S is not zero, lower bound is close to zero

    if T == 0:
            return max(S - K, 0)   # option has expired, return original value
    
    # calculate d1 and d2
    # d1 the option's delta: expected benefit from acquiring the stock outright, weighted by a risk-adjusted probability that the option will be exercised
    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) 
    # d2 the risk-adjusted probability that the option will be exercised at expiration (P(S > K)), and is used t calculate the present value of the exercise payment (K * exp(-rT) * N(d2))
    d2 = d1 - sigma * np.sqrt(T) # risk-adjusted profitability measure

    # calculate the call option price
    call_price = S_safe * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def MOL_RK4_bs_call(S_max, K, T, r, sigma, grid_size, N):
    """
    Computes the numerical price of a European call option using the Method of Lines (MOL) and RK4 time-stepping.
    The method of lines discretizes space (of stock price S) using finite differences and keeps time continuous, making the PDE into a system of ODEs.
    It is then solved by the Runge Kutta 4 ODE solver.
    inputs:
        S_max: maximum stock price for spatial grid ~ uses the S0*3 for pratical boundary condition for this numerical method
        K: strike price 
        T: time to maturity (in years)
        r: risk-free interest rate
        sigma: implied volatility of the stock
        grid_size: number of spatial intervals (stock price discretization)
        N: number of time steps (time discretization)
    outputs:
        stock_price_grid: stock price grid to plot the numerical solution
        call_price: numerical price of the European call option at stock price S0
    """
    # method of lines
    dS = float(S_max) / grid_size # the max stock price divided by the number of spatial intervals
    stock_price_grid = np.linspace(0, float(S_max), grid_size + 1, dtype=float) # stock price grid from 0 to S_max with M+1 points
    # time step using interval T and N steps
    dt = float(T) / N

    # terminal condition (t = T)
    call_option = np.maximum(stock_price_grid - K, 0).astype(float)  # call option payoff at expiration

    # boundary conditions functions ( which should be C(0, t) = 0, C(S_max, t) = S_max - K * exp(-r(T-t)))
    def BC_left(t):
        return 0.0  # C(0, t) = 0 for a call option
    
    def BC_right(t):
        # time remaining to maturity is (T - t)
        return S_max - K * np.exp(-r * (T - t))  # C(S_max, t)
    
    # the function that computes dC/dt = F(C, t) where C is the option price evaluated at stock prices for the RK4 solver.
    # each spatial grid point gives one ODE in time
    def F(call_option, t):
        Cb = call_option.copy().astype(float) # creates a safe copy the solution to ensure it is not overwiting the option price
        # apply the boundary conditions on the copy
        Cb[0] = BC_left(t) # left boundary S = 0, C = 0
        Cb[-1] = BC_right(t) # right boundary S = Smax, C = Smax - Ke^-rt

        dCdt = np.zeros_like(Cb, dtype=float) # initializes the time derivative array, each entry holds the time derivative at one spatia grid point

        # loop over the spatial points between the boundary conditions: i = 1 ... M-1
        # makes one ODE per spatial point
        for i in range(1, grid_size):
            dC_dS  = (Cb[i+1] - Cb[i-1]) / (2 * dS) # first spatial derivative the central difference
            d2C_dS2 = (Cb[i+1] - 2*Cb[i] + Cb[i-1]) / (dS**2) # second spatial derivative using central difference, curvature of the option price to the stock price 

            # the black-scholes PDE transformed into ODE form:
            dCdt[i] = (
                0.5*sigma**2*stock_price_grid[i]**2*d2C_dS2 # 1/2sigma^2S^2del^2C/delS^2
            + r*stock_price_grid[i]*dC_dS # + rSdelC/del^S
            - r*Cb[i] # - rC
            ) 
        
        return dCdt
    
    # RK4 time-stepping backward in time t = T to t = 0
    t = float(T)
    # time-stepping loop for N steps
    for n in range(N):
        k1 = F(call_option, t) # evaluate F at current time and option price to get the current option rate of change
        k2 = F(call_option + 0.5 * dt * k1, t - 0.5 * dt) # evaluate F at midpoint using k1, half time step
        k3 = F(call_option + 0.5 * dt * k2, t - 0.5 * dt) # evaluate F at midpoint using k2, half time step -- refines previous point
        k4 = F(call_option + dt * k3, t - dt) # evaluate F at next time level using k3, slope at end of the time step

        # update call_option backward in time
        call_option += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4) # RK4 takes weighted average of slopes
        t -= dt # decrement time

        # boundary conditions at new time level
        call_option[0] = BC_left(t) # left boundary condition
        call_option[-1] = BC_right(t) # right boundary condition
        
    return stock_price_grid, call_option # return to main function as stock_price_grid and call_option_numerical

def select_representative_strikes(options, S0_main, num_strikes = 5):
    """
    Selects a representative set of strikes within 50% of the current market price (S0_main).
    This set covers deep ITM, ITM, ATM, OTM, deep OTM within a reasonable range where the option prices are well-defined (i.e. not too close to zero or infinity).
    inputs:
        options: list of option dictionaries, to pull strike price 'K' and stock price'S0'
        S0_main: reference stock price to determine ATM
        num_strikes: number of strikes to select
    outputs:
        subset_options: list of selected option dictionaries
    """
    K_min = S0_main * (1 - 0.5) # filter strikes to be within 50% of S0_main
    K_max = S0_main * (1 + 0.5) # filter strikes to be within 50% of S0_main
    options_filtered = [opt for opt in options if K_min <= opt['K'] <= K_max] # filter options within the strike range
    if len(options_filtered) < num_strikes:
        print("Not enough strikes in filtered range will be using all available strikes") # warn if not enough strikes in the filtered range
        options_filtered = options # use all available strikes if not enough in filtered range

    options_sorted = sorted(options_filtered, key=lambda opt: opt['K']) # sort by strike price from the filtered options
    n = len(options_sorted) # number of available options after filtering

    indices = np.linspace(0, n-1, num_strikes, dtype=int) # pick evenly spaced strikes to account for large range of contracts
    subset_options = [options_sorted[i] for i in indices] # select the options at those indices

    subset_options = sorted(subset_options, key=lambda opt: abs(opt['K'] - S0_main)) # sort by distance to S0_main to make sure the ATM option contract is in the middle, and the others are ordered around it
    return subset_options

def print_price_table(subset_options, results, M, N):
    """
    Prints a table of the numerical (for each grid size), analytical, and market prices for each of the of 5 options contracts.
    Also includes the absolute error between the numerical and market, and analytical and numerical.
    inputs:
        subset_options: list of option dictionaries (the selected 5 strikes)
        results: dictionary with the numerical and analytical computations
        M: list of spatial grid sizes corressponding to numerical soltuions
        N: list of time steps corresponding to numerical solutions
    outputs:
        printed table of prices and errors in terminal
    """
    # column widths
    w_K = 10 # width for strike K
    w_S0 = 12 # width for S0
    w_market = 14 # width for market price
    w_ana = 18 # width for analytical price
    w_num = 35 # width for numerical prices and errors

    # column headers
    header = ["Strike K", "Current S0", "Market Price", "Analytical"]
    for i in range(len(M)):
        header.append(f"Num Price (M={M[i]}, N={N[i]})")
    header.append("Error Num-Ana")
    header.append("Error Num-Market")

    print(f"\n{header[0]:<{w_K}}{header[1]:<{w_S0}}{header[2]:<{w_market}}{header[3]:<{w_ana}}", end="") # formatting first 4 columns

    # column headers for numerical prices for each grid size
    for h in header[4:]:
        print(f"{h:<{w_num}}", end="")  # formatting
    print()

    # rows for each option contract
    for opt_id, opt in enumerate(subset_options):

        K = float(opt['K']) # strike price
        S0 = float(opt['S0']) # current stock price
        market = float(opt['market_price']) # market price
        ana_price = float(results[opt_id]['results_by_M'][M[0]]['C_an_S0']) # analytical price at S0 (same for all grid sizes)
        
        print(f"{K:<{w_K}.2f}{S0:<{w_S0}.2f}{market:<{w_market}.2f}{ana_price:<{w_ana}.3f}", end="") # formatting

        # numerical prices at S0 for each grid size
        for grid_size in M:
            num_price = float(results[opt_id]['results_by_M'][grid_size]['C_num_S0'])
            print(f"{num_price:<{w_num}.3f}", end="")

        # errors at finest grid M = 200
        finest_M = M[-1]
        err_ana = float(results[opt_id]['results_by_M'][finest_M]['error_num_ana_S0']) # error between numerical and analytical at S0
        err_market = float(results[opt_id]['results_by_M'][finest_M]['error_num_market']) # error between numerical and market at S0

        print(f"{err_ana:<{w_num}.2e}{err_market:<{w_num}.2e}") # formatting

def plot_price_comparision(subset_options, results, M):
    """
    Plots the options price comparision for numerical, analytical and market price for each of the 5 options contracts on a single plot based on grid size.
    inputs:
        subset_options: list of 5 options dictionaries
        results: results dictionary from the numerical and analyitcal computations
        M: the array of grid sizes
    outputs:
        saved graph of numerical vs analytical vs market prices for all 5 options contracts (one graph per grid size)
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(subset_options))) # distinct colors for each options contract

    # for each grid size M, plot the numerical, analytical and market prices for all 5 options contracts
    for grid_size in M: 
        plt.figure(figsize=(12, 7))
        for opt_id, opt in enumerate(subset_options):
            color = colors[opt_id]
            S = results[opt_id]['results_by_M'][grid_size]['S']
            price_RK4 = results[opt_id]['results_by_M'][grid_size]['C_numerical']
            price_BS = results[opt_id]['results_by_M'][grid_size]['C_analytical']
            S0 = opt['S0']
            market_price = opt['market_price']

            # plot numerical solution
            plt.plot(S, price_RK4, '--', color=color, label=f"RK4 (Strike K = {opt['K']:.2f})")
            # plot analytical solution
            plt.plot(S, price_BS, '-', color=color, label=f"BS (Strike K = {opt['K']:.2f})")
            # plot market_price point
            plt.scatter(S0, market_price, color=color, edgecolor="black", marker='o', s=60, zorder=5)
  
        plt.xlabel("Stock Price - S")
        plt.ylabel("Call Option Price - C")
        plt.title(f"Price Comparison: Numerical RK4 vs Analytical BS vs Market (M = {grid_size})")
        plt.grid(True)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"price_comparison_M{grid_size}.jpeg") # save the figure

def plot_error(subset_options, results, M):
    """
    Plots the error in the numerical options price in comparison to the analytical price based on grid size
    inputs:
        subset_options: list of 5 options dictionaries
        results: results dictionary from the numerical and analyitcal computations
        M: the array of grid sizes
    outputs:
        saved graph of error between RK4 and BSM for each options contract (one graph per grid size)
    """
    # for each grid size M
    for grid_size in M:
        plt.figure(figsize=(12, 7))
        # plot the error between numerical and analytical prices for all 5 options contracts
        for opt_id, opt in enumerate(subset_options):
            S = results[opt_id]['results_by_M'][grid_size]['S']
            error_num_ana = results[opt_id]['results_by_M'][grid_size]['error_num_ana']
            
            plt.plot(S, error_num_ana, label=f'Strike K = {opt["K"]:.2f}')
        
        plt.xlabel("Stock Price - S")
        plt.ylabel("|Price of RK4 - Price of BS|")
        plt.title(f"Absolute Error vs Stock Price for M = {grid_size}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"error_between_RK4_BSM_M{grid_size}.jpeg") # save the figure with the grid size in the filename

def plot_convergence(subset_options, results, M):
    """
    Plots the convergence of the numerical RK4 solution at S0 as the spatial grid size M increases
    input:
        subset_options: list of 5 options dictionaries
        results: results dictionary from the numerical and analyitcal computations
        M: the array of grid sizes
    output:
        saved graph of RK4 convergence based on options contract (one graph per contract)
    """
    # for each options contract plot the convergence of the numerical solution at S0 as the spatial grid size increases
    for opt_id, opt in enumerate(subset_options):
        errors_ana = []
        errors_market = []
        plt.figure(figsize=(10,6))
        for grid_size in M:
            errors_ana.append(results[opt_id]['results_by_M'][grid_size]['error_num_ana_S0'])
            errors_market.append(results[opt_id]['results_by_M'][grid_size]['error_num_market'])
        
        plt.figure(figsize=(8,6))
        plt.plot(M, errors_ana, marker='o', label='Error between Num and Ana')
        plt.plot(M, errors_market, marker='s', label='Error between Num and Market')
        plt.xlabel("Spatial Grid Size (M)")
        plt.ylabel(r"Absoulte Error at $S_0$")
        plt.title(f"Numerical Solution Convergence for TSLA at Strike K = {opt['K']:.0f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout
        plt.savefig(f"convergence_K{opt['K']:.0f}.png") # save the figure with the strike price in the filename

def ensure_unzipped_data(txt_path: str):
    """
    Ensures the cleaned dataset exists as a .txt file.
    If txt_path doesn't exist but txt_path + '.zip' exists, unzip it into the same folder.
    """
    if os.path.exists(txt_path):
        return  # already unzipped and ready

    zip_path = txt_path + ".zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Could not find '{txt_path}' or '{zip_path}'. "
            "Make sure the dataset is in the same folder as this script."
        )

    # Unzip
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=os.path.dirname(txt_path) or ".")
    print(f"Unzipped: {zip_path} -> {txt_path}")

if __name__ == "__main__":
    """
    Main function to load in the file, run the numerical and analytical computations for each option contract, and analyse the results with the plotting functions.
    """
    file = "tsla_2022_clean.txt"  # cleaned dataset (preferred)

    # auto-unzip if needed
    ensure_unzipped_data(file)

    print(file)
    options = read_options_data(file)

    S0_main = options[0]['S0'] # take the first stock price -- using this to pick a representative set of strikes
    subset_options = select_representative_strikes(options, S0_main, num_strikes=5) # pick set of strikes deep ITM, ITM, ATM, OTM, deep OTM within a reasonable range

    M = [25, 50, 100, 200] # number of spatial intervals (the grid sizes for spatial descritization)
    N = [200, 750, 3000, 12000] # number of RK4 time steps (corresponding to each spatial grid size), increase N for stability as M increases

    results = {} # dictionary to store the numerical and analytical computations

    for opt_id, opt in enumerate(subset_options):
        S0 = opt['S0'] # current stock price
        K = opt['K'] # strike price
        r = opt['r'] # risk-free rate
        sigma = opt['sigma'] # implied volatility
        T = opt['T'] # time to maturity
        market_price = opt['market_price'] # market option price

        S_max = 3 * S0  # max stock price on grid to fufill the infinite boundary condition on the black-scholes PDE

        results[opt_id] = {'params': opt, 'results_by_M': {}} # a space to store results for this option at each grid size

        print(f"\nOption contract with Current Price = {S0}, Strike K = {K}, Risk-Free Rate = {r}, Volatility = {sigma}, Time to Expiry = {T*252:.0f} days, Market Price = {market_price}")

        for i in range(len(M)):
            print(f"\nComputing numerical solution with M = {M[i]} spatial intervals...") # print progress
            S, C_num = MOL_RK4_bs_call(S_max, K, T, r, sigma, M[i], N[i]) # compute numerical solution
            C_an = exact_bs_call_price_on_grid(S, K, T, r, sigma) # compute analytical solution

            # extract prices at current stock price for numerical and analytical solutions for results comparison
            C_num_S0 = np.interp(S0, S, C_num) # numerical price at S0 using linear interpolation
            C_an_S0 = compute_bs_call_price_exact(S0, K, T, r, sigma) # analytical price at S0

            print(f"Numerical Price at current price (S0 = ${S0}): ${C_num_S0:.2f}") # print numerical price for comparison
            print(f"Analytical Price at current price (S0 = ${S0}): ${C_an_S0:.2f}") # print analytical price for comparison
            print(f"Market Price: ${market_price}")  # print market price for comparison

            # store results
            results[opt_id]['results_by_M'][M[i]] = {
                'S': S, # stock price grid
                'C_numerical': C_num, # numerical price on the grid
                'C_analytical': C_an, # analytical price on the grid
                'C_num_S0': C_num_S0, # numerical price at S0
                'C_an_S0': C_an_S0, # analytical price at S0
                'error_num_ana': np.abs(C_an - C_num), # error between numerical and analytical price on the grid
                'error_num_ana_S0': abs(C_an_S0 - C_num_S0), # error between numerical and analytical price at S0
                'error_num_market': abs(market_price - C_num_S0), # error between numerical and market price at S0
                'error_ana_market': abs(market_price - C_an_S0) # error between analytical and market price at S0
            }

    print_price_table(subset_options, results, M, N) # print the price comparison table (for all 5 options contracts and all 4 grid sizes)
    plot_price_comparision(subset_options, results, M) # plot the analytical vs numerical vs market prices for all options contracts (4 for each grid size)
    plot_convergence(subset_options, results, M) # plot the convergence of numerical prices at S0 as grid size increases (for each of the 5 options contracts)
    plot_error(subset_options, results, M) # plot the error between numerical and analytical prices for each grid size for each options contract (5 graphs)
