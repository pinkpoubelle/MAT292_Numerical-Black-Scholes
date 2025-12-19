# MAT292_Numerical-Black-Scholes

## Numerical Analysis of the Black–Scholes Option Pricing Model  
**Method of Lines (MOL) with Fourth-Order Runge–Kutta (RK4)**

---

## Overview

This project numerically solves the **Black–Scholes partial differential equation (PDE)** for European call options using the **Method of Lines (MOL)** combined with a **fourth-order Runge–Kutta (RK4)** time-stepping scheme.  

The numerical solution is compared against:
- the **analytical Black–Scholes formula**, and
- **real market option prices** for Tesla (TSLA) sourced from Kaggle (https://www.kaggle.com/datasets/kylegraupe/tsla-daily-eod-options-quotes-2019-2022).

The goal is to analyze **accuracy, convergence, and stability** of the numerical method as the spatial and temporal grid resolutions increase.

This project was developed for **MAT292 – Ordinary Differential Equations**.

---

## Mathematical Background

The Black–Scholes PDE for a European call option C(S, t) is:

∂C/∂t = (1/2) σ² S² ∂²C/∂S² + r S ∂C/∂S − r C

with:

- **Terminal condition**  
  C(S, T) = max(S − K, 0)

- **Boundary conditions**  
  C(0, t) = 0  
  C(S_max, t) = S_max − K · exp(−r (T − t))

The Method of Lines discretizes the spatial variable S using finite differences,
reducing the PDE to a system of ODEs in time, which are solved using RK4.

---

## Features

- Reads real **options chain data** from txt (uses modified txt file to fit into max file size for repo)
- Computes **analytical Black–Scholes prices**
- Solves the PDE numerically using **MOL + RK4**
- Compares:
  - Numerical vs Analytical prices
  - Numerical vs Market prices
- Studies **convergence** as spatial grid size increases
- Generates:
  - Price comparison plots
  - Error plots
  - Convergence plots
  - Summary tables of prices and errors

---

## Project Structure

```

MAT292_Numerical-Black-Scholes-main/
│
├── BlackScholes_MOL_RK4.py      # Main program
├── tsla_2022_clean.txt          # Options data (txt format)
├── README.md                   # Project documentation
│
├── price_comparison_M25.jpeg # Price comparison plots for each spatial grid
├── price_comparison_M50.jpeg
├── price_comparison_M100.jpeg
├── price_comparison_M200.jpeg
├── error_between_RK4_BSM_M25.jpeg # Error plots for each spatial grid
├── error_between_RK4_BSM_M50.jpeg
├── error_between_RK4_BSM_M100.jpeg
├── error_between_RK4_BSM_M200.jpeg
├── convergence_K433.png # Convergence plots for each options contract
├── convergence_K650.png # indicated by strike price
├── convergence_K830.png
├── convergence_K1020.png
├── convergence_K1295.png

````

---

## Dependencies

The project uses standard scientific Python libraries:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`

Install dependencies with:
```bash
pip install numpy pandas scipy matplotlib
````

---

## How to Run

1. Ensure `tsla_2019_2022.txt` is in the same directory as the script.
2. Run the program:

```bash
python BlackScholes_MOL_RK4.py
```

The script will:

* Select representative option contracts (deep ITM → deep OTM)
* Compute numerical solutions for multiple grid sizes
* Print a comparison table in the terminal
* Save all plots to the working directory

---

## Numerical Parameters

The numerical experiments use:

* **Spatial grid sizes:**
  [
  M = [25, 50, 100, 200]
  ]

* **Time steps (RK4):**
  [
  N = [200, 750, 3000, 12000]
  ]

Time steps are increased with spatial resolution to ensure stability.

---

## Outputs

### Terminal Output

* Numerical price at ( S_0 )
* Analytical price at ( S_0 )
* Market price
* Absolute errors

### Plots

* Numerical vs analytical vs market price curves
* Absolute error vs stock price
* Convergence of numerical solution at ( S_0 )

---

## Assumptions

* European call options only
* Constant volatility and interest rate
* No dividends
* Frictionless market
* Risk-free rate fixed at **2%** (based on 2022 treasury yields)
* 252 trading days per year

---

## Notes

* The spatial domain is truncated at ( S_max = 3S_0 ) to approximate the infinite boundary.
* Linear interpolation is used to extract numerical prices at ( S_0 ).
* Results improve monotonically as grid resolution increases, demonstrating convergence.

---

## Authors

* **Kyeongeun (Ellie) Kim**
* **Heidi Ma**
* **Kimmy Tran**

---

## Course

MAT292 — Ordinary Differential Equations
University of Toronto

---

## License

This project is for **educational purposes only**.
