# Time-Series-Forecasting-for-Portfolio-Optimization

This project provides an end-to-end analysis pipeline for forecasting stock prices, analyzing financial market trends, and optimizing an investment portfolio to maximize returns while minimizing risk. The assets analyzed are Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY), each representing different risk and return profiles within a balanced portfolio.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Time Series Forecasting Models](#time-series-forecasting-models)
6. [Market Trend Forecasting](#market-trend-forecasting)
7. [Portfolio Optimization](#portfolio-optimization)
8. [Conclusion](#conclusion)
9. [Requirements](#requirements)

---

## Project Overview
The goal of this project is to:
1. **Predict stock price trends** for Tesla (TSLA) using time series models.
2. **Analyze market trends** and volatility, and evaluate risk and opportunities.
3. **Optimize an investment portfolio** using forecasted trends to maximize returns and minimize risk for TSLA, BND, and SPY.

---

## Setup and Installation
### Prerequisites
Ensure that you have the following installed:
- Python 3.8 or higher
- Git for cloning the repository

### Installation Steps
1. **Clone the repository**:
    ```bash
    git clone https://github.com/shammuse/Time-Series-Forecasting-for-Portfolio-Optimization.git
    cd ime-Series-Forecasting-for-Portfolio-Optimization
    ```

2. **Set up a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up API access**:
    - Register and obtain an API key for [Yahoo Finance](https://pypi.org/project/yfinance/), if required for additional data.
    - Store any sensitive keys or credentials in an `.env` file.

---

## Data Collection and Preprocessing
### Task 1: Preprocess and Explore the Data
Using the Yahoo Finance (YFinance) API, we collect historical price data for Tesla (TSLA), BND, and SPY to represent different risk profiles within a portfolio:
- **TSLA**: High return, high volatility.
- **BND**: Stability with low risk.
- **SPY**: Moderate-risk market exposure.

#### Data Cleaning
- Check for missing values and handle them by filling, interpolating, or removing.
- Ensure data types are appropriate for time series analysis.
- Normalize or scale data as required for machine learning models.

#### Exploratory Data Analysis (EDA)
- Visualize closing prices over time to identify trends and patterns.
- Calculate daily percentage changes to observe volatility.
- Detect outliers and unusual return days.
- Decompose time series into trend, seasonal, and residual components to identify patterns.

---

## Time Series Forecasting Models
### Task 2: Develop Time Series Forecasting Models
We experiment with several time series forecasting models, each offering unique advantages for predicting Tesla's stock prices:
- **ARIMA**: Suitable for non-seasonal univariate data.
- **SARIMA**: ARIMA with seasonal components.
- **LSTM**: A deep learning model that captures long-term dependencies in time series data.

#### Model Training and Evaluation
- Split data into training and testing sets to assess model performance.
- Use evaluation metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) to gauge accuracy.
- Use optimization techniques (e.g., grid search or auto_arima) to identify optimal model parameters.

---

## Market Trend Forecasting
### Task 3: Forecast Future Market Trends
Using the trained model, we forecast Tesla’s stock prices for the next 6-12 months:
- **Forecast Visualization**: Plot historical and forecasted prices with confidence intervals.
- **Trend Analysis**: Identify upward/downward trends and any anomalies.
- **Volatility Analysis**: Evaluate forecasted volatility and highlight periods with potential risks.
- **Risk and Opportunity Analysis**: Identify investment opportunities and risks based on forecasted price movements.

---

## Portfolio Optimization
### Task 4: Optimize Portfolio Based on Forecast
We optimize a simple portfolio of three assets:
1. **Tesla Stock (TSLA)** - Higher risk and growth potential.
2. **Vanguard Total Bond Market ETF (BND)** - Provides stability and low risk.
3. **S&P 500 ETF (SPY)** - Diversified, moderate-risk market exposure.

#### Portfolio Analysis
- Calculate annual returns and daily compound returns for each asset.
- Compute a covariance matrix to understand asset return correlations.
- Define portfolio weights to compute weighted average return and risk.

#### Optimization
- Use the Sharpe Ratio to optimize asset allocation for risk-adjusted returns.
- Adjust portfolio allocations to balance between risk and reward.
- Simulate portfolio performance based on forecasted returns, documenting expected returns, volatility, and the Sharpe Ratio.

#### Risk Management Metrics
- **Value at Risk (VaR)**: Estimate potential losses in Tesla stock at a given confidence level.
- **Sharpe Ratio**: Assess risk-adjusted return, where a higher ratio is preferred.

---

## Conclusion
This project demonstrates a comprehensive approach to data-driven investment strategy, covering data collection, cleaning, forecasting, and portfolio optimization. Key insights include:
- Projected trends in Tesla’s stock price.
- Insights into market volatility and potential risks.
- Optimal portfolio allocations based on forecasted trends and risk tolerance.

---

## Requirements
The following libraries are required:
- `yfinance`: For data extraction.
- `pandas`: Data manipulation.
- `numpy`: Numerical computations.
- `matplotlib` & `seaborn`: Data visualization.
- `scipy.stats`: Statistical analysis.
- `statsmodels`: Time series decomposition.
- `pmdarima`: ARIMA optimization.
- `keras` & `tensorflow`: For LSTM models.

Install dependencies using:
```bash
pip install -r requirements.txt
```
