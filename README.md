# AI Stock Prediction with Technical Indicators

Predicting Swedish stock prices using XGBoost and technical analysis. This project trains a model on OMXS30 stocks to classify whether to buy, hold, or sell based on price patterns and indicators.

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="/visualizer/images/walk_forward_analysis.png" alt="Model" width="49%">
  <img src="/visualizer/images/portfolio_comparison.png" alt="Performance" width="49%">
</div>


## How It Works

The system downloads historical stock data, calculates technical indicators (RSI, MACD, Bollinger Bands, etc.), and trains an XGBoost model using walk-forward validation. The model learns to predict 5-day forward returns based on current market conditions.

## Running the Project

Run these scripts in order:

### 1. Gather Stock Data
```bash
python gather.py
```
Downloads 10 years of daily price data from Yahoo Finance. Edit the `symbols` dictionary to choose which stocks to fetch.

### 2. Calculate Technical Indicators
```bash
python indicators.py
```
Processes the raw data and adds technical indicators. Creates the training dataset with features like moving averages, RSI, volume patterns, and volatility metrics.

### 3. Train the Model
```bash
python xgboost_walk_forward.py
```
Trains an XGBoost classifier using walk-forward validation. The model learns to predict buy/hold/sell signals. Saves the best model to `models/`.

### 4. Visualize Results
```bash
streamlit run visualizer/st.py
```
Opens an interactive dashboard showing portfolio performance, comparing the AI strategy against a buy-and-hold benchmark. Includes trade markers, performance metrics, and signal analysis.

## Requirements

Install dependencies:
```bash
pip install pandas numpy yfinance ta xgboost scikit-learn matplotlib seaborn streamlit plotly joblib progress
```

## Model Details

- **Algorithm**: XGBoost (Gradient Boosting)
- **Target**: 3-class classification (0=Sell, 1=Hold, 2=Buy)
- **Features**: 15-20 technical indicators selected via univariate feature selection
- **Validation**: Walk-forward cross-validation with 24-month training, 3-month validation, 3-month testing
- **Class Balancing**: Moderate balanced weighting to handle class imbalance

The target variable uses volatility-adjusted percentile thresholds - buy signals trigger when 5-day forward returns exceed the 80th percentile, sell signals below the 20th percentile.

## Project Structure

```
.
├── gather.py              # Downloads stock data
├── indicators.py          # Calculates technical indicators
├── xgboost_walk_forward.py # Trains and validates model
├── visualizer/
│   ├── st.py             # Streamlit dashboard
│   └── visualize_portfolio.py # Portfolio simulation logic
├── data/                 # Processed data with indicators
├── rawdata/              # Raw OHLCV data
└── models/               # Trained models
```

## Notes

Transaction fees default to 0.1% per trade. The model typically achieves 40-50% accuracy on test data, which can still be profitable when combined with proper position sizing and risk management.
