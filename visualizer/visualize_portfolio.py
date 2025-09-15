import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class PortfolioVisualizer:
    def __init__(self, rawdata_path="../rawdata"):
        """
        Initialize the portfolio visualizer with paths to raw data
        """
        self.rawdata_path = rawdata_path
        self.omxs30_data = None
        self.sp500_data = None
        self.portfolio_data = None
        
    def load_data(self):
        """
        Load OMXS30 and S&P 500 data from CSV files
        """
        print("Loading portfolio data...")
        
        # Load OMXS30 data
        omxs30_path = os.path.join(self.rawdata_path, "OMXS30_10year_data.csv")
        self.omxs30_data = pd.read_csv(omxs30_path)
        self.omxs30_data['Date'] = pd.to_datetime(self.omxs30_data['Date'])
        self.omxs30_data = self.omxs30_data.sort_values('Date').reset_index(drop=True)
        
        # Load S&P 500 data
        sp500_path = os.path.join(self.rawdata_path, "SP500_10year_data.csv")
        self.sp500_data = pd.read_csv(sp500_path)
        self.sp500_data['Date'] = pd.to_datetime(self.sp500_data['Date'])
        self.sp500_data = self.sp500_data.sort_values('Date').reset_index(drop=True)
        
        print(f"OMXS30 data: {len(self.omxs30_data)} records from {self.omxs30_data['Date'].min()} to {self.omxs30_data['Date'].max()}")
        print(f"S&P 500 data: {len(self.sp500_data)} records from {self.sp500_data['Date'].min()} to {self.sp500_data['Date'].max()}")
        
    def calculate_buy_hold_performance(self, initial_investment=10000, allocation=None):
        """
        Calculate buy-and-hold strategy performance
        
        Args:
            initial_investment (float): Initial investment amount
            allocation (dict): Portfolio allocation {'OMXS30': 0.5, 'SP500': 0.5}
                              If None, defaults to 50/50 split
        """
        if allocation is None:
            allocation = {'OMXS30': 0.5, 'SP500': 0.5}
            
        print(f"Calculating buy-and-hold performance with initial investment: ${initial_investment:,.2f}")
        print(f"Portfolio allocation: OMXS30: {allocation['OMXS30']*100:.1f}%, S&P 500: {allocation['SP500']*100:.1f}%")
        
        # Find common date range
        start_date = max(self.omxs30_data['Date'].min(), self.sp500_data['Date'].min())
        end_date = min(self.omxs30_data['Date'].max(), self.sp500_data['Date'].max())
        
        # Filter data to common date range
        omxs30_filtered = self.omxs30_data[
            (self.omxs30_data['Date'] >= start_date) & 
            (self.omxs30_data['Date'] <= end_date)
        ].copy()
        
        sp500_filtered = self.sp500_data[
            (self.sp500_data['Date'] >= start_date) & 
            (self.sp500_data['Date'] <= end_date)
        ].copy()
        
        # Merge data on dates
        portfolio_data = pd.merge(
            omxs30_filtered[['Date', 'Close']], 
            sp500_filtered[['Date', 'Close']], 
            on='Date', 
            suffixes=('_OMXS30', '_SP500')
        )
        
        # Calculate initial prices and shares
        initial_omxs30_price = portfolio_data['Close_OMXS30'].iloc[0]
        initial_sp500_price = portfolio_data['Close_SP500'].iloc[0]
        
        # Calculate number of shares based on allocation
        omxs30_investment = initial_investment * allocation['OMXS30']
        sp500_investment = initial_investment * allocation['SP500']
        
        omxs30_shares = omxs30_investment / initial_omxs30_price
        sp500_shares = sp500_investment / initial_sp500_price
        
        # Calculate portfolio value over time
        portfolio_data['OMXS30_Value'] = omxs30_shares * portfolio_data['Close_OMXS30']
        portfolio_data['SP500_Value'] = sp500_shares * portfolio_data['Close_SP500']
        portfolio_data['Total_Value'] = portfolio_data['OMXS30_Value'] + portfolio_data['SP500_Value']
        
        # Calculate percentage returns
        portfolio_data['Total_Return_Pct'] = ((portfolio_data['Total_Value'] / initial_investment) - 1) * 100
        portfolio_data['OMXS30_Return_Pct'] = ((portfolio_data['Close_OMXS30'] / initial_omxs30_price) - 1) * 100
        portfolio_data['SP500_Return_Pct'] = ((portfolio_data['Close_SP500'] / initial_sp500_price) - 1) * 100
        
        # Calculate daily returns for volatility
        portfolio_data['Daily_Return'] = portfolio_data['Total_Value'].pct_change() * 100
        portfolio_data['OMXS30_Daily_Return'] = portfolio_data['Close_OMXS30'].pct_change() * 100
        portfolio_data['SP500_Daily_Return'] = portfolio_data['Close_SP500'].pct_change() * 100
        
        self.portfolio_data = portfolio_data
        self.initial_investment = initial_investment
        self.allocation = allocation
        self.omxs30_shares = omxs30_shares
        self.sp500_shares = sp500_shares
        
        return portfolio_data
    
    def calculate_portfolio_metrics(self):
        """
        Calculate key portfolio performance metrics
        """
        if self.portfolio_data is None:
            raise ValueError("Portfolio data not calculated. Run calculate_buy_hold_performance first.")
        
        data = self.portfolio_data
        
        # Total return
        total_return = data['Total_Return_Pct'].iloc[-1]
        omxs30_return = data['OMXS30_Return_Pct'].iloc[-1]
        sp500_return = data['SP500_Return_Pct'].iloc[-1]
        
        # Time period
        start_date = data['Date'].iloc[0]
        end_date = data['Date'].iloc[-1]
        years = (end_date - start_date).days / 365.25
        
        # Annualized returns
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        omxs30_annual = ((1 + omxs30_return/100) ** (1/years) - 1) * 100
        sp500_annual = ((1 + sp500_return/100) ** (1/years) - 1) * 100
        
        # Volatility (standard deviation of daily returns)
        portfolio_volatility = data['Daily_Return'].std() * np.sqrt(252)  # Annualized
        omxs30_volatility = data['OMXS30_Daily_Return'].std() * np.sqrt(252)
        sp500_volatility = data['SP500_Daily_Return'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 2.0
        sharpe_ratio = (annual_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        omxs30_sharpe = (omxs30_annual - risk_free_rate) / omxs30_volatility if omxs30_volatility > 0 else 0
        sp500_sharpe = (sp500_annual - risk_free_rate) / sp500_volatility if sp500_volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = data['Total_Value'].expanding().max()
        drawdown = (data['Total_Value'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        metrics = {
            'period': {
                'start_date': start_date,
                'end_date': end_date,
                'years': years
            },
            'portfolio': {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': data['Total_Value'].iloc[-1]
            },
            'omxs30': {
                'total_return': omxs30_return,
                'annual_return': omxs30_annual,
                'volatility': omxs30_volatility,
                'sharpe_ratio': omxs30_sharpe
            },
            'sp500': {
                'total_return': sp500_return,
                'annual_return': sp500_annual,
                'volatility': sp500_volatility,
                'sharpe_ratio': sp500_sharpe
            }
        }
        
        return metrics
