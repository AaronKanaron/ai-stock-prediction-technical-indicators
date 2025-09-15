import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import joblib
from sklearn.preprocessing import StandardScaler

class PortfolioVisualizer:
    def __init__(self, rawdata_path="./rawdata"):
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
    
    def load_model(self, model_path=None):
        """
        Load the trained Random Forest model
        """
        if model_path is None:
            # Find the most recent model
            model_files = [f for f in os.listdir("./models") if f.startswith("random_forest_model") and f.endswith(".pkl")]
            if not model_files:
                raise FileNotFoundError("No Random Forest models found in models/ directory")
            model_path = os.path.join("./models", sorted(model_files)[-1])
        
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        return self.model
    
    def calculate_technical_indicators(self, data):
        """
        Calculate technical indicators needed for the model
        """
        data = data.copy()
        
        # Daily return
        data['Return_1d'] = data['Close'].pct_change()
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Simple Moving Averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        
        # Bollinger Bands
        sma_20 = data['SMA_20']  # Use the SMA_20 we just calculated
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = sma_20 + (bb_std * 2)
        data['BB_lower'] = sma_20 - (bb_std * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        
        # Rolling Standard Deviation
        data['Rolling_Std_20'] = data['Close'].rolling(window=20).std()
        
        # Enhanced momentum features (lagged to avoid lookahead bias)
        data['Return_5d_lag'] = data['Close'].pct_change(5).shift(1)
        data['Return_10d_lag'] = data['Close'].pct_change(10).shift(1)
        
        # Enhanced trend features
        data['SMA_ratio'] = data['Close'] / data['SMA_20']
        data['Price_change_20d'] = data['Close'] / data['Close'].shift(20) - 1
        
        return data
    
    def generate_model_predictions(self, data, model):
        """
        Generate model predictions for trading signals
        """
        # Calculate technical indicators
        data_with_indicators = self.calculate_technical_indicators(data)
        
        # Model features (same as in main.py - all 11 features used in the combined model)
        model_features = [
            'Return_1d', 'MACD_histogram', 'RSI_14', 'BB_width', 
            'Volume', 'SMA_5', 'Rolling_Std_20', 'Return_5d_lag',
            'Return_10d_lag', 'SMA_ratio', 'Price_change_20d'
        ]
        
        # Prepare features for prediction
        feature_data = data_with_indicators[model_features].dropna()
        
        if feature_data.empty:
            return None
        
        # Generate predictions (probabilities of positive class)
        predictions = model.predict_proba(feature_data.values)[:, 1]
        
        # Create signals dataframe
        signals_df = pd.DataFrame({
            'Date': data_with_indicators.loc[feature_data.index, 'Date'],
            'Close': data_with_indicators.loc[feature_data.index, 'Close'],
            'Prediction_Prob': predictions,
            'Signal': (predictions > 0.5).astype(int)  # 1 = buy/hold, 0 = sell/avoid
        })
        
        return signals_df
    
    def simulate_model_strategy(self, initial_investment=10000):
        """
        Simulate trading strategy based on model predictions
        """
        if self.portfolio_data is None:
            raise ValueError("Portfolio data not calculated. Run calculate_buy_hold_performance first.")
        
        if not hasattr(self, 'model'):
            raise ValueError("Model not loaded. Run load_model first.")
        
        print("Simulating model-based trading strategy...")
        
        # Generate predictions for OMXS30 (using it as the main trading signal)
        omxs30_signals = self.generate_model_predictions(self.omxs30_data, self.model)
        
        if omxs30_signals is None:
            print("Could not generate model predictions")
            return None
        
        # Merge signals with portfolio data
        portfolio_with_signals = pd.merge(
            self.portfolio_data[['Date', 'Close_OMXS30', 'Close_SP500']],
            omxs30_signals[['Date', 'Signal', 'Prediction_Prob']],
            on='Date',
            how='left'
        )
        
        # Forward fill signals for missing dates
        portfolio_with_signals['Signal'] = portfolio_with_signals['Signal'].ffill()
        portfolio_with_signals['Prediction_Prob'] = portfolio_with_signals['Prediction_Prob'].ffill()
        
        # Initialize model portfolio tracking
        cash = initial_investment
        omxs30_shares = 0
        sp500_shares = 0
        portfolio_values = []
        positions = []
        
        for i, row in portfolio_with_signals.iterrows():
            current_signal = row['Signal']
            omxs30_price = row['Close_OMXS30']
            sp500_price = row['Close_SP500']
            
            if pd.isna(current_signal):
                current_signal = 0  # Default to no position
            
            # Current portfolio value
            current_value = cash + (omxs30_shares * omxs30_price) + (sp500_shares * sp500_price)
            
            # Trading logic based on signal
            if current_signal == 1:  # Buy signal
                if cash > 0:
                    # Invest all cash in 50/50 split
                    omxs30_investment = cash * 0.5
                    sp500_investment = cash * 0.5
                    
                    omxs30_shares += omxs30_investment / omxs30_price
                    sp500_shares += sp500_investment / sp500_price
                    cash = 0
                    
            elif current_signal == 0:  # Sell signal
                if omxs30_shares > 0 or sp500_shares > 0:
                    # Sell all positions
                    cash = (omxs30_shares * omxs30_price) + (sp500_shares * sp500_price)
                    omxs30_shares = 0
                    sp500_shares = 0
            
            # Recalculate portfolio value after trading
            portfolio_value = cash + (omxs30_shares * omxs30_price) + (sp500_shares * sp500_price)
            portfolio_values.append(portfolio_value)
            positions.append({
                'cash': cash,
                'omxs30_shares': omxs30_shares,
                'sp500_shares': sp500_shares,
                'signal': current_signal
            })
        
        # Add model results to portfolio data
        portfolio_with_signals['Model_Portfolio_Value'] = portfolio_values
        portfolio_with_signals['Model_Return_Pct'] = ((portfolio_with_signals['Model_Portfolio_Value'] / initial_investment) - 1) * 100
        
        self.model_portfolio_data = portfolio_with_signals
        self.model_initial_investment = initial_investment
        
        return portfolio_with_signals
    
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
    
    def create_portfolio_visualization(self, save_plots=True):
        """
        Create portfolio value comparison visualization
        """
        if self.portfolio_data is None:
            raise ValueError("Portfolio data not calculated. Run calculate_buy_hold_performance first.")
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Create figure with subplots for comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot 1: Portfolio Value Comparison
        ax1.plot(self.portfolio_data['Date'], self.portfolio_data['Total_Value'], 
                linewidth=3, label='Buy & Hold Strategy', color='#2E86AB')
        
        if hasattr(self, 'model_portfolio_data'):
            ax1.plot(self.model_portfolio_data['Date'], self.model_portfolio_data['Model_Portfolio_Value'], 
                    linewidth=3, label='Model Strategy', color='#E74C3C')
        
        ax1.axhline(y=self.initial_investment, color='black', linestyle='--', alpha=0.7, 
                   label=f'Initial Investment (${self.initial_investment:,.0f})')
        
        ax1.set_title('Portfolio Value Comparison: Buy & Hold vs Model Strategy', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 2: Model Trading Signals
        if hasattr(self, 'model_portfolio_data'):
            # Plot OMXS30 price as background
            ax2_price = ax2.twinx()
            ax2_price.plot(self.model_portfolio_data['Date'], self.model_portfolio_data['Close_OMXS30'], 
                          color='gray', alpha=0.3, linewidth=1, label='OMXS30 Price')
            ax2_price.set_ylabel('OMXS30 Price', fontsize=12, color='gray')
            ax2_price.tick_params(axis='y', labelcolor='gray')
            
            # Plot buy/sell signals
            buy_signals = self.model_portfolio_data[self.model_portfolio_data['Signal'] == 1]
            sell_signals = self.model_portfolio_data[self.model_portfolio_data['Signal'] == 0]
            
            # Plot buy signals (green triangles pointing up)
            if not buy_signals.empty:
                ax2.scatter(buy_signals['Date'], [1] * len(buy_signals), 
                           color='green', marker='^', s=20, alpha=0.7, label='Buy Signal')
            
            # Plot sell signals (red triangles pointing down)
            if not sell_signals.empty:
                ax2.scatter(sell_signals['Date'], [0] * len(sell_signals), 
                           color='red', marker='v', s=20, alpha=0.7, label='Sell Signal')
            
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Sell/Cash', 'Buy/Hold'])
        else:
            ax2.text(0.5, 0.5, 'No model signals available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        
        ax2.set_title('Model Trading Signals', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Signal', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('portfolio_comparison.png', dpi=300, bbox_inches='tight')
            print("Portfolio comparison saved as 'portfolio_comparison.png'")
        
        plt.show()
        
        return fig
    
    def print_strategy_comparison(self):
        """
        Print comparison between buy-and-hold and model strategies
        """
        if not hasattr(self, 'model_portfolio_data'):
            print("Model strategy not calculated. Run simulate_model_strategy first.")
            return self.print_portfolio_summary()
        
        # Calculate metrics for both strategies
        buy_hold_metrics = self.calculate_portfolio_metrics()
        
        # Model strategy metrics
        model_data = self.model_portfolio_data
        model_final_value = model_data['Model_Portfolio_Value'].iloc[-1]
        model_total_return = ((model_final_value / self.model_initial_investment) - 1) * 100
        
        # Time period
        start_date = model_data['Date'].iloc[0]
        end_date = model_data['Date'].iloc[-1]
        years = (end_date - start_date).days / 365.25
        
        model_annual_return = ((model_final_value / self.model_initial_investment) ** (1/years) - 1) * 100
        
        print("\n" + "="*90)
        print("STRATEGY PERFORMANCE COMPARISON")
        print("="*90)
        
        # Period information
        print(f"\nAnalysis Period:")
        print(f"  Start Date: {start_date.strftime('%Y-%m-%d')}")
        print(f"  End Date: {end_date.strftime('%Y-%m-%d')}")
        print(f"  Duration: {years:.1f} years")
        print(f"  Initial Investment: ${self.initial_investment:,.2f}")
        
        print(f"\n{'Strategy':<20} {'Final Value':<15} {'Total Return':<15} {'Annual Return':<15}")
        print("-" * 65)
        print(f"{'Buy & Hold':<20} ${buy_hold_metrics['portfolio']['final_value']:<14,.0f} {buy_hold_metrics['portfolio']['total_return']:<14.2f}% {buy_hold_metrics['portfolio']['annual_return']:<14.2f}%")
        print(f"{'Model Strategy':<20} ${model_final_value:<14,.0f} {model_total_return:<14.2f}% {model_annual_return:<14.2f}%")
        
        # Performance comparison
        print(f"\n🏆 WINNER ANALYSIS:")
        if model_total_return > buy_hold_metrics['portfolio']['total_return']:
            outperformance = model_total_return - buy_hold_metrics['portfolio']['total_return']
            print(f"  Model Strategy WINS by {outperformance:.2f} percentage points!")
            print(f"  Model final value: ${model_final_value:,.2f}")
            print(f"  Buy & Hold final value: ${buy_hold_metrics['portfolio']['final_value']:,.2f}")
            print(f"  Extra profit from model: ${model_final_value - buy_hold_metrics['portfolio']['final_value']:,.2f}")
        else:
            underperformance = buy_hold_metrics['portfolio']['total_return'] - model_total_return
            print(f"  Buy & Hold WINS by {underperformance:.2f} percentage points!")
            print(f"  Buy & Hold final value: ${buy_hold_metrics['portfolio']['final_value']:,.2f}")
            print(f"  Model final value: ${model_final_value:,.2f}")
            print(f"  Missed profit: ${buy_hold_metrics['portfolio']['final_value'] - model_final_value:,.2f}")
        
        # Trading activity analysis
        signals = self.model_portfolio_data['Signal'].dropna()
        signal_changes = (signals != signals.shift()).sum()
        
        print(f"\n📊 MODEL STRATEGY DETAILS:")
        print(f"  Total trading signals generated: {len(signals)}")
        print(f"  Signal changes (trades): {signal_changes}")
        print(f"  Time in market: {(signals == 1).mean()*100:.1f}%")
        print(f"  Time in cash: {(signals == 0).mean()*100:.1f}%")
        
        print("="*90)
        
        return {
            'buy_hold': buy_hold_metrics,
            'model_strategy': {
                'final_value': model_final_value,
                'total_return': model_total_return,
                'annual_return': model_annual_return,
                'trades': signal_changes,
                'time_in_market': (signals == 1).mean()*100
            }
        }

def main():
    """
    Main function to run the portfolio visualization with model comparison
    """
    print("🚀 Starting Portfolio Visualization Tool")
    print("📊 Comparing Buy & Hold vs Model Strategy Performance")
    
    # Initialize visualizer
    visualizer = PortfolioVisualizer()
    
    # Load data
    visualizer.load_data()
    
    # Calculate buy-and-hold portfolio performance
    print("\n📈 Calculating buy-and-hold performance...")
    visualizer.calculate_buy_hold_performance(
        initial_investment=10000,
        allocation={'OMXS30': 0.5, 'SP500': 0.5}
    )
    
    try:
        # Load the Random Forest model
        print("\n🤖 Loading Random Forest model...")
        visualizer.load_model()
        
        # Simulate model-based trading strategy
        print("📊 Simulating model-based trading strategy...")
        visualizer.simulate_model_strategy(initial_investment=10000)
        
        # Print comparison summary
        visualizer.print_strategy_comparison()
        
        # Create comparison visualizations
        print("\n📊 Creating strategy comparison visualizations...")
        visualizer.create_portfolio_visualization(save_plots=True)
        
        print("\n✅ Strategy comparison complete!")
        print("📋 Check 'portfolio_comparison.png' for detailed charts")
        
    except Exception as e:
        print(f"\n⚠️ Could not load model or run model strategy: {str(e)}")
        print("📊 Falling back to buy-and-hold analysis only...")
        
        # Fallback to simple buy-and-hold analysis
        metrics = visualizer.calculate_portfolio_metrics()
        
        print("\n" + "="*80)
        print("BUY & HOLD STRATEGY PERFORMANCE")
        print("="*80)
        print(f"Final Value: ${metrics['portfolio']['final_value']:,.2f}")
        print(f"Total Return: {metrics['portfolio']['total_return']:+.2f}%")
        print(f"Annualized Return: {metrics['portfolio']['annual_return']:+.2f}%")
        print("="*80)
        
        # Create simple visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(visualizer.portfolio_data['Date'], visualizer.portfolio_data['Total_Value'], 
                linewidth=3, label='Buy & Hold Portfolio', color='#2E86AB')
        plt.axhline(y=visualizer.initial_investment, color='black', linestyle='--', alpha=0.7, 
                   label=f'Initial Investment (${visualizer.initial_investment:,.0f})')
        plt.title('Portfolio Value Over Time (Buy & Hold Strategy)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.tight_layout()
        plt.savefig('portfolio_value.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📋 Check 'portfolio_value.png' for the chart")

if __name__ == "__main__":
    main()
