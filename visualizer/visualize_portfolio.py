import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import joblib
from sklearn.preprocessing import StandardScaler

class PortfolioVisualizer:
    def __init__(self, data_path="./data", benchmark_stock="OMXS30"):
        """
        Initialize the portfolio visualizer with paths to processed data
        """
        self.data_path = data_path
        self.benchmark_stock = benchmark_stock
        self.benchmark_data = None
        self.portfolio_data = None
        
    def load_data(self):
        """
        Load benchmark stock data from processed CSV files
        """
        print(f"Loading {self.benchmark_stock} data...")
        
        # Load benchmark data
        benchmark_path = os.path.join(self.data_path, f"{self.benchmark_stock}.csv")
        self.benchmark_data = pd.read_csv(benchmark_path)
        self.benchmark_data['Date'] = pd.to_datetime(self.benchmark_data['Date'])
        self.benchmark_data = self.benchmark_data.sort_values('Date').reset_index(drop=True)
        
        print(f"{self.benchmark_stock} data: {len(self.benchmark_data)} records from {self.benchmark_data['Date'].min()} to {self.benchmark_data['Date'].max()}")
        
    def calculate_benchmark_performance(self):
        """
        Calculate buy-and-hold benchmark performance for single stock
        """
        print(f"Calculating benchmark performance for {self.benchmark_stock}")
        
        # Calculate normalized returns (starting from 100)
        portfolio_data = self.benchmark_data[['Date', 'Close']].copy()
        portfolio_data['Benchmark_normalized'] = (portfolio_data['Close'] / portfolio_data['Close'].iloc[0]) * 100
        
        # Calculate percentage returns
        portfolio_data['Benchmark_Return_Pct'] = portfolio_data['Benchmark_normalized'] - 100
        
        # Calculate daily returns for volatility
        portfolio_data['Daily_Return'] = portfolio_data['Benchmark_normalized'].pct_change() * 100
        
        self.portfolio_data = portfolio_data
        
        return portfolio_data
    
    def load_model(self, model_path=None):
        """
        Load the trained machine learning model (XGBoost or Random Forest)
        """
        if model_path is None:
            # Try to find XGBoost model first, then Random Forest
            xgboost_files = [f for f in os.listdir("./models") if f.startswith("walkforward_xgboost") and f.endswith(".pkl")]
            rf_files = [f for f in os.listdir("./models") if f.startswith("random_forest_model") and f.endswith(".pkl")]
            
            if xgboost_files:
                model_path = os.path.join("./models", sorted(xgboost_files)[-1])
                print(f"Found XGBoost model: {sorted(xgboost_files)[-1]}")
            elif rf_files:
                model_path = os.path.join("./models", sorted(rf_files)[-1])
                print(f"Found Random Forest model: {sorted(rf_files)[-1]}")
            else:
                raise FileNotFoundError("No XGBoost or Random Forest models found in models/ directory")
        
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        return self.model
    
    def generate_model_predictions(self, data, model):
        """
        Generate model predictions for trading signals using pre-calculated indicators
        """
        # Model features (all technical indicators are already in the data)
        model_features = [
            "Open", "High", "Low", "Close", "Volume", "SMA_5", "SMA_10", "SMA_20", 
            "RSI_14", "MACD", "MACD_signal", "MACD_histogram", "BB_upper", 
            "BB_middle", "BB_lower", "BB_width", "Rolling_Std_20", "Return_1d", 
            "Return_5d_lag", "Return_10d_lag", "SMA_ratio", "Price_change_20d"
        ]
        
        # Prepare features for prediction
        feature_data = data[model_features].dropna()
        
        if feature_data.empty:
            return None
        
        # Generate predictions for 3-class classification (0=sell, 1=hold, 2=buy)
        class_predictions = model.predict(feature_data.values)
        prediction_probabilities = model.predict_proba(feature_data.values)
        
        # Create signals dataframe
        signals_df = pd.DataFrame({
            'Date': data.loc[feature_data.index, 'Date'],
            'Close': data.loc[feature_data.index, 'Close'],
            'Signal': class_predictions,  # 0=sell, 1=hold, 2=buy
            'Prob_Sell': prediction_probabilities[:, 0],    # Probability of sell (class 0)
            'Prob_Hold': prediction_probabilities[:, 1],    # Probability of hold (class 1)
            'Prob_Buy': prediction_probabilities[:, 2]      # Probability of buy (class 2)
        })
        
        return signals_df
    
    def simulate_model_strategy(self, debug=False):
        """
        Simulate trading strategy based on model predictions using normalized returns
        """
        if self.portfolio_data is None:
            raise ValueError("Portfolio data not calculated. Run calculate_benchmark_performance first.")
        
        if not hasattr(self, 'model'):
            raise ValueError("Model not loaded. Run load_model first.")
        
        print("Simulating model-based trading strategy...")
        
        # Generate predictions for the benchmark stock
        benchmark_signals = self.generate_model_predictions(self.benchmark_data, self.model)
        
        if benchmark_signals is None:
            print("Could not generate model predictions")
            return None
        
        # Merge signals with portfolio data
        portfolio_with_signals = pd.merge(
            self.portfolio_data[['Date', 'Close']],
            benchmark_signals[['Date', 'Signal', 'Prob_Sell', 'Prob_Hold', 'Prob_Buy']],
            on='Date',
            how='left'
        )
        
        # Forward fill signals for missing dates
        portfolio_with_signals['Signal'] = portfolio_with_signals['Signal'].ffill()
        portfolio_with_signals['Prob_Sell'] = portfolio_with_signals['Prob_Sell'].ffill()
        portfolio_with_signals['Prob_Hold'] = portfolio_with_signals['Prob_Hold'].ffill()
        portfolio_with_signals['Prob_Buy'] = portfolio_with_signals['Prob_Buy'].ffill()
        
        # Initialize model portfolio tracking (normalized to 100 at start)
        portfolio_value = 100.0  # Start with value of 100
        in_market = False  # Track if we're in the market or in cash
        model_values = []
        
        # Get benchmark stock values for reference
        benchmark_values = self.portfolio_data['Benchmark_normalized'].values
        
        # Debug tracking
        trades = []
        prev_signal = None
        
        for i, row in portfolio_with_signals.iterrows():
            current_signal = row['Signal']
            date = row['Date']
            
            if pd.isna(current_signal):
                current_signal = 1  # Default to hold position
            
            trade_executed = False
            trade_type = "none"
            
            # Trading logic based on 3-class signal
            if current_signal == 2:  # Buy signal
                if not in_market:
                    # Move from cash to market - start tracking market performance
                    in_market = True
                    trade_executed = True
                    trade_type = "buy"
                    
                    if debug and len(trades) < 20:  # Show first 20 trades for debugging
                        print(f"{date.strftime('%Y-%m-%d')}: BUY - Moved to market at {portfolio_value:.2f}")
                    
            elif current_signal == 0:  # Sell signal
                if in_market:
                    # Move from market to cash - stop tracking market performance
                    in_market = False
                    trade_executed = True
                    trade_type = "sell"
                    
                    if debug and len(trades) < 20:  # Show first 20 trades for debugging
                        print(f"{date.strftime('%Y-%m-%d')}: SELL - Moved to cash at {portfolio_value:.2f}")
            
            # Update portfolio value based on position
            if in_market:
                # If in market, follow the benchmark performance
                if i > 0:
                    daily_return = benchmark_values[i] / benchmark_values[i-1]
                    portfolio_value = portfolio_value * daily_return
            # If in cash (not in_market), portfolio_value stays the same
            
            model_values.append(portfolio_value)
            
            if trade_executed:
                trades.append({
                    'date': date,
                    'type': trade_type,
                    'signal': current_signal,
                    'portfolio_value': portfolio_value
                })
            
            prev_signal = current_signal
        
        # Add model results to portfolio data
        portfolio_with_signals['Model_normalized'] = model_values
        portfolio_with_signals['Model_Return_Pct'] = np.array(model_values) - 100
        
        # Create a separate dataframe for actual trades only
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        self.model_portfolio_data = portfolio_with_signals
        self.model_trades_data = trades_df  # Store actual trades separately
        
        # Print debugging info
        if debug:
            print(f"\n🔍 TRADING ANALYSIS:")
            print(f"  Total trades executed: {len(trades)}")
            if trades:
                buy_trades = [t for t in trades if t['type'] == 'buy']
                sell_trades = [t for t in trades if t['type'] == 'sell']
                print(f"    - Buy trades: {len(buy_trades)}")
                print(f"    - Sell trades: {len(sell_trades)}")
            print(f"  Signal changes (total): {(portfolio_with_signals['Signal'] != portfolio_with_signals['Signal'].shift()).sum()}")
            
            print(f"\n📊 SIGNAL DISTRIBUTION:")
            signal_counts = portfolio_with_signals['Signal'].value_counts().sort_index()
            for signal, count in signal_counts.items():
                signal_name = {0: 'Sell', 1: 'Hold', 2: 'Buy'}.get(signal, 'Unknown')
                print(f"  {signal_name} ({signal}): {count} days ({count/len(portfolio_with_signals)*100:.1f}%)")
                
            # Show first few trades for verification
            if trades:
                print(f"\n💰 FIRST 10 ACTUAL TRADES:")
                for i, trade in enumerate(trades[:10]):
                    print(f"  {i+1}. {trade['date'].strftime('%Y-%m-%d')}: {trade['type'].upper()} (Signal: {trade['signal']}) - Portfolio: {trade['portfolio_value']:.2f}")
        
        return portfolio_with_signals
    
    def calculate_portfolio_metrics(self):
        """
        Calculate key portfolio performance metrics
        """
        if self.portfolio_data is None:
            raise ValueError("Portfolio data not calculated. Run calculate_benchmark_performance first.")
        
        data = self.portfolio_data
        
        # Total return
        total_return = data['Benchmark_Return_Pct'].iloc[-1]
        
        # Time period
        start_date = data['Date'].iloc[0]
        end_date = data['Date'].iloc[-1]
        years = (end_date - start_date).days / 365.25
        
        # Annualized returns
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        
        # Volatility (standard deviation of daily returns)
        portfolio_volatility = data['Daily_Return'].std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 2.0
        sharpe_ratio = (annual_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = data['Benchmark_normalized'].expanding().max()
        drawdown = (data['Benchmark_normalized'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        metrics = {
            'period': {
                'start_date': start_date,
                'end_date': end_date,
                'years': years
            },
            'benchmark': {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': data['Benchmark_normalized'].iloc[-1]
            }
        }
        
        return metrics
    
    def create_portfolio_visualization(self, save_plots=True):
        """
        Create portfolio return comparison visualization (percentage only)
        """
        if self.portfolio_data is None:
            raise ValueError("Portfolio data not calculated. Run calculate_benchmark_performance first.")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure with subplots for comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot 1: Portfolio Return Percentage Comparison
        ax1.plot(self.portfolio_data['Date'], self.portfolio_data['Benchmark_Return_Pct'], 
                linewidth=3, label=f'Benchmark ({self.benchmark_stock})', color='#2E86AB')
        
        if hasattr(self, 'model_portfolio_data'):
            ax1.plot(self.model_portfolio_data['Date'], self.model_portfolio_data['Model_Return_Pct'], 
                    linewidth=3, label='Model Strategy', color='#E74C3C')
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Break-even (0%)')
        
        ax1.set_title(f'Portfolio Return Comparison: {self.benchmark_stock} Benchmark vs Model Strategy (%)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Return (%)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        # Plot 2: Model Trading Signals (Actual Trades Only)
        if hasattr(self, 'model_portfolio_data') and hasattr(self, 'model_trades_data'):
            # Plot stock price as background
            ax2_price = ax2.twinx()
            ax2_price.plot(self.model_portfolio_data['Date'], self.model_portfolio_data['Close'], 
                          color='gray', alpha=0.3, linewidth=1, label=f'{self.benchmark_stock} Price')
            ax2_price.set_ylabel(f'{self.benchmark_stock} Price', fontsize=12, color='gray')
            ax2_price.tick_params(axis='y', labelcolor='gray')
            
            if not self.model_trades_data.empty:
                # Separate actual trades by type
                buy_trades = self.model_trades_data[self.model_trades_data['type'] == 'buy']
                sell_trades = self.model_trades_data[self.model_trades_data['type'] == 'sell']
                
                # Plot actual buy trades (green triangles pointing up)
                if not buy_trades.empty:
                    ax2.scatter(buy_trades['date'], [2] * len(buy_trades), 
                               color='green', marker='^', s=60, alpha=0.9, 
                               label=f'Buy Trades ({len(buy_trades)})', edgecolors='darkgreen', linewidth=1)
                
                # Plot actual sell trades (red triangles pointing down)
                if not sell_trades.empty:
                    ax2.scatter(sell_trades['date'], [0] * len(sell_trades), 
                               color='red', marker='v', s=60, alpha=0.9, 
                               label=f'Sell Trades ({len(sell_trades)})', edgecolors='darkred', linewidth=1)
                
                # Add a horizontal line at hold level for reference
                ax2.axhline(y=1, color='blue', alpha=0.3, linestyle='-', linewidth=2, label='Hold (No Trade)')
                
                ax2.set_ylim(-0.5, 2.5)
                ax2.set_yticks([0, 1, 2])
                ax2.set_yticklabels(['Sell Trade', 'Hold', 'Buy Trade'])
                
                # Add trade count to title
                total_trades = len(buy_trades) + len(sell_trades)
                ax2.set_title(f'Actual Trading Activity - {total_trades} Trades (Buy: {len(buy_trades)}, Sell: {len(sell_trades)})', 
                             fontsize=16, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No trades executed', 
                        transform=ax2.transAxes, ha='center', va='center', fontsize=14)
                ax2.set_title('Actual Trading Activity - No Trades', fontsize=16, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No model data available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
            ax2.set_title('Actual Trading Activity', fontsize=16, fontweight='bold')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Trade Type', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('visualizer/images/portfolio_comparison.png', dpi=300, bbox_inches='tight')
            print("Portfolio comparison saved as 'portfolio_comparison.png'")
        
        plt.show()
        
        return fig
    
    def print_strategy_comparison(self):
        """
        Print comparison between benchmark and model strategies
        """
        if not hasattr(self, 'model_portfolio_data'):
            print("Model strategy not calculated. Run simulate_model_strategy first.")
            return self.print_portfolio_summary()
        
        # Calculate metrics for both strategies
        benchmark_metrics = self.calculate_portfolio_metrics()
        
        # Model strategy metrics
        model_data = self.model_portfolio_data
        model_final_value = model_data['Model_normalized'].iloc[-1]
        model_total_return = model_final_value - 100
        
        # Time period
        start_date = model_data['Date'].iloc[0]
        end_date = model_data['Date'].iloc[-1]
        years = (end_date - start_date).days / 365.25
        
        model_annual_return = ((model_final_value / 100.0) ** (1/years) - 1) * 100
        
        print("\n" + "="*90)
        print("STRATEGY PERFORMANCE COMPARISON")
        print("="*90)
        
        # Period information
        print(f"\nAnalysis Period:")
        print(f"  Start Date: {start_date.strftime('%Y-%m-%d')}")
        print(f"  End Date: {end_date.strftime('%Y-%m-%d')}")
        print(f"  Duration: {years:.1f} years")
        print(f"  Benchmark Stock: {self.benchmark_stock}")
        
        print(f"\n{'Strategy':<20} {'Final Value':<15} {'Total Return':<15} {'Annual Return':<15}")
        print("-" * 65)
        print(f"{'Benchmark':<20} {benchmark_metrics['benchmark']['final_value']:<14.2f} {benchmark_metrics['benchmark']['total_return']:<14.2f}% {benchmark_metrics['benchmark']['annual_return']:<14.2f}%")
        print(f"{'Model Strategy':<20} {model_final_value:<14.2f} {model_total_return:<14.2f}% {model_annual_return:<14.2f}%")
        
        # Performance comparison
        print(f"\n🏆 WINNER ANALYSIS:")
        if model_total_return > benchmark_metrics['benchmark']['total_return']:
            outperformance = model_total_return - benchmark_metrics['benchmark']['total_return']
            print(f"  Model Strategy WINS by {outperformance:.2f} percentage points!")
            print(f"  Model final value: {model_final_value:.2f}")
            print(f"  Benchmark final value: {benchmark_metrics['benchmark']['final_value']:.2f}")
        else:
            underperformance = benchmark_metrics['benchmark']['total_return'] - model_total_return
            print(f"  Benchmark WINS by {underperformance:.2f} percentage points!")
            print(f"  Benchmark final value: {benchmark_metrics['benchmark']['final_value']:.2f}")
            print(f"  Model final value: {model_final_value:.2f}")
        
        # Trading activity analysis
        signals = self.model_portfolio_data['Signal'].dropna()
        signal_changes = (signals != signals.shift()).sum()
        
        print(f"\n📊 MODEL STRATEGY DETAILS:")
        print(f"  Total trading signals generated: {len(signals)}")
        print(f"  Signal changes (trades): {signal_changes}")
        print(f"  Time selling (Signal=0): {(signals == 0).mean()*100:.1f}%")
        print(f"  Time holding (Signal=1): {(signals == 1).mean()*100:.1f}%")
        print(f"  Time buying (Signal=2): {(signals == 2).mean()*100:.1f}%")
        print(f"  Time in market (Hold+Buy): {((signals == 1) | (signals == 2)).mean()*100:.1f}%")
        print(f"  Time in cash (Sell): {(signals == 0).mean()*100:.1f}%")
        
        print("="*90)
        
        return {
            'benchmark': benchmark_metrics,
            'model_strategy': {
                'final_value': model_final_value,
                'total_return': model_total_return,
                'annual_return': model_annual_return,
                'trades': signal_changes,
                'time_selling': (signals == 0).mean()*100,
                'time_holding': (signals == 1).mean()*100,
                'time_buying': (signals == 2).mean()*100,
                'time_in_market': ((signals == 1) | (signals == 2)).mean()*100
            }
        }

def main():
    """
    Main function to run the portfolio visualization with model comparison
    """
    print("🚀 Starting Portfolio Visualization Tool")
    print("📊 Comparing Benchmark vs Model Strategy Performance")
    
    # Initialize visualizer (defaults to OMXS30 as benchmark)
    visualizer = PortfolioVisualizer()
    
    # Load data
    visualizer.load_data()
    
    # Calculate benchmark performance
    print(f"\n📈 Calculating benchmark performance for {visualizer.benchmark_stock}...")
    visualizer.calculate_benchmark_performance()
    
    try:
        # Load the machine learning model (XGBoost or Random Forest)
        print("\n🤖 Loading trained model...")
        visualizer.load_model()
        
        # Simulate model-based trading strategy
        print("📊 Simulating model-based trading strategy...")
        visualizer.simulate_model_strategy(debug=True)
        
        # Print comparison summary
        visualizer.print_strategy_comparison()
        
        # Create comparison visualizations
        print("\n📊 Creating strategy comparison visualizations...")
        visualizer.create_portfolio_visualization(save_plots=True)
        
        print("\n✅ Strategy comparison complete!")
        print("📋 Check 'portfolio_comparison.png' for detailed charts")
        
    except Exception as e:
        print(f"\n⚠️ Could not load model or run model strategy: {str(e)}")
        print("📊 Falling back to benchmark analysis only...")
        
        # Fallback to simple benchmark analysis
        metrics = visualizer.calculate_portfolio_metrics()
        
        print("\n" + "="*80)
        print(f"{visualizer.benchmark_stock} BENCHMARK PERFORMANCE")
        print("="*80)
        print(f"Final Value: {metrics['benchmark']['final_value']:.2f}")
        print(f"Total Return: {metrics['benchmark']['total_return']:+.2f}%")
        print(f"Annualized Return: {metrics['benchmark']['annual_return']:+.2f}%")
        print("="*80)
        
        # Create simple visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(visualizer.portfolio_data['Date'], visualizer.portfolio_data['Benchmark_Return_Pct'], 
                linewidth=3, label=f'{visualizer.benchmark_stock} Benchmark', color='#2E86AB')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.7, 
                   label='Break-even (0%)')
        plt.title(f'{visualizer.benchmark_stock} Returns Over Time (Benchmark Strategy)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        plt.tight_layout()
        plt.savefig('portfolio_value.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📋 Check 'portfolio_value.png' for the chart")

if __name__ == "__main__":
    main()
