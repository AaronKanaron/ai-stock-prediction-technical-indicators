from typing import Literal, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import joblib
from sklearn.preprocessing import StandardScaler

from prettytable import PrettyTable
from xgboost_walk_forward import Stocks


class PortfolioVisualizer:
    def __init__(self, data_path="./data",
                 benchmark_stock: Optional[Stocks] = "OMXS30",
                 transaction_fee: float = 0.001
                 ):
        """
        Initialize the portfolio visualizer with paths to processed data

        Args:
            data_path: Path to data directory
            benchmark_stock: Stock to use as benchmark
            transaction_fee: Transaction fee as percentage (0.001 = 0.1%)
        """
        self.data_path = data_path
        self.benchmark_stock = benchmark_stock
        self.transaction_fee = transaction_fee
        self.benchmark_data = None
        self.portfolio_data = None
        
    def load_data(self):
        """
        Load both processed data (for model features) and raw data (for Close price)
        """
        print(f"Loading {self.benchmark_stock} data...")
        
        # Load processed data (for model predictions)
        processed_path = os.path.join(self.data_path, f"{self.benchmark_stock}.csv")
        self.processed_data = pd.read_csv(processed_path)
        self.processed_data['Date'] = pd.to_datetime(self.processed_data['Date'])
        self.processed_data = self.processed_data.sort_values('Date').reset_index(drop=True)
        
        # Load raw data (for Close price)
        raw_path = os.path.join("rawdata", f"{self.benchmark_stock}_rawdata.csv")
        self.raw_data = pd.read_csv(raw_path)
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        self.raw_data = self.raw_data.sort_values('Date').reset_index(drop=True)
        
        # Merge Close price from raw data with processed data
        self.benchmark_data = pd.merge(
            self.processed_data,
            self.raw_data[['Date', 'Close']],
            on='Date',
            how='left'
        )
        
        print(f"Processed data: {len(self.processed_data)} records")
        print(f"Raw data: {len(self.raw_data)} records") 
        print(f"Combined data: {len(self.benchmark_data)} records from {self.benchmark_data['Date'].min()} to {self.benchmark_data['Date'].max()}")
        
        # Check for missing Close prices
        missing_close = self.benchmark_data['Close'].isna().sum()
        if missing_close > 0:
            print(f"Warning: {missing_close} rows missing Close price - these will be handled during calculations")
        
    def calculate_benchmark_performance(self):
        """
        Calculate buy-and-hold benchmark performance for single stock
        """
        print(f"Calculating benchmark performance for {self.benchmark_stock}")

        # Calculate normalized returns (starting from 100)
        portfolio_data = self.benchmark_data[['Date', 'Close']].copy()
        initial_value = 100.0

        # Apply transaction fee for initial purchase (buy-and-hold has 1 buy transaction)
        transaction_fee_amount = initial_value * self.transaction_fee
        value_after_fee = initial_value - transaction_fee_amount

        # Calculate normalized returns accounting for initial transaction fee
        portfolio_data['Benchmark_normalized'] = (portfolio_data['Close'] / portfolio_data['Close'].iloc[0]) * value_after_fee

        # Calculate percentage returns
        portfolio_data['Benchmark_Return_Pct'] = portfolio_data['Benchmark_normalized'] - 100

        # Calculate daily returns for volatility
        portfolio_data['Daily_Return'] = portfolio_data['Benchmark_normalized'].pct_change() * 100

        # Track total fees paid (just one buy transaction for buy-and-hold)
        self.benchmark_total_fees = transaction_fee_amount

        self.portfolio_data = portfolio_data

        return portfolio_data
    
    def load_model(self, model_path=None):
        """
        Load the trained machine learning model (XGBoost or Random Forest)
        """
        if model_path is None:
            # Try to find XGBoost model
            xgboost_files = [f for f in os.listdir("./models") if f.endswith(".pkl")]
            
            if xgboost_files:
                xgboost_files_with_time = []
                for f in xgboost_files:
                    file_path = os.path.join("./models", f)
                    mtime = os.path.getmtime(file_path)
                    xgboost_files_with_time.append((f, mtime))
                    
                newest_file = sorted(xgboost_files_with_time, key=lambda x: x[1], reverse=True)[0][0]
                model_path = os.path.join("./models", newest_file)
            else:
                raise FileNotFoundError("No XGBoost model found in models/ directory")
        
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        return self.model
    
    def generate_model_predictions(self, data, model):
        """
        Generate model predictions for trading signals using pre-calculated indicators.
        Automatically detects the features the model expects.
        """
        # Try to get feature names from the model
        model_features = None

        # Try different ways to get feature names from the model
        if hasattr(model, 'feature_names_in_'):
            # Sklearn models store feature names here
            model_features = list(model.feature_names_in_)
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
            # XGBoost models
            model_features = model.get_booster().feature_names
        elif hasattr(model, 'feature_name_'):
            # LightGBM models
            model_features = model.feature_name_
        else:
            # Fallback to all available features (minus excluded ones)
            print("Warning: Could not detect model feature names. Using all available features.")
            exclude_columns = ['Date', 'Target', 'dataset_source', 'Close']
            model_features = [col for col in data.columns if col not in exclude_columns]

        if model_features is None or len(model_features) == 0:
            print("Error: Could not determine model features")
            return None

        print(f"Model expects {len(model_features)} features: {model_features[:5]}...")

        # Check which features are actually available in the data
        available_features = [f for f in model_features if f in data.columns]
        missing_features = [f for f in model_features if f not in data.columns]

        if missing_features:
            print(f"Error: Missing required features in data: {missing_features}")
            return None

        if not available_features:
            print("Error: No model features found in data")
            return None

        print(f"Using all {len(available_features)} required features for predictions")
        
        # Prepare features for prediction
        feature_data = data[available_features].dropna()
        
        if feature_data.empty:
            print("Error: No valid feature data after removing NaN values")
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

        print(f"Simulating model-based trading strategy (Transaction fee: {self.transaction_fee*100:.1f}%)...")

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

        # Simulate both with and without fees for comparison
        # First simulate without fees
        no_fee_result = self._simulate_strategy_core(portfolio_with_signals, apply_fees=False, debug=debug)

        # Then simulate with fees
        with_fee_result = self._simulate_strategy_core(portfolio_with_signals, apply_fees=True, debug=debug)

        # Store results
        self.model_portfolio_data = with_fee_result['portfolio_data']
        self.model_trades_data = with_fee_result['trades_data']
        self.model_total_fees = with_fee_result['total_fees']
        self.model_no_fee_return = no_fee_result['final_return']
        self.model_with_fee_return = with_fee_result['final_return']

        return self.model_portfolio_data

    def _simulate_strategy_core(self, portfolio_with_signals, apply_fees=True, debug=False):
        """
        Core simulation logic for trading strategy
        """
        # Initialize model portfolio tracking (normalized to 100 at start)
        portfolio_value = 100.0  # Start with value of 100
        in_market = False  # Track if we're in the market or in cash
        model_values = []
        total_fees_paid = 0.0  # Track total transaction fees

        # Get actual stock prices for calculating daily returns
        stock_prices = portfolio_with_signals['Close'].values

        # Debug tracking
        trades = []
        
        for i, row in portfolio_with_signals.iterrows():
            current_signal = row['Signal']
            date = row['Date']

            if pd.isna(current_signal):
                current_signal = 1  # Default to hold position

            # Update portfolio value FIRST based on current position
            if in_market:
                # If in market, apply daily stock price changes
                if i > 0 and not pd.isna(stock_prices[i]) and not pd.isna(stock_prices[i-1]) and stock_prices[i-1] != 0:
                    daily_return = stock_prices[i] / stock_prices[i-1]
                    portfolio_value = portfolio_value * daily_return
            # If in cash (not in_market), portfolio_value stays the same

            trade_executed = False
            trade_type = "none"

            # Then execute trades based on signals for NEXT period
            if current_signal == 2:  # Buy signal
                if not in_market:
                    if apply_fees:
                        # Apply transaction fee for buying (fee on the transaction amount)
                        fee_amount = portfolio_value * self.transaction_fee
                        portfolio_value -= fee_amount
                        total_fees_paid += fee_amount
                    else:
                        fee_amount = 0.0

                    # Move from cash to market - start tracking market performance
                    in_market = True
                    trade_executed = True
                    trade_type = "buy"

                    if debug and len(trades) < 20:  # Show first 20 trades for debugging
                        fee_text = f", Fee: {fee_amount:.2f}" if apply_fees else ""
                        print(f"{date.strftime('%Y-%m-%d')}: BUY - Value: {portfolio_value:.2f}{fee_text}")

            elif current_signal == 0:  # Sell signal
                if in_market:
                    if apply_fees:
                        # Apply transaction fee for selling (fee on the transaction amount)
                        fee_amount = portfolio_value * self.transaction_fee
                        portfolio_value -= fee_amount
                        total_fees_paid += fee_amount
                    else:
                        fee_amount = 0.0

                    # Move from market to cash - stop tracking market performance
                    in_market = False
                    trade_executed = True
                    trade_type = "sell"

                    if debug and len(trades) < 20:  # Show first 20 trades for debugging
                        fee_text = f", Fee: {fee_amount:.2f}" if apply_fees else ""
                        print(f"{date.strftime('%Y-%m-%d')}: SELL - Value: {portfolio_value:.2f}{fee_text}")

            model_values.append(portfolio_value)
            
            if trade_executed:
                trades.append({
                    'date': date,
                    'type': trade_type,
                    'signal': current_signal,
                    'portfolio_value': portfolio_value
                })
            
        
        # Add model results to portfolio data
        portfolio_with_signals_copy = portfolio_with_signals.copy()
        portfolio_with_signals_copy['Model_normalized'] = model_values
        portfolio_with_signals_copy['Model_Return_Pct'] = np.array(model_values) - 100

        # Create a separate dataframe for actual trades only
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        final_return = model_values[-1] - 100 if model_values else 0.0

        # Print debugging info
        if debug:
            print(f"\n🔍 TRADING ANALYSIS:")
            print(f"  Total trades executed: {len(trades)}")
            print(f"  Total transaction fees paid: {total_fees_paid:.2f}")
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

        return {
            'portfolio_data': portfolio_with_signals_copy,
            'trades_data': trades_df,
            'total_fees': total_fees_paid,
            'final_return': final_return
        }
    
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
        
        ax1.set_title(f'Portfolio Return Comparison: {self.benchmark_stock.capitalize()} Benchmark vs Model Strategy (%)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Return (%)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        
        # Plot 2: Model Trading Signals (Actual Trades Only)
        if hasattr(self, 'model_portfolio_data') and hasattr(self, 'model_trades_data'):
            # Plot stock price as main chart
            ax2.plot(self.model_portfolio_data['Date'], self.model_portfolio_data['Close'], 
                    color='gray', alpha=0.7, linewidth=1, label=f'{self.benchmark_stock} Price')
            
            if not self.model_trades_data.empty:
                # Merge trades with model portfolio data to get stock prices at trade dates
                trades_with_prices = pd.merge(
                    self.model_trades_data,
                    self.model_portfolio_data[['Date', 'Close']],
                    left_on='date',
                    right_on='Date',
                    how='left'
                )
                
                # Separate actual trades by type
                buy_trades = trades_with_prices[trades_with_prices['type'] == 'buy']
                sell_trades = trades_with_prices[trades_with_prices['type'] == 'sell']
                
                # Plot actual buy trades (green triangles pointing up) on stock price
                if not buy_trades.empty:
                    ax2.scatter(buy_trades['date'], buy_trades['Close'], 
                               color='green', marker='^', s=50, alpha=1.0, 
                               label=f'Buy Trades ({len(buy_trades)})', edgecolors='darkgreen', linewidth=1, zorder=5)
                
                # Plot actual sell trades (red triangles pointing down) on stock price
                if not sell_trades.empty:
                    ax2.scatter(sell_trades['date'], sell_trades['Close'], 
                               color='red', marker='v', s=50, alpha=1.0, 
                               label=f'Sell Trades ({len(sell_trades)})', edgecolors='darkred', linewidth=1, zorder=5)
                
                # Add trade count to title
                total_trades = len(buy_trades) + len(sell_trades)
                ax2.set_title(f'Trading Signals - {total_trades} Trades (Buy: {len(buy_trades)}, Sell: {len(sell_trades)})', 
                             fontsize=16, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No trades executed', 
                        transform=ax2.transAxes, ha='center', va='center', fontsize=14)
                ax2.set_title('Trading Signals - No Trades', fontsize=16, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No model data available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
            ax2.set_title('Trading Signals', fontsize=16, fontweight='bold')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Stock Price', fontsize=12)
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
        print(" "*20 + f"{self.benchmark_stock.capitalize()} Strategy Comparison")
        print("="*90)
        
        # Period information
        print(f"\nAnalysis Period:")
        print(f"  Start Date: {start_date.strftime('%Y-%m-%d')}")
        print(f"  End Date: {end_date.strftime('%Y-%m-%d')}")
        print(f"  Duration: {years:.1f} years")
        print(f"  Benchmark Stock: {self.benchmark_stock}\n")
        
        signals = self.model_portfolio_data['Signal'].dropna()
        signal_changes = (signals != signals.shift()).sum()
        
        # Trading activity analysis
        print(f"\nAI Model Strategy Details:")
        label_width = 45 
        num_width   = 8

        print(f"{'  Total trading signals generated:':<{label_width}} {len(signals):>{num_width}}")
        print(f"{'  Signal changes (actual trades made):':<{label_width}} {signal_changes:>{num_width}}")
        print(f"{'  Days selling (Signal=0):':<{label_width}} {(signals == 0).mean()*100:>{num_width}.1f}%")
        print(f"{'  Days holding (Signal=1):':<{label_width}} {(signals == 1).mean()*100:>{num_width}.1f}%")
        print(f"{'  Days buying (Signal=2):':<{label_width}} {(signals == 2).mean()*100:>{num_width}.1f}%")
        print(f"{'  Days in market (Hold+Buy):':<{label_width}} {((signals == 1)|(signals == 2)).mean()*100:>{num_width}.1f}%")
        print(f"{'  Days holding cash (Sell):':<{label_width}} {(signals == 0).mean()*100:>{num_width}.1f}%")

        # Calculate fee loss percentages (what percentage of return was lost to fees)
        benchmark_fee_loss_pct = (self.benchmark_total_fees / 100.0) * 100  # Simple: fee amount as % of initial investment

        # For model strategy: compare return without fees vs with fees
        if hasattr(self, 'model_no_fee_return') and hasattr(self, 'model_with_fee_return'):
            fee_impact = self.model_no_fee_return - self.model_with_fee_return
            model_fee_loss_pct = (fee_impact / 100.0) * 100  # Fee impact as % of initial investment
        else:
            model_fee_loss_pct = (self.model_total_fees / 100.0) * 100  # Fallback to simple calculation

        table = PrettyTable(["Strategy", "Total Return", "Annual Return", "Fee Loss %", "Difference (%p)", "Difference (%)"])
        table.add_row([
            "Benchmark",
            f"{benchmark_metrics['benchmark']['total_return']:.2f}%",
            f"{benchmark_metrics['benchmark']['annual_return']:.2f}%",
            f"{benchmark_fee_loss_pct:.2f}%",
            f"{benchmark_metrics['benchmark']['total_return'] - model_total_return:.2f}%",
            f"{(benchmark_metrics['benchmark']['total_return'] - model_total_return)/abs(benchmark_metrics['benchmark']['total_return'])*100 if benchmark_metrics['benchmark']['total_return'] != 0 else 0:.2f}%"
        ])
        table.add_row([
            "Model Strategy",
            f"{model_total_return:.2f}%",
            f"{model_annual_return:.2f}%",
            f"{model_fee_loss_pct:.2f}%",
            f"{model_total_return - benchmark_metrics['benchmark']['total_return']:.2f}%",
            f"{(model_total_return - benchmark_metrics['benchmark']['total_return'])/abs(benchmark_metrics['benchmark']['total_return'])*100 if benchmark_metrics['benchmark']['total_return'] != 0 else 0:.2f}%"
        ])
        
        table.align["Strategy"] = "l"
        table.align["Total Return"] = "r"
        table.align["Annual Return"] = "r"
        table.align["Fee Loss %"] = "r"
        table.align["Difference (%p)"] = "r"
        table.align["Difference (%)"] = "r"

        print("\n" + str(table) + "\n")        
        print("="*90)
        
        # Calculate actual trade count from trades data
        actual_trades_count = 0
        if hasattr(self, 'model_trades_data') and not self.model_trades_data.empty:
            actual_trades_count = len(self.model_trades_data)

        return {
            'benchmark': benchmark_metrics,
            'model_strategy': {
                'final_value': model_final_value,
                'total_return': model_total_return,
                'annual_return': model_annual_return,
                'trades': actual_trades_count,
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
    print("\n"*5+"🚀 Starting Portfolio Visualization Tool")

    visualizer = PortfolioVisualizer(benchmark_stock="omxs30", transaction_fee=0.0)
    
    visualizer.load_data()
    visualizer.calculate_benchmark_performance()
    
    try:
        visualizer.load_model()
        visualizer.simulate_model_strategy(debug=False)        
        visualizer.print_strategy_comparison()        
        visualizer.create_portfolio_visualization(save_plots=True)
                
    except Exception as e:
        print(f"\n⚠️ Could not load model or run model strategy: {str(e)}")

if __name__ == "__main__":
    main()
