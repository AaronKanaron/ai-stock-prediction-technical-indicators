import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import glob
from datetime import datetime
from typing import get_args
sys.path.append('..')
from visualize_portfolio import PortfolioVisualizer
from xgboost_walk_forward import Stocks

st.set_page_config(
    page_title="Portfolio Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stMetric > label {
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_portfolio_data(stock_name, transaction_fee, model_path=None):
    """Load and process portfolio data with caching"""
    try:
        visualizer = PortfolioVisualizer(
            benchmark_stock=stock_name.lower(),
            transaction_fee=transaction_fee
        )
        visualizer.load_data()
        visualizer.calculate_benchmark_performance()

        try:
            visualizer.load_model(model_path=model_path)
            visualizer.simulate_model_strategy(debug=False)
            has_model = True
        except Exception as e:
            st.warning(f"Could not load model: {str(e)}")
            has_model = False

        return visualizer, has_model
    except Exception as e:
        st.error(f"Error loading data for {stock_name}: {str(e)}")
        return None, False

def create_interactive_chart(portfolio_data, model_data=None, trades_data=None, stock_name=""):
    """Create interactive plotly chart"""
    fig = go.Figure()

    # Benchmark line
    fig.add_trace(go.Scatter(
        x=portfolio_data['Date'],
        y=portfolio_data['Benchmark_Return_Pct'],
        mode='lines',
        name=f'Benchmark ({stock_name})',
        line=dict(color='#2E86AB', width=3),
        hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))

    # Model strategy line
    if model_data is not None:
        fig.add_trace(go.Scatter(
            x=model_data['Date'],
            y=model_data['Model_Return_Pct'],
            mode='lines',
            name='AI Model Strategy',
            line=dict(color='#E74C3C', width=3),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))

    # Add buy/sell markers if trades data exists
    if trades_data is not None and not trades_data.empty:
        # Merge trades with model data to get returns at trade points
        trades_with_returns = pd.merge(
            trades_data,
            model_data[['Date', 'Model_Return_Pct']],
            left_on='date',
            right_on='Date',
            how='left'
        )

        buy_trades = trades_with_returns[trades_with_returns['type'] == 'buy']
        sell_trades = trades_with_returns[trades_with_returns['type'] == 'sell']

        if not buy_trades.empty:
            fig.add_trace(go.Scatter(
                x=buy_trades['date'],
                y=buy_trades['Model_Return_Pct'],
                mode='markers',
                name=f'Buy Trades ({len(buy_trades)})',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green',
                    line=dict(color='darkgreen', width=1)
                ),
                hovertemplate='<b>Buy Trade</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ))

        if not sell_trades.empty:
            fig.add_trace(go.Scatter(
                x=sell_trades['date'],
                y=sell_trades['Model_Return_Pct'],
                mode='markers',
                name=f'Sell Trades ({len(sell_trades)})',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red',
                    line=dict(color='darkred', width=1)
                ),
                hovertemplate='<b>Sell Trade</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7)

    fig.update_layout(
        title=f'Portfolio Performance Comparison - {stock_name}',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

def format_performance_table(comparison_data):
    """Format the performance comparison table for Streamlit display"""
    if not comparison_data:
        return None

    benchmark_data = comparison_data['benchmark']['benchmark']
    model_data = comparison_data['model_strategy']

    # Calculate differences
    total_return_diff = benchmark_data['total_return'] - model_data['total_return']
    total_return_diff_pct = (total_return_diff / abs(benchmark_data['total_return']) * 100) if benchmark_data['total_return'] != 0 else 0

    # Create DataFrame for the table
    table_data = {
        'Strategy': ['Benchmark', 'AI Model Strategy'],
        'Total Return': [f"{benchmark_data['total_return']:.2f}%", f"{model_data['total_return']:.2f}%"],
        'Annual Return': [f"{benchmark_data['annual_return']:.2f}%", f"{model_data['annual_return']:.2f}%"],
        'Difference (%p)': [f"{total_return_diff:.2f}%", f"{-total_return_diff:.2f}%"],
        'Difference (%)': [f"{total_return_diff_pct:.2f}%", f"{-total_return_diff_pct:.2f}%"]
    }

    return pd.DataFrame(table_data)

def get_available_models():
    """Get list of available model files"""
    model_files = glob.glob("models/*.pkl")
    if not model_files:
        return []

    # Sort by modification time (newest first)
    model_files.sort(key=os.path.getmtime, reverse=True)

    # Create display names with timestamps
    model_options = []
    for model_path in model_files:
        filename = os.path.basename(model_path)
        # Extract timestamp from filename
        if '_20' in filename:
            timestamp_part = filename.split('_20')[-1].replace('.pkl', '')
            try:
                timestamp = datetime.strptime('20' + timestamp_part, '%Y%m%d_%H%M%S')
                display_name = f"{timestamp.strftime('%Y-%m-%d %H:%M')} - {filename[:30]}..."
            except:
                display_name = filename
        else:
            display_name = filename

        model_options.append((display_name, model_path))

    return model_options

def main():
    st.title("🚀 AI Portfolio Visualizer")
    st.markdown("*Gymnasiearbete: Kan maskininlärning förutsäga svenska aktiekurser?*")

    # Sidebar configuration
    st.sidebar.title("⚙️ Settings")

    # Stock selection
    available_stocks = get_args(Stocks)

    selected_stock = st.sidebar.selectbox(
        "📈 Select Stock/Index:",
        available_stocks,
        index=0
    )

    # Model selection
    model_options = get_available_models()
    if model_options:
        selected_model_display = st.sidebar.selectbox(
            "🤖 Select AI Model:",
            [option[0] for option in model_options],
            index=0
        )
        # Get the actual file path
        selected_model_path = next(option[1] for option in model_options if option[0] == selected_model_display)
    else:
        st.sidebar.warning("No models found in models/ directory")
        selected_model_path = None

    # Transaction fee slider
    transaction_fee = st.sidebar.slider(
        "💸 Transaction Fee (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        format="%.1f%%"
    ) / 100.0  # Convert to decimal

    # Analysis options
    st.sidebar.subheader("📊 Analysis Options")
    show_trades = st.sidebar.checkbox("Show Trade Markers", value=True)
    show_table = st.sidebar.checkbox("Show Performance Table", value=True)
    show_metrics = st.sidebar.checkbox("Show Key Metrics", value=True)

    # Create a unique key for current settings to trigger reload when they change
    settings_key = f"{selected_stock}_{transaction_fee}_{selected_model_path}"

    # Auto-reload data when settings change
    if ('settings_key' not in st.session_state or
        st.session_state.settings_key != settings_key or
        st.sidebar.button("🔄 Manual Reload")):

        with st.spinner(f"Loading data for {selected_stock}..."):
            visualizer, has_model = load_portfolio_data(selected_stock, transaction_fee, selected_model_path)
            if visualizer:
                st.session_state.visualizer = visualizer
                st.session_state.has_model = has_model
                st.session_state.settings_key = settings_key
                st.success(f"Data loaded successfully for {selected_stock}!")
            else:
                st.error("Failed to load data")

    # Display results if data is loaded
    if 'visualizer' in st.session_state:
        visualizer = st.session_state.visualizer
        has_model = st.session_state.has_model

        # Key metrics row
        if show_metrics and has_model:
            st.subheader("📊 Key Performance Metrics")
            comparison_data = visualizer.print_strategy_comparison()

            if comparison_data:
                benchmark_metrics = comparison_data['benchmark']['benchmark']
                model_metrics = comparison_data['model_strategy']

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "📈 Benchmark Total Return",
                        f"{benchmark_metrics['total_return']:.2f}%",
                        delta=None
                    )

                with col2:
                    st.metric(
                        "🤖 AI Model Total Return",
                        f"{model_metrics['total_return']:.2f}%",
                        delta=f"{(model_metrics['total_return'] / benchmark_metrics['total_return'] - 1) * 100:.2f}%"
                    )

                with col3:
                    st.metric(
                        "📅 Annual Return (Benchmark)",
                        f"{benchmark_metrics['annual_return']:.2f}%",
                        delta=None
                    )

                with col4:
                    st.metric(
                        "🔄 Total Trades Made",
                        f"{model_metrics['trades']}",
                        delta=None
                    )

        # Interactive chart
        st.subheader("📈 Portfolio Performance Chart")

        model_data = None
        trades_data = None

        if has_model:
            model_data = visualizer.model_portfolio_data
            trades_data = visualizer.model_trades_data if show_trades else None

        chart = create_interactive_chart(
            visualizer.portfolio_data,
            model_data,
            trades_data,
            selected_stock
        )
        st.plotly_chart(chart, use_container_width=True)

        # Performance comparison table
        if show_table and has_model:
            st.subheader("📋 Performance Comparison Table")
            comparison_data = visualizer.print_strategy_comparison()

            if comparison_data:
                table_df = format_performance_table(comparison_data)
                if table_df is not None:
                    st.dataframe(
                        table_df,
                        use_container_width=True,
                        hide_index=True
                    )

        # Additional insights
        if has_model:
            st.subheader("🔍 Trading Strategy Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Signal Distribution:**")
                signals = visualizer.model_portfolio_data['Signal'].dropna()
                signal_counts = signals.value_counts().sort_index()
                signal_names = {0: 'Sell', 1: 'Hold', 2: 'Buy'}

                for signal, count in signal_counts.items():
                    percentage = count / len(signals) * 100
                    st.write(f"• {signal_names.get(signal, 'Unknown')}: {percentage:.1f}% ({count} days)")

            with col2:
                st.write("**Market Exposure:**")
                time_in_market = ((signals == 1) | (signals == 2)).mean() * 100
                time_in_cash = (signals == 0).mean() * 100

                st.write(f"• Time in Market: {time_in_market:.1f}%")
                st.write(f"• Time in Cash: {time_in_cash:.1f}%")
                st.write(f"• Total Fees Paid: {visualizer.model_total_fees:.2f}")

        # Model information
        # if has_model and selected_model_path:
        #     st.subheader("🤖 Model Information")
        #     model_filename = os.path.basename(selected_model_path)
        #     model_size = os.path.getsize(selected_model_path) / (1024 * 1024)  # Size in MB
        #     model_modified = datetime.fromtimestamp(os.path.getmtime(selected_model_path))

        #     col1, col2, col3 = st.columns(3)
        #     with col1:
        #         st.metric("📄 Model File", model_filename[:20] + "..." if len(model_filename) > 20 else model_filename)
        #     with col2:
        #         st.metric("💾 File Size", f"{model_size:.1f} MB")
        #     with col3:
        #         st.metric("📅 Last Modified", model_modified.strftime("%Y-%m-%d %H:%M"))

        # Download section
        st.subheader("💾 Export Data")
        if st.button("📊 Generate Detailed Report"):
            if has_model:
                # Create a comprehensive report
                report_data = {
                    'settings': {
                        'stock': selected_stock,
                        'transaction_fee': transaction_fee,
                        'model_path': selected_model_path
                    },
                    'portfolio_data': visualizer.portfolio_data.to_dict(),
                    'model_data': visualizer.model_portfolio_data.to_dict() if has_model else None,
                    'comparison': visualizer.print_strategy_comparison() if has_model else None
                }

                st.success("Report generated successfully!")
                st.json(report_data, expanded=False)
    else:
        st.info("👆 Please select a stock and click 'Reload Data' to begin analysis.")

        # Show available stocks info
        st.subheader("📋 Available Stocks and Indices")
        stocks_df = pd.DataFrame({
            'Stock/Index': available_stocks,
            'Type': ['Index' if x in ['OMXS30', 'SP500'] else 'Stock' for x in available_stocks]
        })
        st.dataframe(stocks_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()