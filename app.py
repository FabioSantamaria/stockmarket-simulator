import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Portfolio Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class PortfolioSimulator:
    def __init__(self):
        self.portfolio_data = None
        self.historical_data = None
        
    def fetch_stock_data(self, tickers, start_date, end_date):
        """Fetch historical stock data for given tickers"""
        try:
            data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[ticker] = hist['Close']
                else:
                    st.warning(f"No data found for {ticker}")
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_portfolio_performance(self, stock_data, weights):
        """Calculate weighted portfolio performance"""
        if stock_data is None or stock_data.empty:
            return None
            
        # Calculate daily returns
        returns = stock_data.pct_change().dropna()
        
        # Calculate weighted portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        return {
            'daily_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'stock_data': stock_data,
            'individual_returns': returns
        }
    
    def calculate_risk_metrics(self, portfolio_returns):
        """Calculate various risk metrics"""
        if portfolio_returns is None or len(portfolio_returns) == 0:
            return None
            
        # Annualized metrics (assuming 252 trading days)
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'total_return': (cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0
        }
    
    def monte_carlo_simulation(self, portfolio_returns, days_ahead=252, num_simulations=1000):
        """Run Monte Carlo simulation for future performance"""
        if portfolio_returns is None or len(portfolio_returns) == 0:
            return None
            
        # Calculate historical mean and std
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Run simulations
        simulations = []
        for _ in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, days_ahead)
            # Calculate cumulative performance
            cumulative_performance = (1 + random_returns).cumprod()
            simulations.append(cumulative_performance)
        
        simulations_df = pd.DataFrame(simulations).T
        
        # Calculate percentiles
        percentiles = {
            '5th': simulations_df.quantile(0.05, axis=1),
            '25th': simulations_df.quantile(0.25, axis=1),
            '50th': simulations_df.quantile(0.50, axis=1),
            '75th': simulations_df.quantile(0.75, axis=1),
            '95th': simulations_df.quantile(0.95, axis=1)
        }
        
        return {
            'simulations': simulations_df,
            'percentiles': percentiles,
            'final_values': simulations_df.iloc[-1]
        }

def main():
    st.markdown('<h1 class="main-header">📈 Stock Portfolio Simulator</h1>', unsafe_allow_html=True)
    
    simulator = PortfolioSimulator()
    
    # Sidebar for portfolio input
    st.sidebar.header("Portfolio Configuration")
    
    # Number of stocks
    num_stocks = st.sidebar.number_input("Number of stocks in portfolio", min_value=1, max_value=10, value=3)
    
    # Stock input
    tickers = []
    weights = []
    
    st.sidebar.subheader("Stock Tickers and Weights")
    for i in range(num_stocks):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            ticker = st.text_input(f"Stock {i+1}", value=f"{'AAPL' if i==0 else 'GOOGL' if i==1 else 'MSFT' if i==2 else ''}", key=f"ticker_{i}")
        with col2:
            weight = st.number_input(f"Weight {i+1} (%)", min_value=0.0, max_value=100.0, value=100.0/num_stocks, key=f"weight_{i}")
        
        if ticker:
            tickers.append(ticker.upper())
            weights.append(weight/100)
    
    # Normalize weights
    if weights and sum(weights) > 0:
        weights = [w/sum(weights) for w in weights]
    
    # Date range selection
    st.sidebar.subheader("Analysis Period")
    end_date = datetime.now()
    start_date = st.sidebar.date_input("Start Date", value=end_date - timedelta(days=365*2))
    end_date = st.sidebar.date_input("End Date", value=end_date)
    
    # Monte Carlo parameters
    st.sidebar.subheader("Monte Carlo Simulation")
    forecast_days = st.sidebar.slider("Forecast Days", min_value=30, max_value=365*2, value=252)
    num_simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=5000, value=1000)
    
    # Analyze button
    if st.sidebar.button("Analyze Portfolio", type="primary"):
        if not tickers:
            st.error("Please enter at least one stock ticker.")
            return
        
        if abs(sum(weights) - 1.0) > 0.01:
            st.error("Portfolio weights must sum to 100%.")
            return
        
        # Show portfolio composition
        st.subheader("Portfolio Composition")
        portfolio_df = pd.DataFrame({
            'Stock': tickers,
            'Weight (%)': [w*100 for w in weights]
        })
        st.dataframe(portfolio_df, use_container_width=True)
        
        # Fetch data and analyze
        with st.spinner("Fetching stock data..."):
            stock_data = simulator.fetch_stock_data(tickers, start_date, end_date)
        
        if stock_data is not None and not stock_data.empty:
            # Calculate portfolio performance
            with st.spinner("Calculating portfolio performance..."):
                performance = simulator.calculate_portfolio_performance(stock_data, weights)
            
            if performance:
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 Visualization", "⚠️ Risk Metrics", "🔮 Forecast"])
                
                with tab1:
                    st.subheader("Historical Performance")
                    
                    # Key metrics
                    risk_metrics = simulator.calculate_risk_metrics(performance['daily_returns'])
                    
                    if risk_metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Return", f"{risk_metrics['total_return']:.2%}")
                        with col2:
                            st.metric("Annual Return", f"{risk_metrics['annual_return']:.2%}")
                        with col3:
                            st.metric("Annual Volatility", f"{risk_metrics['annual_volatility']:.2%}")
                        with col4:
                            st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
                    
                    # Performance table
                    st.subheader("Individual Stock Performance")
                    stock_performance = []
                    for ticker in tickers:
                        if ticker in stock_data.columns:
                            start_price = stock_data[ticker].iloc[0]
                            end_price = stock_data[ticker].iloc[-1]
                            total_return = (end_price - start_price) / start_price
                            stock_performance.append({
                                'Stock': ticker,
                                'Start Price': f"${start_price:.2f}",
                                'End Price': f"${end_price:.2f}",
                                'Total Return': f"{total_return:.2%}"
                            })
                    
                    st.dataframe(pd.DataFrame(stock_performance), use_container_width=True)
                
                with tab2:
                    st.subheader("Portfolio Performance Visualization")
                    
                    # Cumulative returns chart
                    fig = go.Figure()
                    
                    # Portfolio performance
                    fig.add_trace(go.Scatter(
                        x=performance['cumulative_returns'].index,
                        y=performance['cumulative_returns'].values,
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Individual stock performance
                    for ticker in tickers:
                        if ticker in stock_data.columns:
                            normalized_stock = stock_data[ticker] / stock_data[ticker].iloc[0]
                            fig.add_trace(go.Scatter(
                                x=normalized_stock.index,
                                y=normalized_stock.values,
                                mode='lines',
                                name=ticker,
                                opacity=0.7
                            ))
                    
                    fig.update_layout(
                        title="Cumulative Returns Comparison",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Daily returns distribution
                    st.subheader("Daily Returns Distribution")
                    fig_hist = px.histogram(
                        x=performance['daily_returns'],
                        nbins=50,
                        title="Portfolio Daily Returns Distribution"
                    )
                    fig_hist.update_layout(xaxis_title="Daily Return", yaxis_title="Frequency")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with tab3:
                    st.subheader("Risk Analysis")
                    
                    if risk_metrics:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Value at Risk (95%)", f"{risk_metrics['var_95']:.2%}")
                            st.metric("Maximum Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
                        
                        with col2:
                            # Risk interpretation
                            st.markdown("**Risk Interpretation:**")
                            if risk_metrics['annual_volatility'] < 0.15:
                                st.success("Low Risk Portfolio")
                            elif risk_metrics['annual_volatility'] < 0.25:
                                st.warning("Medium Risk Portfolio")
                            else:
                                st.error("High Risk Portfolio")
                            
                            if risk_metrics['sharpe_ratio'] > 1:
                                st.success("Excellent Risk-Adjusted Returns")
                            elif risk_metrics['sharpe_ratio'] > 0.5:
                                st.info("Good Risk-Adjusted Returns")
                            else:
                                st.warning("Poor Risk-Adjusted Returns")
                    
                    # Rolling volatility chart
                    st.subheader("Rolling 30-Day Volatility")
                    rolling_vol = performance['daily_returns'].rolling(window=30).std() * np.sqrt(252)
                    
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol.values,
                        mode='lines',
                        name='30-Day Rolling Volatility',
                        line=dict(color='red')
                    ))
                    
                    fig_vol.update_layout(
                        title="Portfolio Volatility Over Time",
                        xaxis_title="Date",
                        yaxis_title="Annualized Volatility",
                        height=400
                    )
                    
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                with tab4:
                    st.subheader("Monte Carlo Forecast")
                    
                    with st.spinner("Running Monte Carlo simulation..."):
                        mc_results = simulator.monte_carlo_simulation(
                            performance['daily_returns'], 
                            days_ahead=forecast_days, 
                            num_simulations=num_simulations
                        )
                    
                    if mc_results:
                        # Forecast chart
                        fig_mc = go.Figure()
                        
                        # Add percentile bands
                        days = list(range(forecast_days))
                        
                        fig_mc.add_trace(go.Scatter(
                            x=days + days[::-1],
                            y=list(mc_results['percentiles']['95th']) + list(mc_results['percentiles']['5th'][::-1]),
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='90% Confidence Interval'
                        ))
                        
                        fig_mc.add_trace(go.Scatter(
                            x=days,
                            y=mc_results['percentiles']['50th'],
                            mode='lines',
                            name='Median Forecast',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_mc.add_trace(go.Scatter(
                            x=days,
                            y=mc_results['percentiles']['25th'],
                            mode='lines',
                            name='25th Percentile',
                            line=dict(color='orange', dash='dash')
                        ))
                        
                        fig_mc.add_trace(go.Scatter(
                            x=days,
                            y=mc_results['percentiles']['75th'],
                            mode='lines',
                            name='75th Percentile',
                            line=dict(color='green', dash='dash')
                        ))
                        
                        fig_mc.update_layout(
                            title=f"Monte Carlo Forecast ({forecast_days} days, {num_simulations} simulations)",
                            xaxis_title="Days Ahead",
                            yaxis_title="Portfolio Value (Normalized)",
                            height=500
                        )
                        
                        st.plotly_chart(fig_mc, use_container_width=True)
                        
                        # Forecast statistics
                        st.subheader("Forecast Statistics")
                        final_values = mc_results['final_values']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Expected Return", f"{(final_values.mean() - 1):.2%}")
                        with col2:
                            st.metric("Best Case (95th %ile)", f"{(final_values.quantile(0.95) - 1):.2%}")
                        with col3:
                            st.metric("Worst Case (5th %ile)", f"{(final_values.quantile(0.05) - 1):.2%}")
                        with col4:
                            prob_positive = (final_values > 1).mean()
                            st.metric("Probability of Gain", f"{prob_positive:.1%}")
                        
                        # Final value distribution
                        st.subheader("Final Value Distribution")
                        fig_final = px.histogram(
                            x=final_values,
                            nbins=50,
                            title=f"Distribution of Portfolio Values after {forecast_days} days"
                        )
                        fig_final.update_layout(
                            xaxis_title="Final Portfolio Value (Normalized)",
                            yaxis_title="Frequency"
                        )
                        st.plotly_chart(fig_final, use_container_width=True)
        else:
            st.error("Unable to fetch stock data. Please check your ticker symbols and try again.")
    
    # Information section
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        **Stock Portfolio Simulator** helps you analyze and forecast portfolio performance:
        
        1. **Portfolio Setup**: Enter stock tickers (e.g., AAPL, GOOGL, MSFT) and their weights
        2. **Historical Analysis**: View past performance, returns, and risk metrics
        3. **Risk Assessment**: Understand volatility, Value at Risk, and drawdowns
        4. **Future Forecasting**: Use Monte Carlo simulation to project potential outcomes
        
        **Key Metrics Explained**:
        - **Sharpe Ratio**: Risk-adjusted returns (higher is better)
        - **Value at Risk (VaR)**: Potential loss in worst 5% of cases
        - **Maximum Drawdown**: Largest peak-to-trough decline
        - **Monte Carlo**: Statistical method using random sampling to model uncertainty
        
        **Note**: This is for educational purposes only and should not be considered investment advice.
        """)

if __name__ == "__main__":
    main()