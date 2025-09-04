import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class UIComponents:
    """UI components and interface logic for the Streamlit app"""
    
    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Stock Portfolio Simulator",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styling"""
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
            .ticker-search-result {
                padding: 0.5rem;
                margin: 0.2rem 0;
                border: 1px solid #ddd;
                border-radius: 0.3rem;
                cursor: pointer;
            }
            .ticker-search-result:hover {
                background-color: #f0f2f6;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render the main header"""
        st.markdown('<h1 class="main-header">📈 Stock Portfolio Simulator</h1>', unsafe_allow_html=True)
    
    @staticmethod
    def render_ticker_search(simulator):
        """Render ticker search functionality"""
        st.sidebar.subheader("🔍 Search & Add Stocks")
        
        # Search input
        search_query = st.sidebar.text_input(
            "Search for stocks (company name or ticker)",
            placeholder="e.g., Apple, AAPL, Microsoft"
        )
        
        # Search results
        if search_query:
            search_results = simulator.search_ticker(search_query)
            
            if search_results:
                st.sidebar.write("**Search Results:**")
                
                # Create a container for search results
                for i, result in enumerate(search_results):
                    col1, col2 = st.sidebar.columns([3, 1])
                    
                    # Handle both old and new API response formats
                    ticker = result.get('symbol', result.get('ticker', ''))
                    name = result.get('name', ticker)
                    
                    with col1:
                        st.write(f"**{ticker}** - {name}")
                    
                    with col2:
                        # Use unique key for each button
                        if st.button("➕", key=f"add_{ticker}_{i}", help=f"Add {ticker} to portfolio"):
                            # Add to session state portfolio
                            if 'portfolio_tickers' not in st.session_state:
                                st.session_state.portfolio_tickers = []
                            if 'portfolio_weights' not in st.session_state:
                                st.session_state.portfolio_weights = []
                            
                            if ticker not in st.session_state.portfolio_tickers:
                                st.session_state.portfolio_tickers.append(ticker)
                                # Add equal weight for new stock
                                current_count = len(st.session_state.portfolio_tickers)
                                equal_weight = 100.0 / current_count
                                
                                # Adjust existing weights
                                st.session_state.portfolio_weights = [equal_weight] * current_count
                                
                                st.sidebar.success(f"Added {ticker} to portfolio!")
                                st.rerun()
                            else:
                                st.sidebar.warning(f"{ticker} is already in your portfolio")
            else:
                st.sidebar.write("No results found. Try searching for company names or ticker symbols.")
    
    @staticmethod
    def render_portfolio_input():
        """Render portfolio composition input"""
        st.sidebar.subheader("📊 Portfolio Composition")
        
        # Initialize session state if not exists
        if 'portfolio_tickers' not in st.session_state:
            st.session_state.portfolio_tickers = ['AAPL', 'GOOGL', 'MSFT']
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = [33.33, 33.33, 33.34]
        
        # Number of stocks
        num_stocks = len(st.session_state.portfolio_tickers)
        
        # Allow adding more stocks manually
        if st.sidebar.button("➕ Add Stock Manually"):
            st.session_state.portfolio_tickers.append('')
            # Redistribute weights equally
            equal_weight = 100.0 / len(st.session_state.portfolio_tickers)
            st.session_state.portfolio_weights = [equal_weight] * len(st.session_state.portfolio_tickers)
            st.rerun()
        
        # Stock input fields
        tickers = []
        weights = []
        
        for i in range(len(st.session_state.portfolio_tickers)):
            col1, col2, col3 = st.sidebar.columns([3, 2, 1])
            
            with col1:
                ticker = st.text_input(
                    f"Stock {i+1}", 
                    value=st.session_state.portfolio_tickers[i], 
                    key=f"ticker_input_{i}"
                )
                st.session_state.portfolio_tickers[i] = ticker.upper() if ticker else ''
            
            with col2:
                weight = st.number_input(
                    f"Weight {i+1} (%)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=st.session_state.portfolio_weights[i] if i < len(st.session_state.portfolio_weights) else 0.0,
                    key=f"weight_input_{i}"
                )
                if i < len(st.session_state.portfolio_weights):
                    st.session_state.portfolio_weights[i] = weight
                else:
                    st.session_state.portfolio_weights.append(weight)
            
            with col3:
                if len(st.session_state.portfolio_tickers) > 1:
                    if st.button("🗑️", key=f"remove_{i}", help=f"Remove stock {i+1}"):
                        st.session_state.portfolio_tickers.pop(i)
                        if i < len(st.session_state.portfolio_weights):
                            st.session_state.portfolio_weights.pop(i)
                        st.rerun()
            
            if ticker:
                tickers.append(ticker.upper())
                weights.append(weight)
        
        # Normalize weights button
        if st.sidebar.button("⚖️ Normalize Weights"):
            # Check if weights exist and are valid
            if "portfolio_weights" in st.session_state and all(isinstance(w, (int, float)) for w in st.session_state.portfolio_weights):
                total_weight = sum(st.session_state.portfolio_weights)

                # Normalize only if the total is positive
                if total_weight > 0:
                    st.session_state.portfolio_weights = [
                        (w / total_weight) * 100 for w in st.session_state.portfolio_weights
                    ]
                    
                    # The st.rerun() call is crucial here to force the UI to update with the new weights
                    st.rerun()
        
        # Filter out empty tickers
        valid_tickers = [t for t in tickers if t]
        valid_weights = [w for t, w in zip(tickers, weights) if t]
        
        # Normalize weights
        if valid_weights and sum(valid_weights) > 0:
            valid_weights = [w/sum(valid_weights) * 100 for w in valid_weights]
        
        return valid_tickers, valid_weights
    
    @staticmethod
    def render_portfolio_input_main():
        """Render portfolio composition input in main area (not sidebar)"""
        st.subheader("📊 Portfolio Composition")
        
        # Initialize session state if not exists
        if 'portfolio_tickers' not in st.session_state:
            st.session_state.portfolio_tickers = ['AAPL', 'GOOGL', 'MSFT']
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = [33.33, 33.33, 33.34]
        
        # Stock input fields
        tickers = []
        weights = []
        
        for i in range(len(st.session_state.portfolio_tickers)):
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                ticker = st.text_input(
                    f"Stock {i+1}", 
                    value=st.session_state.portfolio_tickers[i], 
                    key=f"main_ticker_input_{i}"
                )
                st.session_state.portfolio_tickers[i] = ticker.upper() if ticker else ''
            
            with col2:
                weight = st.number_input(
                    f"Weight {i+1} (%)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=st.session_state.portfolio_weights[i] if i < len(st.session_state.portfolio_weights) else 0.0,
                    key=f"main_weight_input_{i}"
                )
                if i < len(st.session_state.portfolio_weights):
                    st.session_state.portfolio_weights[i] = weight
                else:
                    st.session_state.portfolio_weights.append(weight)
            
            with col3:
                if len(st.session_state.portfolio_tickers) > 1:
                    if st.button("🗑️", key=f"main_remove_{i}", help=f"Remove stock {i+1}"):
                        st.session_state.portfolio_tickers.pop(i)
                        if i < len(st.session_state.portfolio_weights):
                            st.session_state.portfolio_weights.pop(i)
                        st.rerun()
            
            if ticker:
                tickers.append(ticker.upper())
                weights.append(weight)
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Stock Manually"):
                st.session_state.portfolio_tickers.append('')
                # Redistribute weights equally
                equal_weight = 100.0 / len(st.session_state.portfolio_tickers)
                st.session_state.portfolio_weights = [equal_weight] * len(st.session_state.portfolio_tickers)
                st.rerun()
        
        with col2:
            if st.button("⚖️ Normalize Weights"):
                # Check if weights exist and are valid
                if "portfolio_weights" in st.session_state and all(isinstance(w, (int, float)) for w in st.session_state.portfolio_weights):
                    total_weight = sum(st.session_state.portfolio_weights)
                    # Normalize only if the total is positive
                    if total_weight > 0:
                        st.session_state.portfolio_weights = [
                            (w / total_weight) * 100 for w in st.session_state.portfolio_weights
                        ]
                        st.rerun()
        
        # Filter out empty tickers
        valid_tickers = [t for t in tickers if t]
        valid_weights = [w for t, w in zip(tickers, weights) if t]
        
        # Normalize weights
        if valid_weights and sum(valid_weights) > 0:
            valid_weights = [w/sum(valid_weights) * 100 for w in valid_weights]
        
        return valid_tickers, valid_weights
    
    @staticmethod
    def render_stock_discovery(data_fetcher, simulator):
        """Render stock discovery section with sector search and trending stocks"""
        st.subheader("🔍 Stock Discovery")
        
        # Ticker search functionality
        search_query = st.text_input("🔍 Search for stocks", placeholder="Enter company name or ticker symbol...")
        
        if search_query:
            with st.spinner("Searching for stocks..."):
                results = simulator.search_ticker(search_query)
            
            if results:
                st.write(f"Found {len(results)} result(s):")
                for result in results[:5]:  # Limit to top 5 results
                    ticker = result.get('symbol', result.get('ticker', ''))
                    name = result.get('name', result.get('longName', 'N/A'))
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{ticker}** - {name}")
                    with col2:
                        if st.button(f"Add {ticker}", key=f"discovery_add_{ticker}"):
                            if ticker not in st.session_state.portfolio_tickers:
                                st.session_state.portfolio_tickers.append(ticker)
                                # Add equal weight for new stock
                                current_count = len(st.session_state.portfolio_tickers)
                                equal_weight = 100.0 / current_count
                                # Adjust existing weights
                                st.session_state.portfolio_weights = [equal_weight] * current_count
                                st.success(f"Added {ticker} to portfolio!")
                                st.rerun()
                            else:
                                st.warning(f"{ticker} is already in your portfolio")
            else:
                st.write("No results found. Try searching for company names or ticker symbols.")
        
        # Sector-based discovery
        st.markdown("---")
        st.write("**Browse by Sector:**")
        
        sectors = {
            "Technology": ["AAPL", "GOOGL", "MSFT", "NVDA", "META"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Finance": ["JPM", "BAC", "WFC", "GS", "MS"],
            "Consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB"]
        }
        
        selected_sector = st.selectbox("Select a sector:", list(sectors.keys()))
        
        if selected_sector:
            st.write(f"Popular {selected_sector} stocks:")
            sector_stocks = sectors[selected_sector]
            
            cols = st.columns(len(sector_stocks))
            for i, stock in enumerate(sector_stocks):
                with cols[i]:
                    if st.button(f"Add {stock}", key=f"sector_add_{stock}"):
                        if stock not in st.session_state.portfolio_tickers:
                            st.session_state.portfolio_tickers.append(stock)
                            # Add equal weight for new stock
                            current_count = len(st.session_state.portfolio_tickers)
                            equal_weight = 100.0 / current_count
                            # Adjust existing weights
                            st.session_state.portfolio_weights = [equal_weight] * current_count
                            st.success(f"Added {stock} to portfolio!")
                            st.rerun()
                        else:
                            st.warning(f"{stock} is already in your portfolio")
        
        # Trending stocks
        st.markdown("---")
        st.write("**Trending Stocks:**")
        
        trending_stocks = ["AAPL", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "MSFT", "NFLX"]
        
        cols = st.columns(4)
        for i, stock in enumerate(trending_stocks):
            with cols[i % 4]:
                if st.button(f"Add {stock}", key=f"trending_add_{stock}"):
                    if stock not in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers.append(stock)
                        # Add equal weight for new stock
                        current_count = len(st.session_state.portfolio_tickers)
                        equal_weight = 100.0 / current_count
                        # Adjust existing weights
                        st.session_state.portfolio_weights = [equal_weight] * current_count
                        st.success(f"Added {stock} to portfolio!")
                        st.rerun()
                    else:
                        st.warning(f"{stock} is already in your portfolio")
    
    @staticmethod
    def render_analysis_parameters():
        """Render analysis parameters input"""
        st.sidebar.subheader("📅 Analysis Parameters")
        
        # Date range selection
        end_date = datetime.now()
        start_date = st.sidebar.date_input(
            "Start Date", 
            value=end_date - timedelta(days=365*2)
        )
        end_date = st.sidebar.date_input("End Date", value=end_date)
        
        # Monte Carlo parameters
        st.sidebar.subheader("🎲 Monte Carlo Simulation")
        forecast_days = st.sidebar.slider(
            "Forecast Days", 
            min_value=30, 
            max_value=365*2, 
            value=252
        )
        num_simulations = st.sidebar.slider(
            "Number of Simulations", 
            min_value=100, 
            max_value=5000, 
            value=1000
        )
        
        return start_date, end_date, forecast_days, num_simulations
    
    @staticmethod
    def render_portfolio_composition_table(tickers, weights):
        """Render portfolio composition table"""
        st.subheader("Portfolio Composition")
        portfolio_df = pd.DataFrame({
            'Stock': tickers,
            'Weight (%)': [w for w in weights]
        })
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def render_performance_metrics(risk_metrics):
        """Render key performance metrics"""
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
    
    @staticmethod
    def render_stock_performance_table(tickers, stock_data):
        """Render individual stock performance table"""
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
        
        st.dataframe(pd.DataFrame(stock_performance), use_container_width=True, hide_index=True)
    
    @staticmethod
    def render_cumulative_returns_chart(performance, tickers, stock_data):
        """Render cumulative returns comparison chart"""
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
    
    @staticmethod
    def render_returns_distribution(performance):
        """Render daily returns distribution"""
        st.subheader("Daily Returns Distribution")
        fig_hist = px.histogram(
            x=performance['daily_returns'],
            nbins=50,
            title="Portfolio Daily Returns Distribution"
        )
        fig_hist.update_layout(xaxis_title="Daily Return", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    @staticmethod
    def render_risk_analysis(risk_metrics, performance):
        """Render risk analysis section"""
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
    
    @staticmethod
    def render_monte_carlo_forecast(mc_results, forecast_days, num_simulations):
        """Render Monte Carlo forecast visualization"""
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
    
    @staticmethod
    def render_help_section():
        """Render help and information section"""
        with st.expander("ℹ️ How to Use This App"):
            st.markdown("""
            **Stock Portfolio Simulator** helps you analyze and forecast portfolio performance:
            
            1. **Search & Add Stocks**: Use the search bar to find stocks by company name or ticker symbol
            2. **Portfolio Setup**: Adjust weights and add/remove stocks as needed
            3. **Analysis Period**: Choose your historical analysis timeframe
            4. **Monte Carlo Settings**: Configure forecast parameters
            5. **Analyze**: Click the analyze button to generate comprehensive results
            
            **Key Features**:
            - **Ticker Search**: Find stocks easily by company name
            - **Interactive Portfolio Management**: Add, remove, and reweight stocks dynamically
            - **Risk Assessment**: Comprehensive risk metrics and volatility analysis
            - **Future Forecasting**: Monte Carlo simulation for potential outcomes
            
            **Key Metrics Explained**:
            - **Sharpe Ratio**: Risk-adjusted returns (higher is better)
            - **Value at Risk (VaR)**: Potential loss in worst 5% of cases
            - **Maximum Drawdown**: Largest peak-to-trough decline
            - **Monte Carlo**: Statistical method using random sampling to model uncertainty
            
            **Note**: This is for educational purposes only and should not be considered investment advice.
            """)
    

    
    @staticmethod
    def render_correlation_matrix(data_fetcher, stock_data, tickers):
        """Render correlation matrix for portfolio stocks"""
        st.subheader("🔗 Portfolio Correlation Matrix")
        
        if stock_data is not None and not stock_data.empty and len(tickers) > 1:
            correlation_matrix = data_fetcher.calculate_correlation_matrix(stock_data)
            
            if correlation_matrix is not None:
                # Create heatmap
                fig = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Stock Correlation Matrix"
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Stocks",
                    yaxis_title="Stocks"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.info("""
                **Correlation Matrix Interpretation:**
                - **1.0**: Perfect positive correlation (stocks move together)
                - **0.0**: No correlation (independent movement)
                - **-1.0**: Perfect negative correlation (stocks move opposite)
                - **High correlation (>0.7)**: Stocks tend to move in the same direction
                - **Low correlation (<0.3)**: More diversified portfolio
                """)
            else:
                st.warning("Unable to calculate correlation matrix.")
        else:
            st.info("Add at least 2 stocks to your portfolio to see correlation analysis.")