import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from modules.portfolio_simulator import PortfolioSimulator
from modules.ui_components import UIComponents
from modules.data_fetcher import DataFetcher



def main():
    # Initialize components
    ui = UIComponents()
    simulator = PortfolioSimulator()
    data_fetcher = DataFetcher()
    
    # Configure page and render header
    ui.setup_page_config()
    ui.apply_custom_css()
    ui.render_header()
    
    # Initialize session state for portfolio
    if 'portfolio_tickers' not in st.session_state:
        st.session_state.portfolio_tickers = ['AAPL', 'GOOGL', 'MSFT']
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = [33.33, 33.33, 33.34]
    
    # Sidebar configuration
    st.sidebar.header("Portfolio Configuration")
    
    # Ticker search functionality
    ui.render_ticker_search(simulator)
    
    # Portfolio input
    tickers, weights = ui.render_portfolio_input()
    
    # Analysis parameters
    start_date, end_date, forecast_days, num_simulations = ui.render_analysis_parameters()
    
    # Analyze button
    if st.sidebar.button("Analyze Portfolio", type="primary"):
        if not tickers:
            st.error("Please add at least one stock ticker to your portfolio.")
            return
        
        if abs(sum(weights) - 100.0) > 0.01:
            st.error(f"Portfolio weights must sum to 100%. not {weights}")
            return
        
        # Convert weights to decimal
        weights_decimal = [w/100 for w in weights]
        
        # Show portfolio composition
        ui.render_portfolio_composition_table(tickers, weights)
        
        # Fetch data and analyze
        with st.spinner("Fetching stock data..."):
            stock_data = data_fetcher.fetch_stock_data(tickers, start_date, end_date)
        
        if stock_data is not None and not stock_data.empty:
            # Calculate portfolio performance
            with st.spinner("Calculating portfolio performance..."):
                performance = simulator.calculate_portfolio_performance(stock_data, weights_decimal)
            
            if performance:
                # Calculate risk metrics
                risk_metrics = simulator.calculate_risk_metrics(performance['daily_returns'])
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 Visualization", "⚠️ Risk Metrics", "🔮 Forecast"])
                
                with tab1:
                    ui.render_performance_metrics(risk_metrics)
                    ui.render_stock_performance_table(tickers, stock_data)
                
                with tab2:
                    ui.render_cumulative_returns_chart(performance, tickers, stock_data)
                    ui.render_returns_distribution(performance)
                
                with tab3:
                    ui.render_risk_analysis(risk_metrics, performance)
                
                with tab4:
                    with st.spinner("Running Monte Carlo simulation..."):
                        mc_results = simulator.monte_carlo_simulation(
                            performance['daily_returns'], 
                            days_ahead=forecast_days, 
                            num_simulations=num_simulations
                        )
                    
                    if mc_results:
                        ui.render_monte_carlo_forecast(mc_results, forecast_days, num_simulations)
        else:
            st.error("Unable to fetch stock data. Please check your ticker symbols and try again.")
    
    # Additional Analysis Sections
    st.markdown("---")
    st.header("📊 Additional Market Analysis")
    
    # Create tabs for additional features
    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
        "📈 Market Indices", 
        "🔗 Correlation Matrix", 
        "🌍 Economic Indicators", 
        "🏭 Sector Search", 
        "🔥 Trending Stocks"
    ])
    
    with analysis_tab1:
        ui.render_market_indices(data_fetcher, start_date, end_date)
    
    with analysis_tab2:
        # Only show correlation matrix if we have portfolio data
        if 'stock_data' in locals() and stock_data is not None:
            ui.render_correlation_matrix(data_fetcher, stock_data, tickers)
        else:
            st.info("Analyze your portfolio first to see correlation matrix.")
    
    with analysis_tab3:
        ui.render_economic_indicators(data_fetcher, start_date, end_date)
    
    with analysis_tab4:
        ui.render_sector_search(data_fetcher, simulator)
    
    with analysis_tab5:
        ui.render_trending_stocks(data_fetcher)
    
    # Help section
    ui.render_help_section()

if __name__ == "__main__":
    main()