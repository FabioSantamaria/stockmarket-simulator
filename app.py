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
    
    # Portfolio Configuration Section (moved from sidebar to main area)
    st.header("📊 Portfolio Configuration")
    
    # Create columns for portfolio setup
    config_col1, config_col2 = st.columns([2, 1])
    
    with config_col1:
        # Portfolio input (modified to work in main area)
        tickers, weights = ui.render_portfolio_input_main()
    
    with config_col2:
        # Stock discovery section
        ui.render_stock_discovery(data_fetcher, simulator)
    
    # Analysis parameters in sidebar
    st.sidebar.header("Analysis Parameters")
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
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Performance", "📈 Visualization", "🔗 Correlation Matrix", "⚠️ Risk Metrics", "🔮 Forecast"])
                
                with tab1:
                    ui.render_performance_metrics(risk_metrics)
                    ui.render_stock_performance_table(tickers, stock_data)
                
                with tab2:
                    ui.render_cumulative_returns_chart(performance, tickers, stock_data)
                    ui.render_returns_distribution(performance)
                
                with tab3:
                    ui.render_correlation_matrix(data_fetcher, stock_data, tickers)
                
                with tab4:
                    ui.render_risk_analysis(risk_metrics, performance)
                
                with tab5:
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
    

    
    # Help section
    ui.render_help_section()

if __name__ == "__main__":
    main()