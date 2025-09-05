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
    if 'analysis_triggered' not in st.session_state:
        st.session_state.analysis_triggered = False
    if 'portfolio_changed' not in st.session_state:
        st.session_state.portfolio_changed = False
    
    # === UNIFIED PORTFOLIO SETUP SECTION ===
    st.header("📊 Portfolio Setup & Stock Discovery")
    
    # Create columns for unified setup
    setup_col1, setup_col2 = st.columns([3, 2])
    
    with setup_col1:
        st.subheader("Stock Discovery")
        # Stock discovery section (no auto-refresh)
        ui.render_stock_discovery_no_refresh(data_fetcher)

    with setup_col2:
        st.subheader("Portfolio Composition")
        # Portfolio input (no auto-refresh)
        tickers, weights = ui.render_portfolio_input_no_refresh()
    

    
    # Analysis parameters and trigger
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📅 Analysis Parameters")
        start_date, end_date, forecast_days, num_simulations = ui.render_analysis_parameters_inline()
    
    # with col2:
    #     st.subheader("📊 Portfolio Summary")
    #     if tickers:
    #         ui.render_portfolio_composition_table(tickers, weights)
    #     else:
    #         st.info("Add stocks to see portfolio composition")
    
    with col2:
        st.subheader("🚀 Analysis")
        # Reset analysis if portfolio changed
        if st.session_state.get('portfolio_changed', False):
            st.session_state.analysis_triggered = False
            st.session_state.portfolio_changed = False
        
        analyze_button = st.button("Analyze Portfolio", type="primary", use_container_width=True)
        if analyze_button:
            st.session_state.analysis_triggered = True
    
    # === ANALYSIS RESULTS SECTION ===
    if st.session_state.analysis_triggered:
        st.markdown("---")
        st.header("📈 Portfolio Analysis Results")
        
        if not tickers:
            st.error("Please add at least one stock ticker to your portfolio.")
            st.session_state.analysis_triggered = False
        elif abs(sum(weights) - 100.0) > 0.01:
            st.error(f"Portfolio weights must sum to 100%, currently {sum(weights):.2f}%")
            st.session_state.analysis_triggered = False
        else:
            # Convert weights to decimal
            weights_decimal = [w/100 for w in weights]
            
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
                    st.error("Unable to calculate portfolio performance. Please check your data.")
                    st.session_state.analysis_triggered = False
            else:
                st.error("Unable to fetch stock data. Please check your ticker symbols and try again.")
                st.session_state.analysis_triggered = False
    
    # === HELP SECTION ===
    st.markdown("---")
    ui.render_help_section()

if __name__ == "__main__":
    main()