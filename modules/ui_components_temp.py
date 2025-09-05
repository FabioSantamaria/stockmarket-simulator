# Temporary file with the new functions to add to ui_components.py

@staticmethod
def render_portfolio_input_no_refresh():
    """Render portfolio composition input without auto-refresh"""
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
                key=f"no_refresh_ticker_{i}"
            )
            if ticker != st.session_state.portfolio_tickers[i]:
                st.session_state.portfolio_tickers[i] = ticker.upper() if ticker else ''
                st.session_state.portfolio_changed = True
        
        with col2:
            weight = st.number_input(
                f"Weight {i+1} (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=st.session_state.portfolio_weights[i] if i < len(st.session_state.portfolio_weights) else 0.0,
                key=f"no_refresh_weight_{i}"
            )
            if i < len(st.session_state.portfolio_weights):
                if weight != st.session_state.portfolio_weights[i]:
                    st.session_state.portfolio_weights[i] = weight
                    st.session_state.portfolio_changed = True
            else:
                st.session_state.portfolio_weights.append(weight)
                st.session_state.portfolio_changed = True
        
        with col3:
            if len(st.session_state.portfolio_tickers) > 1:
                if st.button("🗑️", key=f"no_refresh_remove_{i}", help=f"Remove stock {i+1}"):
                    st.session_state.portfolio_tickers.pop(i)
                    if i < len(st.session_state.portfolio_weights):
                        st.session_state.portfolio_weights.pop(i)
                    st.session_state.portfolio_changed = True
                    st.rerun()
        
        if st.session_state.portfolio_tickers[i]:
            tickers.append(st.session_state.portfolio_tickers[i].upper())
            weights.append(weight)
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Stock", key="no_refresh_add_stock"):
            st.session_state.portfolio_tickers.append('')
            # Redistribute weights equally
            equal_weight = 100.0 / len(st.session_state.portfolio_tickers)
            st.session_state.portfolio_weights = [equal_weight] * len(st.session_state.portfolio_tickers)
            st.session_state.portfolio_changed = True
            st.rerun()
    
    with col2:
        if st.button("⚖️ Normalize Weights", key="no_refresh_normalize"):
            if st.session_state.portfolio_weights and all(isinstance(w, (int, float)) for w in st.session_state.portfolio_weights):
                total_weight = sum(st.session_state.portfolio_weights)
                if total_weight > 0:
                    st.session_state.portfolio_weights = [
                        (w / total_weight) * 100 for w in st.session_state.portfolio_weights
                    ]
                    st.session_state.portfolio_changed = True
                    st.rerun()
    
    # Filter out empty tickers
    valid_tickers = [t for t in tickers if t]
    valid_weights = [w for t, w in zip(tickers, weights) if t]
    
    return valid_tickers, valid_weights

@staticmethod
def render_stock_discovery_no_refresh(data_fetcher):
    """Render stock discovery section without auto-refresh"""
    # Ticker search functionality
    search_query = st.text_input("🔍 Search for stocks", placeholder="Enter company name or ticker symbol...", key="stock_search_no_refresh")
    
    if search_query:
        with st.spinner("Searching for stocks..."):
            # Use data_fetcher for search
            results = data_fetcher.search_ticker(search_query)
        
        if results:
            st.write(f"Found {len(results)} result(s):")
            for result in results[:5]:  # Limit to top 5 results
                ticker = result.get('symbol', result.get('ticker', ''))
                name = result.get('name', result.get('longName', 'N/A'))
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{ticker}** - {name}")
                with col2:
                    if st.button(f"Add {ticker}", key=f"search_add_{ticker}"):
                        if ticker not in st.session_state.portfolio_tickers:
                            st.session_state.portfolio_tickers.append(ticker)
                            # Add equal weight for new stock
                            current_count = len(st.session_state.portfolio_tickers)
                            equal_weight = 100.0 / current_count
                            st.session_state.portfolio_weights = [equal_weight] * current_count
                            st.session_state.portfolio_changed = True
                            st.success(f"Added {ticker} to portfolio!")
                            st.rerun()
                        else:
                            st.warning(f"{ticker} is already in your portfolio")
        else:
            st.write("No results found. Try searching for company names or ticker symbols.")
    
    # Dynamic sector-based discovery
    st.markdown("---")
    st.write("**Browse by Sector:**")
    
    # Get dynamic sector data from data_fetcher
    sectors = data_fetcher.get_sector_stocks()
    
    selected_sector = st.selectbox("Select a sector:", list(sectors.keys()), key="sector_select_no_refresh")
    
    if selected_sector:
        st.write(f"Popular {selected_sector} stocks:")
        sector_stocks = sectors[selected_sector]
        
        cols = st.columns(min(len(sector_stocks), 5))  # Max 5 columns
        for i, stock in enumerate(sector_stocks[:5]):  # Limit to 5 stocks
            with cols[i]:
                if st.button(f"Add {stock}", key=f"sector_add_no_refresh_{stock}"):
                    if stock not in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers.append(stock)
                        # Add equal weight for new stock
                        current_count = len(st.session_state.portfolio_tickers)
                        equal_weight = 100.0 / current_count
                        st.session_state.portfolio_weights = [equal_weight] * current_count
                        st.session_state.portfolio_changed = True
                        st.success(f"Added {stock} to portfolio!")
                        st.rerun()
                    else:
                        st.warning(f"{stock} is already in your portfolio")
    
    # Dynamic trending stocks
    st.markdown("---")
    st.write("**Trending Stocks:**")
    
    # Get dynamic trending data from data_fetcher
    trending_stocks = data_fetcher.get_trending_stocks()
    
    cols = st.columns(4)
    for i, stock_info in enumerate(trending_stocks[:8]):  # Limit to 8 stocks
        stock = stock_info.get('symbol', stock_info.get('ticker', ''))
        with cols[i % 4]:
            if st.button(f"Add {stock}", key=f"trending_add_no_refresh_{stock}"):
                if stock not in st.session_state.portfolio_tickers:
                    st.session_state.portfolio_tickers.append(stock)
                    # Add equal weight for new stock
                    current_count = len(st.session_state.portfolio_tickers)
                    equal_weight = 100.0 / current_count
                    st.session_state.portfolio_weights = [equal_weight] * current_count
                    st.session_state.portfolio_changed = True
                    st.success(f"Added {stock} to portfolio!")
                    st.rerun()
                else:
                    st.warning(f"{stock} is already in your portfolio")

@staticmethod
def render_analysis_parameters_inline():
    """Render analysis parameters inline instead of in sidebar"""
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        end_date = datetime.now()
        start_date = st.date_input(
            "Start Date", 
            value=end_date - timedelta(days=365*2),
            key="inline_start_date"
        )
    
    with col2:
        end_date = st.date_input("End Date", value=end_date, key="inline_end_date")
    
    # Monte Carlo parameters
    col3, col4 = st.columns(2)
    
    with col3:
        forecast_days = st.slider(
            "Forecast Days", 
            min_value=30, 
            max_value=365*2, 
            value=252,
            key="inline_forecast_days"
        )
    
    with col4:
        num_simulations = st.slider(
            "Number of Simulations", 
            min_value=100, 
            max_value=10000, 
            value=1000,
            key="inline_num_simulations"
        )
    
    return start_date, end_date, forecast_days, num_simulations