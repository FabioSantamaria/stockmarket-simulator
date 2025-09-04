import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioSimulator:
    """Core portfolio simulation and analysis functionality"""
    
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
                    print(f"Warning: No data found for {ticker}")
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
    
    def search_ticker(self, query):
        """Search for ticker symbols based on company name or ticker"""
        try:
            # This is a simplified search - in production, you might use a more comprehensive API
            # For now, we'll return some common tickers that match the query
            common_tickers = {
                'apple': 'AAPL',
                'microsoft': 'MSFT', 
                'google': 'GOOGL',
                'alphabet': 'GOOGL',
                'amazon': 'AMZN',
                'tesla': 'TSLA',
                'meta': 'META',
                'facebook': 'META',
                'netflix': 'NFLX',
                'nvidia': 'NVDA',
                'intel': 'INTC',
                'amd': 'AMD',
                'coca cola': 'KO',
                'johnson': 'JNJ',
                'walmart': 'WMT',
                'disney': 'DIS',
                'boeing': 'BA',
                'goldman': 'GS',
                'jpmorgan': 'JPM',
                'visa': 'V',
                'mastercard': 'MA',
                'paypal': 'PYPL',
                'salesforce': 'CRM',
                'oracle': 'ORCL',
                'adobe': 'ADBE',
                'zoom': 'ZM',
                'uber': 'UBER',
                'airbnb': 'ABNB',
                'spotify': 'SPOT',
                'twitter': 'TWTR',
                'snap': 'SNAP',
                'pinterest': 'PINS'
            }
            
            query_lower = query.lower()
            matches = []
            
            # Search by company name
            for name, ticker in common_tickers.items():
                if query_lower in name or name in query_lower:
                    matches.append({'name': name.title(), 'ticker': ticker})
            
            # Search by ticker symbol
            for name, ticker in common_tickers.items():
                if query_lower == ticker.lower():
                    matches.append({'name': name.title(), 'ticker': ticker})
            
            # If query looks like a ticker (all caps, 1-5 chars), add it directly
            if query.isupper() and 1 <= len(query) <= 5:
                matches.append({'name': query, 'ticker': query})
            
            return matches[:10]  # Return top 10 matches
            
        except Exception as e:
            print(f"Error searching ticker: {str(e)}")
            return []
    
    def validate_ticker(self, ticker):
        """Validate if a ticker exists and has data"""
        try:
            stock = yf.Ticker(ticker)
            # Try to get recent data to validate
            hist = stock.history(period="5d")
            return not hist.empty
        except:
            return False
    
    def get_stock_info(self, ticker):
        """Get basic information about a stock"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0)
            }
        except:
            return {
                'name': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown', 
                'market_cap': 0,
                'current_price': 0
            }
    
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