import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class DataFetcher:
    """Handle all data fetching and caching operations"""
    
    def __init__(self):
        self.cache = {}
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_stock_data(_self, tickers, start_date, end_date):
        """Fetch historical stock data with caching"""
        try:
            data = {}
            failed_tickers = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    if not hist.empty and len(hist) > 10:  # Ensure we have enough data
                        data[ticker] = hist['Close']
                    else:
                        failed_tickers.append(ticker)
                        
                except Exception as e:
                    failed_tickers.append(ticker)
                    print(f"Error fetching {ticker}: {str(e)}")
            
            if failed_tickers:
                st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")
            
            if data:
                df = pd.DataFrame(data)
                # Forward fill missing values
                df = df.fillna(method='ffill')
                # Drop any remaining NaN values
                df = df.dropna()
                return df
            else:
                return None
                
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return None
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_stock_info(_self, ticker):
        """Get detailed stock information with caching"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get current price from recent data if not in info
            current_price = info.get('currentPrice')
            if not current_price:
                try:
                    recent_data = stock.history(period="1d")
                    if not recent_data.empty:
                        current_price = recent_data['Close'].iloc[-1]
                except:
                    current_price = 0
            
            return {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': current_price or 0,
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'website': info.get('website', ''),
                'business_summary': info.get('longBusinessSummary', ''),
                'employees': info.get('fullTimeEmployees', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0)
            }
            
        except Exception as e:
            print(f"Error getting info for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'name': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'current_price': 0,
                'currency': 'USD',
                'exchange': 'Unknown',
                'country': 'Unknown',
                'website': '',
                'business_summary': '',
                'employees': 0,
                'pe_ratio': 0,
                'dividend_yield': 0,
                'beta': 0
            }
    
    def validate_ticker(self, ticker):
        """Validate if a ticker exists and has recent data"""
        try:
            stock = yf.Ticker(ticker)
            # Try to get recent data to validate
            hist = stock.history(period="5d")
            return not hist.empty and len(hist) > 0
        except:
            return False
    
    def get_market_indices(self, start_date, end_date):
        """Fetch major market indices for comparison"""
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT'
        }
        
        try:
            data = {}
            for name, ticker in indices.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    if not hist.empty:
                        data[name] = hist['Close']
                except:
                    continue
            
            return pd.DataFrame(data) if data else None
            
        except Exception as e:
            print(f"Error fetching market indices: {str(e)}")
            return None
    
    def search_similar_tickers(self, ticker):
        """Find similar tickers in the same sector/industry"""
        try:
            stock_info = self.get_stock_info(ticker)
            sector = stock_info.get('sector', '')
            
            # This is a simplified approach - in production you'd use a more comprehensive database
            sector_tickers = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM'],
                'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'BIIB'],
                'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
                'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG'],
                'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'TWTR'],
                'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY'],
                'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'FDX'],
                'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED'],
                'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EXR'],
                'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF'],
                'Basic Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF']
            }
            
            similar = sector_tickers.get(sector, [])
            # Remove the original ticker from suggestions
            similar = [t for t in similar if t != ticker.upper()]
            
            return similar[:5]  # Return top 5 suggestions
            
        except Exception as e:
            print(f"Error finding similar tickers: {str(e)}")
            return []
    
    def get_trending_stocks(self):
        """Get a list of trending/popular stocks"""
        # This would ideally come from a real trending API
        # For now, return a curated list of popular stocks
        trending = [
            {'ticker': 'AAPL', 'name': 'Apple Inc.'},
            {'ticker': 'MSFT', 'name': 'Microsoft Corporation'},
            {'ticker': 'GOOGL', 'name': 'Alphabet Inc.'},
            {'ticker': 'AMZN', 'name': 'Amazon.com Inc.'},
            {'ticker': 'TSLA', 'name': 'Tesla Inc.'},
            {'ticker': 'META', 'name': 'Meta Platforms Inc.'},
            {'ticker': 'NVDA', 'name': 'NVIDIA Corporation'},
            {'ticker': 'NFLX', 'name': 'Netflix Inc.'},
            {'ticker': 'JPM', 'name': 'JPMorgan Chase & Co.'},
            {'ticker': 'JNJ', 'name': 'Johnson & Johnson'}
        ]
        
        return trending
    
    def calculate_correlation_matrix(self, stock_data):
        """Calculate correlation matrix for portfolio stocks"""
        if stock_data is None or stock_data.empty:
            return None
        
        try:
            # Calculate daily returns
            returns = stock_data.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            return correlation_matrix
            
        except Exception as e:
            print(f"Error calculating correlation matrix: {str(e)}")
            return None
    
    def search_ticker(self, query):
        """Search for ticker symbols using yfinance API"""
        try:
            # Use yfinance Search API for real-time ticker search
            search = yf.Search(query, max_results=10)
            results = []
            
            if hasattr(search, 'quotes') and search.quotes:
                for quote in search.quotes:
                    results.append({
                        'symbol': quote.get('symbol', ''),
                        'name': quote.get('longname', quote.get('shortname', '')),
                        'exchange': quote.get('exchange', ''),
                        'type': quote.get('quoteType', '')
                    })
            
            # Fallback: try Lookup if Search doesn't work
            if not results:
                try:
                    lookup = yf.Lookup(query)
                    if hasattr(lookup, 'quotes') and lookup.quotes:
                        for quote in lookup.quotes[:10]:  # Limit to 10 results
                            results.append({
                                'symbol': quote.get('symbol', ''),
                                'name': quote.get('longname', quote.get('shortname', '')),
                                'exchange': quote.get('exchange', ''),
                                'type': quote.get('quoteType', '')
                            })
                except:
                    pass
            
            # Final fallback: validate if query is already a ticker
            if not results and self.validate_ticker(query.upper()):
                results.append({
                    'symbol': query.upper(),
                    'name': query.upper(),
                    'exchange': 'Unknown',
                    'type': 'EQUITY'
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching ticker: {str(e)}")
            return []
    
    def get_sector_stocks(self):
        """Get dynamic sector-based stock recommendations"""
        return {
            "Technology": ["AAPL", "GOOGL", "MSFT", "NVDA", "META"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Finance": ["JPM", "BAC", "WFC", "GS", "MS"],
            "Consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB"]
        }
    
    def get_economic_indicators(self, start_date, end_date):
        """Get economic indicators data"""
        indicators = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^TNX': '10-Year Treasury'
        }
        
        try:
            data = {}
            for symbol, name in indicators.items():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[name] = hist['Close']
            return pd.DataFrame(data) if data else None
        except Exception as e:
            print(f"Error fetching economic indicators: {str(e)}")
            return None