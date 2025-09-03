# Stock Portfolio Simulator 📈

A comprehensive Streamlit application that simulates different stock portfolios, enabling users to visualize historical behavior, quantify risk, and project potential future performance using Monte Carlo simulations.

## Features

### 1. Portfolio Composition
- Interactive interface to input stock tickers and percentage weights
- Support for up to 10 stocks in a portfolio
- Automatic weight normalization

### 2. Historical Performance Analysis
- Fetch real-time historical stock data using Yahoo Finance
- Calculate weighted portfolio performance over time
- Compare portfolio performance against individual stocks
- Interactive charts with Plotly visualization

### 3. Risk Metrics & Analysis
- **Annual Return & Volatility**: Annualized performance metrics
- **Sharpe Ratio**: Risk-adjusted returns measurement
- **Value at Risk (VaR)**: Potential losses at 95% confidence level
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Rolling Volatility**: 30-day rolling volatility analysis

### 4. Monte Carlo Forecasting
- Statistical simulation of future portfolio performance
- Customizable forecast period (30 days to 2 years)
- Adjustable number of simulations (100 to 5,000)
- Confidence intervals and probability distributions
- Best/worst case scenario analysis

## Installation

1. **Clone or download this repository**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## How to Use

### Step 1: Configure Your Portfolio
1. In the sidebar, set the number of stocks (1-10)
2. Enter stock tickers (e.g., AAPL, GOOGL, MSFT, TSLA)
3. Set the weight percentage for each stock
4. Choose your analysis period (start and end dates)

### Step 2: Set Monte Carlo Parameters
1. Select forecast period (days ahead)
2. Choose number of simulations for accuracy

### Step 3: Analyze Portfolio
1. Click "Analyze Portfolio" button
2. Explore the four main tabs:
   - **Performance**: Key metrics and individual stock performance
   - **Visualization**: Interactive charts and return distributions
   - **Risk Metrics**: Comprehensive risk analysis
   - **Forecast**: Monte Carlo simulation results

## Example Portfolios

### Conservative Portfolio
- **AAPL** (Apple): 30%
- **MSFT** (Microsoft): 25%
- **JNJ** (Johnson & Johnson): 25%
- **KO** (Coca-Cola): 20%

### Growth Portfolio
- **TSLA** (Tesla): 40%
- **NVDA** (NVIDIA): 30%
- **AMZN** (Amazon): 30%

### Balanced Portfolio
- **SPY** (S&P 500 ETF): 40%
- **QQQ** (NASDAQ ETF): 30%
- **VTI** (Total Stock Market ETF): 30%

## Key Metrics Explained

- **Total Return**: Overall portfolio performance over the selected period
- **Annual Return**: Annualized return based on daily performance
- **Annual Volatility**: Standard deviation of returns (risk measure)
- **Sharpe Ratio**: Risk-adjusted returns (higher is better, >1 is excellent)
- **Value at Risk (VaR)**: Potential loss in the worst 5% of scenarios
- **Maximum Drawdown**: Largest decline from peak to trough
- **Monte Carlo Simulation**: Uses historical volatility to project future scenarios

## Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **yfinance**: Yahoo Finance data fetching
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **scipy**: Statistical functions
- **matplotlib & seaborn**: Additional plotting capabilities

### Data Sources
- Historical stock data from Yahoo Finance
- Real-time price updates
- Adjusted closing prices for accurate calculations

### Risk Calculations
- Assumes 252 trading days per year
- Uses 2% risk-free rate for Sharpe ratio calculation
- Monte Carlo simulations use normal distribution of historical returns

## Limitations & Disclaimers

⚠️ **Important**: This application is for educational and research purposes only. It should not be considered as investment advice.

- Historical performance does not guarantee future results
- Monte Carlo simulations are based on historical volatility patterns
- Real market conditions may differ significantly from simulations
- Consider transaction costs, taxes, and other factors in real investing
- Consult with financial professionals for investment decisions

## Troubleshooting

### Common Issues
1. **"No data found for ticker"**: Verify ticker symbols are correct and traded on major exchanges
2. **Empty charts**: Check date range and ensure it includes trading days
3. **Slow performance**: Reduce number of Monte Carlo simulations or forecast period
4. **Installation errors**: Ensure Python 3.7+ is installed and try upgrading pip

### Performance Tips
- Use shorter analysis periods for faster loading
- Start with fewer simulations (500-1000) for initial analysis
- Popular tickers (AAPL, GOOGL, MSFT) typically have more reliable data

## Future Enhancements

- [ ] Support for international markets
- [ ] Additional risk metrics (Beta, Alpha, Sortino ratio)
- [ ] Portfolio optimization algorithms
- [ ] Backtesting with rebalancing strategies
- [ ] Export functionality for reports
- [ ] Real-time portfolio tracking

## License

This project is open source and available under the MIT License.

---

**Happy Investing! 📊💰**