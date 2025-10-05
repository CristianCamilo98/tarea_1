# Financial Market Data Toolkit

A comprehensive Python 3 toolkit for fetching, cleaning, and analyzing financial market data. Built with Object-Oriented Programming (OOP) principles, this toolkit provides modules for multi-API data extraction, standardized price dataclasses, portfolio simulation using Monte Carlo methods, markdown reporting, and visualization.

## Features

- **Multi-API Data Extraction**: Fetch financial data from multiple sources (Yahoo Finance, Alpha Vantage)
- **Standardized Data Models**: Clean dataclass-based structures for price data and portfolios
- **Data Cleaning**: Comprehensive utilities for handling missing values, outliers, and data validation
- **Portfolio Simulation**: Monte Carlo simulation for risk analysis and portfolio optimization
- **Risk Metrics**: Calculate volatility, Sharpe ratio, VaR, CVaR, and maximum drawdown
- **Markdown Reporting**: Generate professional analysis reports in markdown format
- **Visualization**: Create publication-quality charts for price history, simulations, and portfolios

## Installation

```bash
# Clone the repository
git clone https://github.com/CristianCamilo98/tarea_1.git
cd tarea_1

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
tarea_1/
├── src/
│   └── financial_toolkit/
│       ├── __init__.py
│       ├── data_models.py          # Dataclasses for financial data
│       ├── data_extractor.py       # Multi-API data fetching
│       ├── data_cleaner.py         # Data cleaning utilities
│       ├── portfolio_simulator.py  # Monte Carlo simulation
│       ├── report_generator.py     # Markdown report generation
│       └── visualizer.py           # Visualization tools
├── tests/
│   ├── test_data_models.py
│   ├── test_data_cleaner.py
│   ├── test_portfolio_simulator.py
│   └── test_report_generator.py
├── examples/
│   └── complete_analysis.py        # Complete workflow example
├── requirements.txt
└── README.md
```

## Quick Start

### Basic Usage

```python
from datetime import datetime, timedelta
from src.financial_toolkit import (
    DataExtractor,
    DataCleaner,
    Portfolio,
    PortfolioSimulator,
    ReportGenerator,
    Visualizer
)

# 1. Extract data
extractor = DataExtractor()
data = extractor.fetch_multiple(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now(),
    source='yahoo'
)

# 2. Clean data
cleaner = DataCleaner()
cleaned_data = cleaner.clean_data(
    data,
    remove_duplicates=True,
    handle_missing='ffill',
    validate_prices=True,
    calculate_returns=True
)

# 3. Create portfolio
portfolio = Portfolio(
    name='My Portfolio',
    assets={'AAPL': 0.4, 'GOOGL': 0.35, 'MSFT': 0.25},
    initial_value=100000.0,
    data=cleaned_data
)

# 4. Run Monte Carlo simulation
simulator = PortfolioSimulator(portfolio)
result = simulator.simulate(
    num_simulations=1000,
    time_horizon=252,
    random_seed=42
)

# 5. Generate report
report_gen = ReportGenerator()
report = report_gen.generate_full_report(
    portfolio=portfolio,
    simulation_result=result,
    risk_metrics=simulator.calculate_risk_metrics()
)
report_gen.save_report(report, 'analysis_report.md')

# 6. Create visualizations
visualizer = Visualizer()
fig = visualizer.plot_simulation_results(result)
visualizer.save_figure(fig, 'simulation.png')
```

## Module Documentation

### Data Models (`data_models.py`)

Provides standardized dataclasses for financial data:

- **PriceData**: Standardized structure for OHLCV data
- **Portfolio**: Portfolio structure with asset allocations
- **SimulationResult**: Monte Carlo simulation results

### Data Extractor (`data_extractor.py`)

Multi-API data extraction:

```python
extractor = DataExtractor(alpha_vantage_key='YOUR_API_KEY')

# Fetch from Yahoo Finance
data = extractor.fetch_yahoo_finance('AAPL', start_date, end_date)

# Fetch from Alpha Vantage
data = extractor.fetch_alpha_vantage('GOOGL', start_date, end_date)

# Fetch multiple symbols
df = extractor.fetch_multiple(['AAPL', 'GOOGL'], source='yahoo')
```

### Data Cleaner (`data_cleaner.py`)

Comprehensive data cleaning utilities:

```python
cleaner = DataCleaner()

# Remove duplicates
clean_df = cleaner.remove_duplicates(df)

# Handle missing values
clean_df = cleaner.handle_missing_values(df, method='ffill')

# Validate price data
clean_df = cleaner.validate_price_data(df)

# Calculate returns
clean_df = cleaner.calculate_returns(df, method='log')

# Full pipeline
clean_df = cleaner.clean_data(df, remove_duplicates=True,
                               handle_missing='ffill',
                               validate_prices=True,
                               calculate_returns=True)
```

### Portfolio Simulator (`portfolio_simulator.py`)

Monte Carlo simulation and risk analysis:

```python
simulator = PortfolioSimulator(portfolio)

# Run simulation
result = simulator.simulate(
    num_simulations=1000,
    time_horizon=252,
    random_seed=42
)

# Calculate risk metrics
metrics = simulator.calculate_risk_metrics()
# Returns: volatility, sharpe_ratio, max_drawdown, var_95, cvar_95

# Get optimal weights
weights = simulator.calculate_optimal_weights(method='sharpe')
```

### Report Generator (`report_generator.py`)

Generate professional markdown reports:

```python
generator = ReportGenerator()

# Portfolio report
report = generator.generate_portfolio_report(portfolio)

# Simulation report
report = generator.generate_simulation_report(simulation_result)

# Risk report
report = generator.generate_risk_report(risk_metrics, 'Portfolio Name')

# Comprehensive report
report = generator.generate_full_report(
    portfolio=portfolio,
    simulation_result=result,
    risk_metrics=metrics,
    data_summary=df
)

# Save report
generator.save_report(report, 'report.md')
```

### Visualizer (`visualizer.py`)

Create publication-quality visualizations:

```python
visualizer = Visualizer()

# Price history
fig = visualizer.plot_price_history(data, symbol='AAPL')

# Returns distribution
fig = visualizer.plot_returns_distribution(returns)

# Simulation results
fig = visualizer.plot_simulation_results(result, num_paths_to_plot=100)

# Portfolio allocation
fig = visualizer.plot_portfolio_allocation(portfolio)

# Correlation matrix
fig = visualizer.plot_correlation_matrix(returns_df)

# Save figure
visualizer.save_figure(fig, 'chart.png', dpi=300)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_models.py

# Run with coverage
pytest --cov=src/financial_toolkit tests/
```

## Complete Example

Run the complete example demonstrating all features:

```bash
python examples/complete_analysis.py
```

This will:
1. Fetch market data for multiple symbols
2. Clean and validate the data
3. Create a diversified portfolio
4. Run Monte Carlo simulation (1000 simulations)
5. Calculate risk metrics
6. Generate a comprehensive markdown report
7. Create visualization charts

Output files:
- `financial_analysis_report.md`: Comprehensive analysis report
- `portfolio_allocation.png`: Portfolio allocation chart
- `simulation_results.png`: Monte Carlo simulation visualization
- `price_history_aapl.png`: Historical price chart

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- requests >= 2.31.0
- yfinance >= 0.2.0
- scipy >= 1.11.0
- pytest >= 7.4.0 (for testing)

## API Keys

To use Alpha Vantage as a data source, you need an API key:

1. Register at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Pass your API key when initializing the DataExtractor:

```python
extractor = DataExtractor(alpha_vantage_key='YOUR_API_KEY')
```

## License

MIT License - see LICENSE file for details.

## Contributing

This project was created as part of Master MIAX coursework. Feel free to use and adapt for educational purposes.

## Author

CristianCamilo98

## Acknowledgments

Built using industry-standard libraries:
- pandas for data manipulation
- numpy for numerical computing
- matplotlib for visualization
- yfinance for Yahoo Finance API access
- scipy for statistical calculations
