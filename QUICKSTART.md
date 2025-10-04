# Quick Start Guide

Get started with the Financial Market Data Toolkit in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/CristianCamilo98/tarea_1.git
cd tarea_1

# Install dependencies
pip install -r requirements.txt
```

## 5-Minute Tutorial

### Step 1: Run a Complete Example

```bash
python examples/complete_analysis.py
```

This will:
- Fetch market data (or use simulated data)
- Create a diversified portfolio
- Run Monte Carlo simulation
- Calculate risk metrics
- Generate reports and visualizations

**Output files:**
- `financial_analysis_report.md`: Comprehensive markdown report
- `portfolio_allocation.png`: Portfolio pie chart
- `simulation_results.png`: Monte Carlo simulation visualization
- `price_history_aapl.png`: Price history chart

### Step 2: Try Basic Usage

```python
from src.financial_toolkit import Portfolio, PortfolioSimulator

# Create a portfolio
portfolio = Portfolio(
    name='My Portfolio',
    assets={'AAPL': 0.5, 'GOOGL': 0.5},
    initial_value=10000.0
)

# Note: You'll need to add historical data to the portfolio
# See examples/basic_usage.py for a complete working example
```

### Step 3: Run Tests

```bash
pytest tests/ -v
```

All 37 tests should pass!

## Common Workflows

### Workflow 1: Fetch and Clean Data

```python
from datetime import datetime, timedelta
from src.financial_toolkit import DataExtractor, DataCleaner

# Fetch data
extractor = DataExtractor()
data = extractor.fetch_multiple(
    symbols=['AAPL', 'GOOGL'],
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now()
)

# Clean data
cleaner = DataCleaner()
clean_data = cleaner.clean_data(
    data,
    remove_duplicates=True,
    handle_missing='ffill',
    validate_prices=True,
    calculate_returns=True
)
```

### Workflow 2: Run Portfolio Simulation

```python
from src.financial_toolkit import Portfolio, PortfolioSimulator

# Assuming you have 'data' from Workflow 1
portfolio = Portfolio(
    name='Tech Portfolio',
    assets={'AAPL': 0.6, 'GOOGL': 0.4},
    initial_value=50000.0,
    data=data
)

simulator = PortfolioSimulator(portfolio)
result = simulator.simulate(
    num_simulations=1000,
    time_horizon=252,
    random_seed=42
)

# View statistics
print(f"Expected return: {result.statistics['expected_return']:.2%}")
print(f"Risk of loss: {result.statistics['probability_of_loss']:.2%}")
```

### Workflow 3: Generate Reports

```python
from src.financial_toolkit import ReportGenerator

generator = ReportGenerator()

# Generate comprehensive report
report = generator.generate_full_report(
    portfolio=portfolio,
    simulation_result=result,
    risk_metrics=simulator.calculate_risk_metrics()
)

# Save to file
generator.save_report(report, 'my_analysis.md')
```

### Workflow 4: Create Visualizations

```python
from src.financial_toolkit import Visualizer

visualizer = Visualizer()

# Portfolio allocation
fig1 = visualizer.plot_portfolio_allocation(portfolio)
visualizer.save_figure(fig1, 'allocation.png')

# Simulation results
fig2 = visualizer.plot_simulation_results(result)
visualizer.save_figure(fig2, 'simulation.png')

# Price history
fig3 = visualizer.plot_price_history(data, symbol='AAPL')
visualizer.save_figure(fig3, 'prices.png')
```

## Examples

Three complete examples are provided:

1. **`examples/complete_analysis.py`**
   - Full end-to-end workflow
   - Data fetching, cleaning, analysis, reporting, visualization
   - Best for understanding the complete toolkit

2. **`examples/basic_usage.py`**
   - Simplified usage example
   - Focus on portfolio simulation
   - Great for quick start

3. **`examples/data_cleaning_demo.py`**
   - Data extraction and cleaning focus
   - Shows all cleaning options
   - Best for data preprocessing

## Tips

### Using Real Market Data

To fetch real market data from Yahoo Finance:

```python
extractor = DataExtractor()
data = extractor.fetch_multiple(
    symbols=['AAPL', 'MSFT'],
    source='yahoo'
)
```

### Using Alpha Vantage

To use Alpha Vantage API:

```python
extractor = DataExtractor(alpha_vantage_key='YOUR_API_KEY')
data = extractor.fetch_data('AAPL', source='alpha_vantage')
```

Get your free API key at: https://www.alphavantage.co/support/#api-key

### Handling Network Issues

If you can't access external APIs, the examples will automatically fall back to simulated data for demonstration purposes.

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [FEATURES.md](FEATURES.md) for complete feature list
3. Explore the source code in `src/financial_toolkit/`
4. Review tests in `tests/` for usage patterns
5. Modify examples to analyze your own portfolios

## Need Help?

- Check the docstrings in each module
- Review the test files for usage examples
- Run examples with different parameters
- Read the comprehensive README

## Common Issues

**Issue**: "No data fetched for any symbol"
**Solution**: Network restrictions. Examples will use simulated data automatically.

**Issue**: "Missing optional dependency 'tabulate'"
**Solution**: `pip install tabulate`

**Issue**: Import errors
**Solution**: Make sure you're in the project root directory when running scripts

Happy analyzing! 🚀📈
