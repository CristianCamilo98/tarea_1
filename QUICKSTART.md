# Quick Start Guide

Get started with the Financial Market Data Toolkit in 5 minutes!

## Installation

### Using Docker (Recommended)

1. Build the Docker image

```bash
docker build -t financial-toolkit .
```

2. Run the Docker container

```bash
docker run --rm -it financial-toolkit
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/CristianCamilo98/tarea_1.git
cd tarea_1

# Install dependencies
python -m venv venv
source venv/bin/activate
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

### Step 2: Interactive Portfolio Analysis

Run the interactive CLI tool to explore features interactively:

```bash
python examples/interactive_portfolio_analysis.py
```

### Step 3: Try Basic Usage

```python
from src.financial_toolkit import (
    DataExtractor,
    DataCleaner,
    Portfolio,
    PortfolioSimulator,
    ReportGenerator,
    Visualizer
)

# Extract and preprocess data
extractor = DataExtractor()
data = extractor.fetch_multiple(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now()
)
cleaner = DataCleaner()
cleaned_data = cleaner.clean_data(
    data,
    remove_duplicates=True,
    handle_missing='ffill',
    validate_prices=True,
    calculate_returns=True
)

# Create a portfolio
portfolio = Portfolio(
    name='My Portfolio',
    assets={'AAPL': 0.5, 'GOOGL': 0.3, 'MSFT': 0.2},
    initial_value=100000.0
)
simulator = PortfolioSimulator(portfolio)
result = simulator.simulate(num_simulations=1000)
report_gen = ReportGenerator()
visualizer = Visualizer()

# Generate a report
report = report_gen.generate_full_report(portfolio, result, simulator.calculate_risk_metrics())
report_gen.save_report(report, 'portfolio_analysis.md')

# Create a visualization
visualization = visualizer.plot_simulation_results(result)
visualizer.save_figure(visualization, 'simulation.png')
```

### Step 4: Run Tests

```bash
pytest tests/ -v
```

All 37 tests should pass!
