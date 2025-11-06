#!/usr/bin/env python3
"""
Interactive Portfolio Analysis CLI
This script provides an interactive command-line interface for portfolio analysis,
including data extraction, portfolio creation, and Monte Carlo simulation.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.financial_toolkit import DataExtractor, Portfolio
from dotenv import load_dotenv

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_user_symbols() -> List[str]:
    """Get stock symbols from user input."""
    print("\n=== Stock Symbol Input ===")
    print("Enter stock symbols (e.g., AAPL, MSFT, GOOGL)")
    print("Enter a blank line when done")

    symbols = []
    while True:
        symbol = input("Enter symbol (or press Enter to finish): ").strip().upper()
        if not symbol:
            if not symbols:
                print("Please enter at least one symbol!")
                continue
            break
        symbols.append(symbol)
    return symbols

def get_date_range() -> tuple:
    """Get date range from user input."""
    print("\n=== Date Range Selection ===")
    print("Default: Last 10 years to present")

    use_default = input("Use default date range? (y/n): ").lower().strip() == 'y'
    if use_default:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        return start_date, end_date

    while True:
        try:
            start_str = input("Enter start date (YYYY-MM-DD): ")
            start_date = datetime.strptime(start_str, "%Y-%m-%d")

            end_str = input("Enter end date (YYYY-MM-DD): ")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")

            if end_date <= start_date:
                print("End date must be after start date!")
                continue

            return start_date, end_date
        except ValueError:
            print("Invalid date format! Please use YYYY-MM-DD")

def get_portfolio_weights(symbols: List[str]) -> Dict[str, float]:
    """Get portfolio weights from user input."""
    print("\n=== Portfolio Weight Assignment ===")
    print("Enter weights for each symbol (must sum to 1.0)")

    weights = {}
    remaining = 1.0

    for i, symbol in enumerate(symbols):
        while True:
            try:
                if i == len(symbols) - 1:
                    print(f"Remaining weight {remaining:.2f} will be assigned to {symbol}")
                    weights[symbol] = remaining
                    break

                weight = float(input(f"Enter weight for {symbol} (remaining: {remaining:.2f}): "))
                if weight < 0:
                    print("Weight must be positive!")
                    continue
                if weight > remaining:
                    print(f"Weight cannot exceed remaining amount: {remaining:.2f}")
                    continue

                weights[symbol] = weight
                remaining -= weight
                break
            except ValueError:
                print("Invalid input! Please enter a number.")

    return weights

def get_simulation_params() -> Dict:
    """Get Monte Carlo simulation parameters from user."""
    print("\n=== Monte Carlo Simulation Parameters ===")

    params = {}

    # Number of simulations
    while True:
        try:
            num_sims = int(input("Enter number of simulations (default 1000): ") or "1000")
            if num_sims < 100:
                print("Please enter at least 100 simulations!")
                continue
            params['num_simulations'] = num_sims
            break
        except ValueError:
            print("Invalid input! Please enter a number.")

    # Time horizon
    while True:
        try:
            horizon = int(input("Enter time horizon in days (default 252): ") or "252")
            if horizon < 1:
                print("Time horizon must be positive!")
                continue
            params['horizon_days'] = horizon
            break
        except ValueError:
            print("Invalid input! Please enter a number.")

    # Initial investment
    while True:
        try:
            initial = float(input("Enter initial investment amount (default 10000): ") or "10000")
            if initial <= 0:
                print("Initial investment must be positive!")
                continue
            params['V0'] = initial
            break
        except ValueError:
            print("Invalid input! Please enter a number.")

    # Simulation method
    while True:
        method = input("Choose simulation method (gaussian/bootstrap) [default: gaussian]: ").lower().strip() or "gaussian"
        if method in ['gaussian', 'bootstrap']:
            params['method'] = method
            break
        print("Invalid method! Please choose 'gaussian' or 'bootstrap'.")

    return params

def show_menu() -> str:
    """Display main menu and get user choice."""
    print("\n=== Portfolio Analysis Menu ===")
    print("1. View Portfolio Composition")
    print("2. View Basic Statistics")
    print("3. Run Monte Carlo Simulation")
    print("4. Generate Full Report")
    print("5. Plot Simulation Results")
    print("6. Plot Detailed Analysis")
    print("7. Change Portfolio Weights")
    print("8. Exit")

    while True:
        choice = input("\nEnter your choice (1-8): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
            return choice
        print("Invalid choice! Please enter a number between 1 and 8.")

def main():
    """Main function to run the interactive portfolio analysis."""
    clear_screen()
    print("Welcome to Interactive Portfolio Analysis")
    print("========================================")

    # Load environment variables
    load_dotenv()

    # Initialize data extractor
    extractor = DataExtractor()

    # Get symbols from user
    symbols = get_user_symbols()

    # Get date range
    start_date, end_date = get_date_range()

    # Fetch market data
    print("\nFetching market data...")
    data = extractor.fetch_historic_prices(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        source='yahoo'
    )
    print("✓ Market data fetched successfully")

    # Get portfolio weights
    weights = get_portfolio_weights(symbols)

    # Create portfolio
    portfolio = Portfolio(
        holdings={symbol_data.prices[0].symbol: symbol_data for symbol_data in data},
        weights=weights
    )

    simulation_result = None

    while True:
        choice = show_menu()

        if choice == '1':
            print("\n=== Portfolio Composition ===")
            print(portfolio)
            input("\nPress Enter to continue...")

        elif choice == '2':
            print("\n=== Basic Statistics ===")
            for symbol, holding in portfolio.holdings.items():
                print(f"\nStatistics for {symbol}:")
                stats = holding.get_statistics_summary()
                for key, value in stats.items():
                    print(f"{key}: {value}")
            input("\nPress Enter to continue...")

        elif choice == '3':
            params = get_simulation_params()
            print("\nRunning Monte Carlo simulation...")
            simulation_result = portfolio.monte_carlo_simulation(**params)
            print("✓ Simulation completed successfully")
            input("\nPress Enter to continue...")

        elif choice == '4':
            if simulation_result is None:
                print("\nPlease run a simulation first (option 3)!")
                input("\nPress Enter to continue...")
                continue

            print("\nGenerating full report...")
            report = portfolio.report(simulation_result, save_to_file=True)
            print(report)
            input("\nPress Enter to continue...")

        elif choice == '5':
            if simulation_result is None:
                print("\nPlease run a simulation first (option 3)!")
                input("\nPress Enter to continue...")
                continue

            print("\nGenerating simulation plots...")
            simulation_result.plots_report()
            input("\nPress Enter to continue...")

        elif choice == '6':
            if simulation_result is None:
                print("\nPlease run a simulation first (option 3)!")
                input("\nPress Enter to continue...")
                continue

            print("\nGenerating detailed analysis plots...")
            simulation_result.plots_detailed_report()
            input("\nPress Enter to continue...")

        elif choice == '7':
            weights = get_portfolio_weights(symbols)
            portfolio = Portfolio(
                holdings={symbol_data.prices[0].symbol: symbol_data for symbol_data in data},
                weights=weights
            )
            simulation_result = None  # Reset simulation result as weights changed
            print("✓ Portfolio weights updated")
            input("\nPress Enter to continue...")

        elif choice == '8':
            print("\nThank you for using Interactive Portfolio Analysis!")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)

