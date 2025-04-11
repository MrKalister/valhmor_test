# Supplier Optimization

This project solves a supply chain optimization task using linear programming. The goal is to determine how many units of each item to order from which suppliers to minimize costs, while meeting business constraints.

## Features

- Restock planning for items based on:
  - Minimum and maximum stock levels
  - Supplier availability
  - Expiry dates and lead times
  - Pallet constraints (integer number of 24-unit pallets)
- Cost minimization using `pulp` (linear programming)
- Tabular CSV output of the final purchasing plan

## Input

Three CSV files are required (in `data/`):

- `items_updated.csv` — Item data (ID, name, stock levels, demand, expiry)
- `pricing.csv` — Cost per pallet for each item-supplier pair
- `suppliers.csv` — Supplier data (pallet limits, lead times)

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run app
```bash
python app.py
```

### Run tests
```bash
pytest -v test_optimizer.py
```

### Coverage
#### to make
```bash
coverage run --source=app -m pytest
```
#### to display
```bash
coverage report -m
```

#### *purchasing plan will be saved to 'optimized_plan.csv'.