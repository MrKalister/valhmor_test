import pytest
import pandas as pd
from pulp import LpStatus, LpMinimize
import tempfile
import os
from app import SupplierOptimizer


@pytest.fixture
def sample_data():
    """Fixture providing test data in temporary CSV files."""

    test_data = {
        'items.csv': pd.DataFrame({
            'ItemID': [1, 2],
            'Name': ['Item1', 'Item2'],
            'CurrentStock': [50, 100],
            'MinStock': [100, 150],
            'MaxStock': [200, 300],
            'AverageDailySale': [5, 10],
            'Expiry (days)': [30, 45],
        }),
        'suppliers.csv': pd.DataFrame({
            'SupplierID': [101, 102],
            'Name': ['SupplierA', 'SupplierB'],
            'MinPallets': [1, 2],
            'MaxPallets': [5, 6],
            'LeadTime (days)': [3, 5],
        }),
        'pricing.csv': pd.DataFrame({
            'ItemID': [1, 1, 2],
            'SupplierID': [101, 102, 101],
            'CostPerPallet': [240, 300, 480],
        })
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, df in test_data.items():
            df.to_csv(os.path.join(temp_dir, filename), index=False)
        yield temp_dir


@pytest.fixture
def base_optimizer(sample_data):
    """Fixture: Base optimizer instance without initialization."""

    return SupplierOptimizer(
        items_file=os.path.join(sample_data, 'items.csv'),
        pricing_file=os.path.join(sample_data, 'pricing.csv'),
        suppliers_file=os.path.join(sample_data, 'suppliers.csv')
    )


@pytest.fixture
def initialized_optimizer(base_optimizer):
    """Fixture: Fully initialized optimizer with variables and constraints."""

    base_optimizer.define_variables()
    return base_optimizer


def test_initialization(base_optimizer):
    """Test successful initialization with CSV data."""

    assert len(base_optimizer.items_df) == 2
    assert len(base_optimizer.suppliers_df) == 2
    assert len(base_optimizer.pricing_df) == 3


def test_available_pairs(base_optimizer):
    """Test creation of valid supplier-item pairs."""

    pairs = base_optimizer._prepare_available_pairs()
    expected_columns = {
        'ItemID',
        'SupplierID',
        'CostPerPallet',
        'SupplierName',
        'MinPallets',
        'MaxPallets',
        'LeadTime (days)',
        'ItemName',
        'CurrentStock',
        'MinStock',
        'MaxStock',
        'AverageDailySale',
        'Expiry (days)',
    }

    assert len(pairs) == 3
    assert set(pairs.columns) == expected_columns


def test_variable_creation(initialized_optimizer):
    """Test decision variables initialization."""

    # Validate x variables (item-supplier combinations)
    assert len(initialized_optimizer.x) == 3
    assert all((i, j) in initialized_optimizer.x for (i, j) in [(1, 101), (1, 102), (2, 101)])

    # Validate y variables (suppliers)
    assert len(initialized_optimizer.y) == 2
    assert all(sid in initialized_optimizer.y for sid in [101, 102])


def test_objective_function(initialized_optimizer):
    """Test objective function configuration."""

    initialized_optimizer.set_objective()

    assert initialized_optimizer.model.name == "Supplier_Optimization"
    assert initialized_optimizer.model.sense == LpMinimize
    assert initialized_optimizer.model.objective is not None


def test_stock_constraints(initialized_optimizer):
    """Test inventory constraints configuration."""

    initialized_optimizer.add_stock_constraints()

    # Verify constraints for ItemID=1
    min_constraint = initialized_optimizer.model.constraints.get("StockMin_1")
    max_constraint = initialized_optimizer.model.constraints.get("StockMax_1")

    assert min_constraint is not None
    assert max_constraint is not None
    assert str(min_constraint) == "x_1_101 + x_1_102 >= 50.0"
    assert str(max_constraint) == "x_1_101 + x_1_102 <= 150.0"


def test_pallet_constraints(initialized_optimizer):
    """Test pallet-related constraints configuration."""

    initialized_optimizer.add_pallet_constraints()

    # Verify constraints for SupplierID=101
    units_constraint = initialized_optimizer.model.constraints.get("PalletUnits_101")
    min_constraint = initialized_optimizer.model.constraints.get("PalletMin_101")
    max_constraint = initialized_optimizer.model.constraints.get("PalletMax_101")

    assert all(c is not None for c in [units_constraint, min_constraint, max_constraint])
    assert str(units_constraint) == "x_1_101 + x_2_101 - 24*y_101 = -0.0"
    assert str(min_constraint) == "y_101 >= 1"
    assert str(max_constraint) == "y_101 <= 5"


def test_expiry_constraints(initialized_optimizer):
    """Test expiry date constraints."""

    initialized_optimizer.add_expiry_constraints()

    # For SupplierID=101 & ItemID=1
    exp_constraint = initialized_optimizer.model.constraints.get("ExpLead_1_101")

    # self.x[item_id, j] + current_stock <= avg_daily_sale * (expiry - lead_time)
    # x_1_101 + 50 <= 5 * (30 - 3) -> x_1_101 + 50 <= 135 - 50 -> x_1_101 <= 85
    assert str(exp_constraint) == "x_1_101 <= 85"


def test_supplier_overlap(sample_data):
    """Test overlapping supplier availability."""

    # Change data to check overlapping
    pricing_df = pd.read_csv(os.path.join(sample_data, 'pricing.csv'))
    pricing_df = pd.concat([pricing_df, pd.DataFrame({
        'ItemID': [2],
        'SupplierID': [102],
        'CostPerPallet': [500]
    })])
    pricing_df.to_csv(os.path.join(sample_data, 'pricing.csv'), index=False)

    optimizer = SupplierOptimizer(
        items_file=os.path.join(sample_data, 'items.csv'),
        pricing_file=os.path.join(sample_data, 'pricing.csv'),
        suppliers_file=os.path.join(sample_data, 'suppliers.csv')
    )
    optimizer.define_variables()

    # Check available combinations
    pairs = optimizer._prepare_available_pairs()
    assert len(pairs[pairs['SupplierID'] == 102]) == 2


@pytest.mark.parametrize("supplier_id,expected_constraint", [
    (101, ("-24*y_101 = -0.0", "y_101 >= 1")),
    (102, ("-24*y_102 = -0.0", "y_102 >= 2"))
])
def test_pallet_edge_cases(sample_data, supplier_id, expected_constraint):
    """Test pallet constraints when suppliers have no items."""

    # Remove all items from supplier
    pricing_df = pd.read_csv(os.path.join(sample_data, 'pricing.csv'))
    pricing_df = pricing_df[pricing_df['SupplierID'] != supplier_id]
    pricing_df.to_csv(os.path.join(sample_data, 'pricing.csv'), index=False)

    optimizer = SupplierOptimizer(
        items_file=os.path.join(sample_data, 'items.csv'),
        pricing_file=os.path.join(sample_data, 'pricing.csv'),
        suppliers_file=os.path.join(sample_data, 'suppliers.csv')
    )
    optimizer.define_variables()
    optimizer.add_pallet_constraints()

    units_constraint = optimizer.model.constraints.get(f"PalletUnits_{supplier_id}")
    min_constraint = optimizer.model.constraints.get(f"PalletMin_{supplier_id}")

    assert str(units_constraint) == expected_constraint[0]
    assert str(min_constraint) == expected_constraint[1]


def test_full_workflow(base_optimizer, capsys):
    """Integration test for complete optimization workflow."""

    base_optimizer.run()

    # Validate solution status
    assert LpStatus[base_optimizer.model.status] == 'Optimal'


def test_infeasible_scenario(sample_data):
    """Test scenario with conflicting constraints."""

    # Create impossible inventory situation
    items_df = pd.read_csv(os.path.join(sample_data, 'items.csv'))
    items_df.loc[0, 'MaxStock'] = 0  # CurrentStock=50, MinStock=100
    items_df.to_csv(os.path.join(sample_data, 'items.csv'), index=False)

    optimizer = SupplierOptimizer(
        items_file=os.path.join(sample_data, 'items.csv'),
        pricing_file=os.path.join(sample_data, 'pricing.csv'),
        suppliers_file=os.path.join(sample_data, 'suppliers.csv')
    )
    optimizer.run()

    assert LpStatus[optimizer.model.status] == 'Infeasible'


def test_objective_value(base_optimizer):
    """Test total cost calculation."""

    base_optimizer.run()

    # 0.92*240 + 2*300 + 2.08*480
    expected_cost = 1820.0

    # Acceptable inaccuracy
    print(base_optimizer.model.objective.value())
    assert abs(base_optimizer.model.objective.value() - expected_cost) < 0.1
