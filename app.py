import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, lpSum, LpStatus
from typing import Dict, Tuple


class SupplierOptimizer:
    """
    Class to solve the supplier optimization problem where items are sourced from suppliers
    under stock, pallet, expiry, and cost constraints.
    """

    PALLET_SIZE = 24

    def __init__(self, items_file: str, pricing_file: str, suppliers_file: str) -> None:
        """Load data from CSV files and initialize the optimization model."""

        self.items_df = pd.read_csv(items_file)
        self.pricing_df = pd.read_csv(pricing_file)
        self.suppliers_df = pd.read_csv(suppliers_file)

        self.model = LpProblem("Supplier_Optimization", LpMinimize)
        self.available_pairs = self._prepare_available_pairs()
        self.x: Dict[Tuple[int, int], LpVariable] = {}
        self.y: Dict[int, LpVariable] = {}

    def _prepare_available_pairs(self) -> pd.DataFrame:
        """Return valid (item, supplier) combinations by merging input data."""

        merged = self.pricing_df.merge(
            self.suppliers_df, on='SupplierID'
        ).merge(
            self.items_df, on='ItemID', suffixes=('_supplier', '_item')
        )
        return merged.rename(columns={
            'Name_supplier': 'SupplierName',
            'Name_item': 'ItemName'
        })

    def define_variables(self) -> None:
        """Define decision variables for units ordered and pallet usage."""

        self.x = {
            (row['ItemID'], row['SupplierID']):
            LpVariable(f"x_{row['ItemID']}_{row['SupplierID']}", lowBound=0, cat=LpInteger)
            for _, row in self.available_pairs.iterrows()
        }

        self.y = {
            supplier_id: LpVariable(f"y_{supplier_id}", lowBound=0, cat=LpInteger)
            for supplier_id in self.suppliers_df['SupplierID']
        }

    def set_objective(self) -> None:
        """Set the objective function to minimize total purchasing cost."""

        self.model += lpSum(
            self.x[i, j] * (
                self.pricing_df.query("ItemID == @i and SupplierID == @j")['CostPerPallet'].values[0] / self.PALLET_SIZE
            )
            for (i, j) in self.x
        )

    def add_stock_constraints(self) -> None:
        """Add constraints to keep stock within allowed min/max levels."""

        for _, row in self.items_df.iterrows():
            item_id = row['ItemID']
            current_stock = row['CurrentStock']
            min_stock = row['MinStock']
            max_stock = row['MaxStock']
            item_suppliers = [j for (i, j) in self.x if i == item_id]

            self.model += (
                lpSum(self.x[item_id, j] for j in item_suppliers) + current_stock >= min_stock,
                f"StockMin_{item_id}"
            )

            self.model += (
                lpSum(self.x[item_id, j] for j in item_suppliers) + current_stock <= max_stock,
                f"StockMax_{item_id}"
            )

    def add_pallet_constraints(self) -> None:
        """Add constraints on pallet usage, including minimum and maximum pallets."""

        for _, row in self.suppliers_df.iterrows():
            sup_id = row['SupplierID']
            min_pallets = row['MinPallets']
            max_pallets = row['MaxPallets']
            supplier_items = [i for (i, j) in self.x if j == sup_id]

            units_sum = lpSum(self.x[i, sup_id] for i in supplier_items)
            self.model += units_sum == self.y[sup_id] * self.PALLET_SIZE, f"PalletUnits_{sup_id}"
            self.model += self.y[sup_id] >= min_pallets, f"PalletMin_{sup_id}"
            self.model += self.y[sup_id] <= max_pallets, f"PalletMax_{sup_id}"

    def add_expiry_constraints(self) -> None:
        """Add constraints to prevent excess stock past the expiry window."""

        for _, row in self.items_df.iterrows():
            item_id = row['ItemID']
            avg_daily_sale = row['AverageDailySale']
            current_stock = row['CurrentStock']
            expiry = row['Expiry (days)']
            item_suppliers = [j for (i, j) in self.x if i == item_id]

            for j in item_suppliers:
                lead_time = self.suppliers_df.query("SupplierID == @j")['LeadTime (days)'].values[0]
                expected_demand = avg_daily_sale * (expiry - lead_time)
                self.model += self.x[item_id, j] + current_stock <= expected_demand, f"ExpLead_{item_id}_{j}"

    def solve(self) -> None:
        """Run the solver on the defined optimization problem."""

        self.model.solve()

    def export_solution_csv(self, filename: str = "optimized_plan.csv") -> None:
        """Export the optimal purchasing plan in a tabular format (CSV)."""

        rows = []
        for (i, j), var in self.x.items():
            if var.varValue and var.varValue > 0:
                item_name = self.items_df.query("ItemID == @i")['Name'].values[0]
                supplier_name = self.suppliers_df.query("SupplierID == @j")['Name'].values[0]
                units = int(var.varValue)
                pallets = round(units / self.PALLET_SIZE, 2)
                rows.append({
                    'ItemID': i,
                    'ItemName': item_name,
                    'SupplierID': j,
                    'SupplierName': supplier_name,
                    'Units': units,
                    'Pallets': pallets
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"\nOptimal purchasing plan saved to '{filename}'.")

    def print_solution(self) -> None:

        print("\nOptimal purchasing plan:")
        print(f"{'Item':<20} {'Supplier':<20} {'Units':>10} {'Pallets':>10}")
        print("-" * 60)
        for (i, j), var in self.x.items():
            if var.varValue and var.varValue > 0:
                item_name = self.items_df.query("ItemID == @i")['Name'].values[0]
                supplier_name = self.suppliers_df.query("SupplierID == @j")['Name'].values[0]
                units = int(var.varValue)
                pallets = round(units / self.PALLET_SIZE, 2)
                print(f"{item_name:<20} {supplier_name:<20} {units:>10} {pallets:>10.2f}")

        print("\nTotal pallets per supplier:")
        for sup_id, var in self.y.items():
            if var.varValue and var.varValue > 0:
                name = self.suppliers_df.query("SupplierID == @sup_id")['Name'].values[0]
                print(f"{name:<20} {int(var.varValue)} pallets")

        print("\nModel status:", LpStatus[self.model.status])

    def run(self) -> None:
        """Execute the full workflow: setup, solve, and output results."""

        self.define_variables()
        self.set_objective()
        self.add_stock_constraints()
        self.add_pallet_constraints()
        self.add_expiry_constraints()
        self.solve()
        self.print_solution()
        self.export_solution_csv()


if __name__ == "__main__":
    optimizer = SupplierOptimizer(
        items_file="data/items_updated.csv",
        pricing_file="data/pricing.csv",
        suppliers_file="data/suppliers.csv"
    )
    optimizer.run()
