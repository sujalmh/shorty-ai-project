from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import io
import pandas as pd
import logging

logger = logging.getLogger("ai.data_service")


class DataService:
    """
    In-memory dataframe store with simple cell metadata.
    NOTE: This is a single-process demo store. Replace with a DB for multi-user scenarios.
    """

    def __init__(self) -> None:
        self.df: pd.DataFrame = pd.DataFrame()
        # metadata keyed by (rowIndex, field) -> attributes dict
        self.cell_meta: Dict[Tuple[int, str], Dict[str, Any]] = {}
        # initialize with a small demo dataset
        self.seed_sample()

    def seed_sample(self) -> None:
        data = [
            {"region": "North", "product": "A", "q1": 120, "q2": 150, "revenue": 270, "cost": 180},
            {"region": "South", "product": "B", "q1": 80, "q2": 95, "revenue": 175, "cost": 120},
            {"region": "East", "product": "C", "q1": 200, "q2": 210, "revenue": 410, "cost": 260},
            {"region": "West", "product": "A", "q1": 50, "q2": 70, "revenue": 120, "cost": 130},
        ]
        self.df = pd.DataFrame(data)
        self.cell_meta.clear()

    def import_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.df = pd.DataFrame(rows) if rows else pd.DataFrame()
        self.cell_meta.clear()

    # -------- file imports --------
    def import_csv_bytes(self, content: bytes, encoding: Optional[str] = None) -> None:
        """Replace the current dataframe with CSV content."""
        buf = io.BytesIO(content)
        df = pd.read_csv(buf, encoding=encoding) if encoding else pd.read_csv(buf)
        self.df = df
        self.cell_meta.clear()

    def import_excel_bytes(self, content: bytes, sheet_name: Optional[str] = None) -> None:
        """Replace the current dataframe with Excel content (first sheet by default)."""
        buf = io.BytesIO(content)
        df = pd.read_excel(buf, sheet_name=sheet_name if sheet_name is not None else 0)
        self.df = df
        self.cell_meta.clear()

    # -------- simple structural ops --------
    def add_row(self, values: Optional[Dict[str, Any]] = None) -> int:
        """Append a new row and return its index."""
        if self.df is None or self.df.empty:
            # If no columns yet and values provided, infer columns from values
            if values:
                self.df = pd.DataFrame([values])
            else:
                # start with a single empty row if columns are unknown
                self.df = pd.DataFrame([{}])
            return int(len(self.df.index) - 1)

        idx = int(len(self.df.index))
        row = {c: (values.get(c) if values and c in values else None) for c in self.df.columns}
        self.df.loc[idx] = row  # type: ignore[call-overload, assignment]
        return idx

    def add_empty_column(self, name: str, fill: Any = None) -> None:
        """Create a new empty column filled with 'fill'."""
        if not name or name.strip() == "":
            raise ValueError("Column name cannot be empty")
        if name in self.df.columns:
            return
        self.df[name] = fill
        for idx in range(len(self.df.index)):
            self.cell_meta[(idx, name)] = {"type": "string", "source": "manual"}

    def delete_rows(self, rows: List[int]) -> List[int]:
        """
        Delete rows by zero-based indices. Returns the list of actually-deleted indices (within range).
        Resets the index and clears cell metadata (simple approach).
        """
        if self.df.empty or not rows:
            return []
        max_idx = len(self.df.index) - 1
        to_delete = sorted({int(r) for r in rows if 0 <= int(r) <= max_idx})
        if not to_delete:
            return []
        self.df = self.df.drop(index=to_delete, errors="ignore").reset_index(drop=True)
        # For now, clear all cell metadata (can be optimized to remap later)
        self.cell_meta.clear()
        return to_delete

    def delete_column(self, name: str) -> bool:
        """
        Delete a column by exact name if it exists. Returns True if removed.
        """
        if name in self.df.columns:
            self.df.drop(columns=[name], inplace=True)
            # Clear metadata for that column
            keys = [k for k in self.cell_meta.keys() if k[1] == name]
            for k in keys:
                self.cell_meta.pop(k, None)
            return True
        return False

    def _dtype_to_type(self, dtype) -> str:
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "date"
        return "string"

    def get_columns(self) -> List[Dict[str, Any]]:
        cols: List[Dict[str, Any]] = []
        for name, dtype in self.df.dtypes.items():
            cols.append(
                {
                    "field": str(name),
                    "type": self._dtype_to_type(dtype),
                }
            )
        return cols

    def get_rows(self) -> List[Dict[str, Any]]:
        if self.df.empty:
            return []
        # Pylance: pandas returns list[dict[Hashable, Any]]; narrow for our string-keyed columns.
        return cast(List[Dict[str, Any]], self.df.to_dict(orient="records"))

    def get_metadata_summary(self) -> Dict[str, Any]:
        return {
            "rowCount": int(self.df.shape[0]) if not self.df.empty else 0,
            "columnCount": int(self.df.shape[1]) if not self.df.empty else 0,
        }

    def update_cell(
        self,
        row_index: int,
        field: str,
        value: Any,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        # auto-extend dataframe if row index out of range (simple support)
        if row_index >= len(self.df.index):
            # append empty rows until desired index exists
            for _ in range(row_index - len(self.df.index) + 1):
                self.df.loc[len(self.df.index)] = {c: None for c in self.df.columns}  # type: ignore[call-overload]

        # create column if missing
        if field not in self.df.columns:
            self.df[field] = None

        self.df.at[row_index, field] = value
        if attributes:
            self.cell_meta[(row_index, field)] = attributes

    def add_column_expression(self, name: str, expression: str) -> None:
        """
        Evaluate an expression against the dataframe to produce a new column.
        Example: name='profit', expression='revenue - cost'
        """
        if name.strip() == "":
            raise ValueError("Column name cannot be empty")

        # Simpler and type-stable: rely on df.eval which returns a Series for column-wise expressions.
        series = self.df.eval(expression)

        self.df[name] = series
        # store formula metadata on each cell (optional)
        for idx in range(len(self.df.index)):
            self.cell_meta[(idx, name)] = {
                "type": "formula",
                "source": "ai",
                "formula": expression,
            }

    def highlight_rows(self, condition: str) -> List[int]:
        """
        Return row indices where condition holds true.
        Example: 'profit < 0' or 'revenue > cost * 1.2'
        """
        if self.df.empty:
            return []
        mask = self.df.eval(condition)
        if isinstance(mask, pd.Series):
            mask_bool = mask.astype(bool)
        else:
            raise ValueError("Condition must evaluate to a boolean Series")
        return self.df[mask_bool].index.astype(int).tolist()

    def plot(self, x: str, y: Optional[str], kind: str = "bar") -> Dict[str, Any]:
        """
        Build a Plotly figure dict (frontend can render via plotly.js).
        Coerce numeric columns as needed and handle simple aggregation for pie charts.
        """
        import plotly.express as px  # lazy import

        if self.df.empty:
            raise ValueError("No data to plot")

        # Work on a copy to avoid mutating the main df
        df = self.df.copy()
        logger.info("plot: kind=%s x=%s y=%s cols=%s", kind, x, y, list(df.columns))
        try:
            dtypes = {c: str(df[c].dtype) for c in df.columns}
            logger.info("plot dtypes: %s", dtypes)
        except Exception:
            pass

        if kind == "pie":
            if y is None:
                raise ValueError("Pie chart requires 'y' as values")
            # Ensure numeric values for y and drop missing categories
            if y not in df.columns:
                raise ValueError(f"Column '{y}' not found")
            if x not in df.columns:
                raise ValueError(f"Column '{x}' not found")
            df[y] = pd.to_numeric(df[y], errors="coerce")
            df = df.dropna(subset=[x, y])
            # Aggregate by category to avoid duplicate slices
            agg = df.groupby(x, as_index=False)[y].sum()
            fig = px.pie(agg, names=x, values=y, title=f"{y} by {x}")
        elif kind == "line":
            if y is None:
                raise ValueError("Line chart requires 'y'")
            if y not in df.columns or x not in df.columns:
                raise ValueError("Columns not found for line chart")
            df[y] = pd.to_numeric(df[y], errors="coerce")
            df = df.dropna(subset=[x, y])
            fig = px.line(df, x=x, y=y, title=f"{y} over {x}")
        else:  # default bar
            if y is None:
                raise ValueError("Bar chart requires 'y'")
            if y not in df.columns or x not in df.columns:
                raise ValueError("Columns not found for bar chart")
            df[y] = pd.to_numeric(df[y], errors="coerce")
            df = df.dropna(subset=[x, y])
            fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")

        # Use Plotly's JSON-ready format to avoid numpy serialization issues
        fig_json = fig.to_plotly_json()
        try:
            logger.info("plot ok: traces=%s title=%s", len(fig_json.get("data", [])), (fig_json.get("layout") or {}).get("title"))
        except Exception:
            pass
        return fig_json


# Global singleton for simple demo purposes
data_service = DataService()
