# -*- coding: utf-8 -*-
"""Code to read in data files from Lanhe instruments"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from echemistpy.io.structures import RawData, RawDataInfo

class LanheReadError(Exception):
    """Raised when a Lanhe file cannot be parsed."""


class LanheReader:
    """Reader for Lanhe .xlsx files."""

    def __init__(self, path_to_file: str | Path | None = None) -> None:
        self.path_to_file = Path(path_to_file) if path_to_file else None

    def read(self) -> tuple[RawData, RawDataInfo]:
        """Read the file and return RawData and RawDataInfo."""
        if self.path_to_file is None:
            raise LanheReadError("No file path provided for Lanhe reader.")

        if not self.path_to_file.exists():
            raise LanheReadError(f"File not found: {self.path_to_file}")

        suffix = self.path_to_file.suffix.lower()
        if suffix == ".xlsx" or suffix == ".xls":
            return self._read_excel()
        else:
            raise LanheReadError(
                "Only Lanhe .xlsx or .xls exports are supported by this reader.",
            )

    def _read_excel(self) -> tuple[RawData, RawDataInfo]:
        try:
            # Read all sheets to handle multi-sheet structure
            # User specified: Sheet 1 & 2 are metadata, Sheet 3 is data
            # We use index 0, 1 for meta, 2 for data
            # sheet_name=None returns a dict of {sheet_name: dataframe}
            dfs = pd.read_excel(self.path_to_file, sheet_name=None, header=None)
            sheet_names = list(dfs.keys())
        except Exception as exc:
            raise LanheReadError(f"Failed to parse Excel file: {exc}") from exc

        if not dfs:
            raise LanheReadError("Excel file is empty.")

        meta = {"filename": self.path_to_file.name}
        df_data = None

        # Logic to handle multi-sheet structure
        if len(sheet_names) >= 3:
            # Assume Sheet 0 and 1 are metadata
            meta.update(self._parse_metadata(dfs[sheet_names[0]]))
            meta.update(self._parse_metadata(dfs[sheet_names[1]]))
            
            # Assume Sheet 2 is data
            df_raw_data = dfs[sheet_names[2]]
        else:
            # Fallback: If fewer than 3 sheets, assume the first sheet contains data
            # and try to parse metadata from it as well (before header)
            # This maintains backward compatibility with single-sheet files
            df_raw_data = dfs[sheet_names[0]]
            # We will parse metadata from this sheet up to the header row later

        # Process Data Sheet
        # Find the header row in the data sheet
        header_row_idx = self._find_header_row(df_raw_data)
        
        if header_row_idx is None:
             raise LanheReadError("Could not find a valid data header in the data sheet.")

        # If we are in the fallback single-sheet mode, extract metadata from top rows
        if len(sheet_names) < 3:
             meta.update(self._parse_metadata(df_raw_data.iloc[:header_row_idx]))

        # Extract data (rows after header)
        df_data = df_raw_data.iloc[header_row_idx + 1:].copy()
        df_data.columns = df_raw_data.iloc[header_row_idx].values
        
        # Deduplicate columns if necessary
        df_data = self._deduplicate_columns(df_data)

        # Reset index
        df_data.reset_index(drop=True, inplace=True)
        df_data.index.name = "row"

        # Clean and standardize columns
        df_data = self._standardize_columns(df_data)
        
        # Deduplicate columns again after standardization (in case multiple cols mapped to same name)
        df_data = self._deduplicate_columns(df_data)
        
        # Convert to numeric, coercing errors to NaN
        # Note: we iterate over a list of columns to avoid issues if duplicates still somehow exist (though they shouldn't)
        for col in df_data.columns:
            # If we still have duplicates, df_data[col] would be a DataFrame, so we check
            if isinstance(df_data[col], pd.DataFrame):
                 # This should be handled by _deduplicate_columns, but as a safety net:
                 # We take the first one or handle it. 
                 # But _deduplicate_columns should prevent this.
                 pass
            else:
                df_data[col] = pd.to_numeric(df_data[col], errors='coerce')

        dataset = xr.Dataset.from_dataframe(df_data)

        return RawData(data=dataset), RawDataInfo(meta=meta)

    def _find_header_row(self, df: pd.DataFrame) -> int | None:
        """Find the row index that contains data headers."""
        # Common columns found in Lanhe files (Chinese or English)
        # We look for a row that contains at least a few of these
        candidates = [
            "Test Time", "Voltage", "Current", "Capacity", "Energy",
            "测试时间", "电压", "电流", "容量", "能量", "Step Time", "工步时间"
        ]
        
        for idx, row in df.iterrows():
            # Convert row values to string and check for intersection with candidates
            row_values = [str(v).strip() for v in row.values if pd.notna(v)]
            matches = [c for c in candidates if any(c in rv for rv in row_values)]
            
            # If we find at least 3 matches, we assume this is the header
            if len(matches) >= 3:
                return idx
        return None

    def _parse_metadata(self, df_meta: pd.DataFrame) -> dict[str, Any]:
        """Parse rows before the header as metadata."""
        meta = {}
        for _, row in df_meta.iterrows():
            # Simple key-value extraction: assume "Key: Value" or "Key Value" in cells
            # Or adjacent cells: Cell A = Key, Cell B = Value
            row_values = [v for v in row.values if pd.notna(v)]
            if not row_values:
                continue
            
            # Strategy: iterate through pairs
            for i in range(0, len(row_values) - 1, 2):
                key = str(row_values[i]).strip().rstrip(':')
                value = row_values[i+1]
                if key:
                    meta[key] = value
        return meta

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to standard echemistpy names."""
        # Map Lanhe specific names to standard names
        # This mapping can be expanded
        column_map = {
            "Test Time": "time/s",
            "测试时间": "time/s",
            "Total Time": "time/s",
            "总时间": "time/s",
            "Step Time": "step time/s",
            "工步时间": "step time/s",
            "Voltage": "Ewe/V",
            "电压": "Ewe/V",
            "Current": "I/mA", # Usually mA or A, need to check units if possible. Assuming mA for now or just I
            "电流": "I/mA",
            "Capacity": "Capacity/mAh",
            "容量": "Capacity/mAh",
            "Energy": "Energy/Wh",
            "能量": "Energy/Wh",
            "Step ID": "mode",
            "工步号": "mode",
            "Cycle ID": "cycle number",
            "循环号": "cycle number",
        }
        
        new_columns = {}
        for col in df.columns:
            col_str = str(col).strip()
            # Check for exact match or partial match if needed
            for k, v in column_map.items():
                if k in col_str:
                    new_columns[col] = v
                    break
            if col not in new_columns:
                new_columns[col] = col_str # Keep original if no match
                
        return df.rename(columns=new_columns)

    def _deduplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate column names by appending .1, .2, etc."""
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        return df
