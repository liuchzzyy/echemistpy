#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XLSX Data Reader for LANHE battery test files with metadata extraction using traitlets."""

import logging
from pathlib import Path
from typing import Any

import openpyxl
import openpyxl.worksheet.worksheet
import xarray as xr
from traitlets import HasTraits, Unicode

from echemistpy.io.structures import RawData, RawDataInfo

logger = logging.getLogger(__name__)


class LanheXLSXReader(HasTraits):
    """Reader for LANHE exported XLSX files."""

    filepath = Unicode()

    def __init__(self, filepath: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        if filepath:
            self.filepath = str(filepath)

    def load(self) -> tuple[RawData, RawDataInfo]:
        """Load LANHE XLSX file and return RawData and RawDataInfo."""
        if not self.filepath:
            raise ValueError("filepath not set")

        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Read raw data and metadata
        metadata, data_dict = self._read_xlsx()

        # Clean metadata and data
        cleaned_metadata = self._clean_metadata(metadata)
        cleaned_data = self._clean_data(data_dict)

        # Convert to xarray Dataset
        ds = xr.Dataset({k: (("row",), v) for k, v in cleaned_data.items()})

        # Create RawData and RawDataInfo
        raw_data = RawData(data=ds)
        raw_info = RawDataInfo(meta=cleaned_metadata)

        return raw_data, raw_info

    def _read_xlsx(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read metadata and data from XLSX file."""
        metadata: dict[str, Any] = {}
        data_dict: dict[str, Any] = {}
        data_sheet_name: str | None = None

        wb = openpyxl.load_workbook(self.filepath, read_only=True, data_only=True)
        try:
            self._read_test_info(wb, metadata)
            self._read_proc_info(wb, metadata)
            self._read_log_info(wb, metadata)
            data_sheet_name = self._find_data_sheet(wb)

            if data_sheet_name:
                data_dict = self._read_record_data_from_ws(wb[data_sheet_name], data_sheet_name)
        finally:
            wb.close()

        return metadata, data_dict

    def _read_record_data_from_ws(self, ws: openpyxl.worksheet.worksheet.Worksheet, data_sheet_name: str) -> dict[str, Any]:
        """Extract record-level data from an open worksheet."""
        data_dict = {}
        record_header_row = None
        record_data_rows = []

        for row in ws.iter_rows(values_only=True):
            if not row[0]:
                continue

            # Look for the record-level header (starts with "Cycle")
            if str(row[0]).strip().lower() == "cycle" and len(row) > 15:
                # Check if it has the characteristic record columns
                row_str = " ".join(str(cell).lower() for cell in row[:10] if cell)
                if "step" in row_str and "record" in row_str:
                    record_header_row = [str(cell) if cell else "" for cell in row]
                    record_header_row = [h.strip() for h in record_header_row if h.strip()]
                    continue

            # Extract record data rows (numeric cycle + step + record)
            if record_header_row and row[0] and row[1] and row[2]:
                try:
                    int(row[0])  # cycle
                    int(row[1])  # step
                    int(row[2])  # record
                    workmode = str(row[3]) if len(row) > 3 else ""
                    if any(kw in workmode for kw in ["REST", "CRATE", "LOOP", "END", "CHARGE", "DISCHARGE"]):
                        record_data_rows.append(row)
                except (ValueError, TypeError):
                    pass

        # Convert record data to dict
        if record_header_row and record_data_rows:
            for col_idx, header in enumerate(record_header_row):
                col_data = []
                for row_data in record_data_rows:
                    if col_idx < len(row_data):
                        col_data.append(self._convert_value(row_data[col_idx]))
                    else:
                        col_data.append(None)
                data_dict[header] = col_data

            data_dict["_metadata"] = {
                "sheet_name": data_sheet_name,
                "num_rows": len(record_data_rows),
                "num_columns": len(record_header_row),
                "columns": record_header_row,
            }
        return data_dict

    def _read_test_info(self, wb: openpyxl.Workbook, metadata: dict[str, Any]) -> None:
        """Read Test Information sheet."""
        if "Test information" in wb.sheetnames:
            try:
                ws = wb["Test information"]
                headers = [str(cell.value) for cell in ws[1] if cell.value]
                if ws.max_row >= 2:
                    test_info = {}
                    for i, header in enumerate(headers):
                        if i < len(ws[2]):
                            value = self._convert_value(ws[2][i].value)
                            test_info[header] = value
                    metadata["Test_Information"] = test_info
            except Exception as e:
                logger.debug("Error reading Test Information: %s", e)

    def _read_proc_info(self, wb: openpyxl.Workbook, metadata: dict[str, Any]) -> None:
        """Read Ch1_Proc sheet and extract Work Mode table."""
        if "Ch1_Proc" in wb.sheetnames:
            try:
                ws = wb["Ch1_Proc"]
                proc_info = {}
                work_mode_table = []
                table_headers = None

                for row in ws.iter_rows(values_only=True):
                    if not row[0]:
                        continue
                    first_col = str(row[0]) if row[0] else ""
                    if first_col == "Order":
                        table_headers = [str(cell) if cell else "" for cell in row]
                        table_headers = [h for h in table_headers if h]
                        continue
                    if table_headers:
                        if not row[0] or (isinstance(row[0], str) and not row[0].strip()):
                            break
                        table_row = {}
                        for i, header in enumerate(table_headers):
                            value = self._convert_value(row[i]) if i < len(row) else None
                            table_row[header] = value
                        work_mode_table.append(table_row)
                    elif row[1]:
                        proc_info[first_col] = self._convert_value(row[1])
                if work_mode_table:
                    proc_info["Work_Mode"] = work_mode_table
                metadata["Channel_Process_Info"] = proc_info
            except Exception as e:
                logger.debug("Error reading Channel Process Info: %s", e)

    def _read_log_info(self, wb: openpyxl.Workbook, metadata: dict[str, Any]) -> None:
        """Read Log sheet."""
        if "Log" in wb.sheetnames:
            try:
                ws = wb["Log"]
                headers = [str(cell.value) for cell in ws[1] if cell.value]
                log_info: list[dict[str, Any]] = []
                for row in ws.iter_rows(min_row=2, values_only=True):
                    if row[0]:
                        log_entry = {}
                        for i, header in enumerate(headers):
                            value = self._convert_value(row[i]) if i < len(row) else None
                            log_entry[header] = value
                        log_info.append(log_entry)
                metadata["Log_Info"] = log_info
            except Exception as e:
                logger.debug("Error reading Log Info: %s", e)

    @staticmethod
    def _find_data_sheet(wb: openpyxl.Workbook) -> str | None:
        """Find DefaultGroup sheet."""
        for sheet_name in wb.sheetnames:
            if "DefaultGroup" in sheet_name:
                return sheet_name
        return None

    @staticmethod
    def _convert_value(value: Any) -> Any:
        """Convert datetime/timedelta objects to string format."""
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        elif hasattr(value, "total_seconds"):
            return str(value)
        return value

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata to keep only essential fields."""
        cleaned = {}

        # Test Information fields to keep
        test_info_keys = ["Test Name", "Start Time", "Finish Time", "Active material"]
        if "Test_Information" in metadata:
            test_info = metadata["Test_Information"]
            cleaned_test_info = {k: test_info[k] for k in test_info_keys if k in test_info}

            # Parse and split "Active material" field if needed
            if "Active material" in cleaned_test_info:
                active_material_str = str(cleaned_test_info["Active material"])
                if " Active material: " in active_material_str:
                    parts = active_material_str.split(" Active material: ")
                    if len(parts) == 2:
                        capacity = parts[0].replace("Nominal specific capacity: ", "").strip()
                        material = parts[1].strip()
                        cleaned_test_info["nominal_specific_capacity"] = capacity
                        cleaned_test_info["active_material_mass"] = material

            cleaned["test_information"] = cleaned_test_info

        # Channel Process Info fields to keep
        proc_info_keys = ["Channel Number", "Name", "Description", "Unit Scheme", "Safety", "Work_Mode"]
        if "Channel_Process_Info" in metadata:
            proc_info = metadata["Channel_Process_Info"]
            cleaned["channel_process_info"] = {k: proc_info[k] for k in proc_info_keys if k in proc_info}

        # Ensure technique is set
        cleaned["technique"] = metadata.get("technique", "echem")

        # Merge with original metadata to ensure nothing is lost
        final_meta = cleaned.copy()
        final_meta.update(metadata)
        return final_meta

    @staticmethod
    def _clean_data(data: dict[str, Any]) -> dict[str, Any]:
        """Clean data to keep only specified columns from record-level data."""
        # For now, we keep all columns as per "不需要统一列的名字"
        if not data:
            return {}

        cleaned_data = data.copy()
        # Remove internal metadata that shouldn't be in the Dataset variables
        if "_metadata" in cleaned_data:
            del cleaned_data["_metadata"]

        return cleaned_data
