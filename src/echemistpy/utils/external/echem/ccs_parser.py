"""
Enhanced LANHE CCS File Parser with Complete Header Metadata

This version extracts all identified header fields including:
- Basic metadata (test name, channel, equipment, GUID, barcode)
- Timestamps (start time, end time)
- Cycle and step counts
- Process parameters
- All other identified header fields

It uses a block-based extraction method to handle the 128-byte file structure
where records are split across blocks.
"""

import struct
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os


class CCSParser:
    """Enhanced parser for LANHE CCS files with complete metadata extraction."""
    
    # File structure constants
    HEADER_SIZE = 0xA8C  # 2700 bytes
    RECORD_SIZE = 20     # bytes per record
    
    # Known header field offsets
    OFFSET_CHANNEL = 0x010
    OFFSET_EQUIPMENT = 0x058
    OFFSET_GUID = 0x078
    OFFSET_TEST_NAME = 0x0B0
    OFFSET_BARCODE = 0x0F0
    OFFSET_TIMESTAMP_START = 0x0A0
    OFFSET_TIMESTAMP_END = 0x0A8
    OFFSET_CYCLE_COUNT = 0x0400
    OFFSET_STEP_COUNT = 0x072C
    OFFSET_RECORD_SIZE = 0x0A88
    
    def __init__(self, filename: str):
        """Initialize parser with CCS file."""
        self.filename = filename
        with open(filename, 'rb') as f:
            self.data = f.read()
        self.file_size = len(self.data)
        self.header_info = {}
        self.records = []
        
    def _extract_string(self, offset: int, max_len: int = 100) -> str:
        """Extract null-terminated UTF-8 string from data."""
        end = offset
        while end < offset + max_len and end < len(self.data) and self.data[end] != 0:
            end += 1
        try:
            return self.data[offset:end].decode('utf-8', errors='ignore')
        except:
            return ""
    
    def _extract_uint32(self, offset: int) -> int:
        """Extract unsigned 32-bit integer."""
        return struct.unpack('<I', self.data[offset:offset+4])[0]
    
    def _extract_uint64(self, offset: int) -> int:
        """Extract unsigned 64-bit integer."""
        return struct.unpack('<Q', self.data[offset:offset+8])[0]
    
    def _unix_ms_to_datetime(self, timestamp_ms: int) -> datetime:
        """Convert Unix timestamp (milliseconds) to datetime."""
        return datetime.utcfromtimestamp(timestamp_ms / 1000.0)
    
    def parse_header(self) -> Dict[str, Any]:
        """Extract all metadata from file header."""
        
        # Extract text fields
        channel = self._extract_string(self.OFFSET_CHANNEL, 40)
        equipment = self._extract_string(self.OFFSET_EQUIPMENT, 20)
        guid = self._extract_string(self.OFFSET_GUID, 40)
        test_name = self._extract_string(self.OFFSET_TEST_NAME, 100)
        barcode = self._extract_string(self.OFFSET_BARCODE, 20)
        
        # Extract timestamps (Unix milliseconds)
        timestamp_start_ms = self._extract_uint64(self.OFFSET_TIMESTAMP_START)
        timestamp_end_ms = self._extract_uint64(self.OFFSET_TIMESTAMP_END)
        
        try:
            start_time = self._unix_ms_to_datetime(timestamp_start_ms)
            end_time = self._unix_ms_to_datetime(timestamp_end_ms)
            duration = end_time - start_time
        except:
            start_time = None
            end_time = None
            duration = None
        
        # Extract counts
        cycle_count = self._extract_uint32(self.OFFSET_CYCLE_COUNT)
        step_count = self._extract_uint32(self.OFFSET_STEP_COUNT)
        record_size = self._extract_uint32(self.OFFSET_RECORD_SIZE)
        
        # Extract other identified fields
        unknown_0x0000 = self._extract_uint32(0x0000)
        unknown_0x0004 = self._extract_uint32(0x0004)
        unknown_0x0008 = self._extract_uint32(0x0008)
        unknown_0x0A58 = self._extract_uint32(0x0A58)
        unknown_0x0A6C = self._extract_uint32(0x0A6C)
        unknown_0x0A80 = self._extract_uint32(0x0A80)
        unknown_0x0A84 = self._extract_uint32(0x0A84)
        
        self.header_info = {
            # Basic metadata
            'file_size': self.file_size,
            'channel': channel,
            'equipment': equipment,
            'guid': guid,
            'test_name': test_name,
            'barcode': barcode,
            
            # Timestamps
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'timestamp_start_ms': timestamp_start_ms,
            'timestamp_end_ms': timestamp_end_ms,
            
            # Counts
            'cycle_count': cycle_count,
            'step_count': step_count,
            'record_size': record_size,
            
            # Unknown fields (for future analysis)
            'unknown_0x0000': unknown_0x0000,
            'unknown_0x0004': unknown_0x0004,
            'unknown_0x0008': unknown_0x0008,
            'unknown_0x0A58': unknown_0x0A58,
            'unknown_0x0A6C': unknown_0x0A6C,
            'unknown_0x0A80': unknown_0x0A80,
            'unknown_0x0A84': unknown_0x0A84,
        }
        
        return self.header_info
    
    def parse_records(self) -> List[Dict[str, Any]]:
        """Parse all measurement records using block-based extraction."""
        records = []
        
        # Block structure constants
        BLOCK_SIZE = 128
        BLOCK_HEADER_SIZE = 8
        DATA_START_OFFSET = 0xA80  # Start of the first block
        
        print(f"Reconstructing data stream from blocks starting at 0x{DATA_START_OFFSET:X}...")
        
        # Reconstruct continuous data stream
        data_stream = bytearray()
        offset = DATA_START_OFFSET
        
        while offset + BLOCK_SIZE <= len(self.data):
            # Read block
            block = self.data[offset:offset + BLOCK_SIZE]
            
            # Extract data payload (skip 8-byte header)
            payload = block[BLOCK_HEADER_SIZE:]
            data_stream.extend(payload)
            
            offset += BLOCK_SIZE
            
        # Handle remaining partial block if any
        if offset < len(self.data):
            remaining = self.data[offset:]
            if len(remaining) > BLOCK_HEADER_SIZE:
                data_stream.extend(remaining[BLOCK_HEADER_SIZE:])
        
        print(f"Reconstructed stream size: {len(data_stream)} bytes")
        
        # Parse records from reconstructed stream
        # Skip first 4 bytes (Record Size field: 14 00 00 00)
        stream_offset = 4
        record_num = 0
        cumulative_time_ms = 0
        
        # Get start time from header if available
        if not self.header_info:
            self.parse_header()
        
        start_time = self.header_info.get('start_time')
        
        while stream_offset + self.RECORD_SIZE <= len(data_stream):
            try:
                # Parse 20-byte record
                record_bytes = data_stream[stream_offset:stream_offset + self.RECORD_SIZE]
                
                voltage = struct.unpack('<f', record_bytes[0:4])[0]
                current = struct.unpack('<f', record_bytes[4:8])[0]
                capacity = struct.unpack('<f', record_bytes[8:12])[0]
                energy = struct.unpack('<f', record_bytes[12:16])[0]
                time_interval = struct.unpack('<i', record_bytes[16:20])[0]
                
                # Skip invalid records (all zeros)
                if voltage == 0 and current == 0 and time_interval == 0:
                    stream_offset += self.RECORD_SIZE
                    continue
                
                record_num += 1
                cumulative_time_ms += time_interval
                
                # Calculate absolute timestamp
                absolute_time = None
                if start_time:
                    try:
                        absolute_time = start_time + timedelta(milliseconds=cumulative_time_ms)
                    except:
                        pass
                
                record = {
                    'Record': record_num,
                    'Voltage/V': voltage,
                    'Current/uA': current,
                    'Capacity/uAh': capacity,
                    'Energy/uWh': energy,
                    'TimeInterval/ms': time_interval,
                    'TestTime/ms': cumulative_time_ms,
                    'TestTime/h': cumulative_time_ms / (1000 * 3600),
                    'SysTime': absolute_time,
                    'Cycle': self.header_info.get('cycle_count', 1),
                    'Step': 1
                }
                
                records.append(record)
                stream_offset += self.RECORD_SIZE
                
                # Progress indicator
                if record_num % 10000 == 0:
                    print(f"  Parsed {record_num} records...")
                    
            except Exception as e:
                print(f"Error at stream offset {stream_offset}: {e}")
                break
        
        print(f"Total records parsed: {len(records)}")
        self.records = records
        return records
    
    def export_to_csv(self, output_file: str):
        """Export parsed data to CSV using standard library."""
        if not self.records:
            print("No records to export")
            return
            
        print(f"Exporting {len(self.records)} records to {output_file}...")
        
        # Define column order
        fieldnames = [
            'Cycle', 'Step', 'Record',
            'Voltage/V', 'Current/uA', 'Capacity/uAh', 'Energy/uWh',
            'TimeInterval/ms', 'TestTime/ms', 'TestTime/h',
            'SysTime'
        ]
        
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.records)
            print(f"Export successful!")
        except Exception as e:
            print(f"Export failed: {e}")
    
    def print_header_info(self):
        """Print all header information in a readable format."""
        if not self.header_info:
            self.parse_header()
        
        print("\n" + "="*60)
        print("CCS FILE HEADER INFORMATION")
        print("="*60)
        
        print("\n--- Basic Metadata ---")
        print(f"File size:      {self.header_info['file_size']:,} bytes")
        print(f"Channel:        {self.header_info['channel']}")
        print(f"Equipment:      {self.header_info['equipment']}")
        print(f"GUID:           {self.header_info['guid']}")
        print(f"Test name:      {self.header_info['test_name']}")
        print(f"Barcode:        {self.header_info['barcode']}")
        
        print("\n--- Timestamps ---")
        if self.header_info['start_time']:
            print(f"Start time:     {self.header_info['start_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
        else:
            print(f"Start time:     (raw: {self.header_info['timestamp_start_ms']})")
        
        if self.header_info['end_time']:
            print(f"End time:       {self.header_info['end_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
        else:
            print(f"End time:       (raw: {self.header_info['timestamp_end_ms']})")
        
        if self.header_info['duration']:
            print(f"Duration:       {self.header_info['duration']}")
        
        print("\n--- Test Configuration ---")
        print(f"Cycle count:    {self.header_info['cycle_count']}")
        print(f"Step count:     {self.header_info['step_count']}")
        print(f"Record size:    {self.header_info['record_size']} bytes")
        
        print("\n--- Unknown Fields (for future analysis) ---")
        print(f"0x0000:         {self.header_info['unknown_0x0000']}")
        print(f"0x0004:         {self.header_info['unknown_0x0004']}")
        print(f"0x0008:         {self.header_info['unknown_0x0008']}")
        print(f"0x0A58:         {self.header_info['unknown_0x0A58']}")
        print(f"0x0A6C:         {self.header_info['unknown_0x0A6C']}")
        print(f"0x0A80:         {self.header_info['unknown_0x0A80']}")
        print(f"0x0A84:         {self.header_info['unknown_0x0A84']}")
        
        print("\n" + "="*60)


def main():
    """Main function."""
    ccs_file = 'LANHE_GPCL.ccs'
    
    if not os.path.exists(ccs_file):
        print(f"Error: {ccs_file} not found")
        return
    
    print("="*60)
    print("LANHE CCS File Parser - Enhanced Version")
    print("="*60)
    
    # Parse CCS file
    parser = CCSParser(ccs_file)
    
    # Parse and display header
    parser.parse_header()
    parser.print_header_info()
    
    # Parse records
    print("\n" + "="*60)
    print("PARSING MEASUREMENT RECORDS")
    print("="*60)
    records = parser.parse_records()
    
    # Export to CSV
    output_csv = 'parsed_ccs_data_full.csv'
    parser.export_to_csv(output_csv)
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total records:  {len(records)}")
    if len(records) > 0:
        voltages = [r['Voltage/V'] for r in records]
        currents = [r['Current/uA'] for r in records]
        test_times = [r['TestTime/h'] for r in records]
        
        print(f"Voltage range:  {min(voltages):.6f} - {max(voltages):.6f} V")
        print(f"Current range:  {min(currents):.6f} - {max(currents):.6f} µA")
        print(f"Test duration:  {max(test_times):.2f} hours")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
