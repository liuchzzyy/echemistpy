#!/usr/bin/env python3
"""Utilities for parsing LANHE *.ccs electrochemistry files.

This module was created by reverse engineering the example file found in
``examples/echem/LANHE_GPCL.ccs``.  The format is a proprietary binary layout
used by LANHE battery cyclers.  We do not have an official specification, so
this reader relies on observations of the example file:

* The file begins with a ~4 kB header that stores metadata as fixed-length
  zero-terminated ASCII strings along with several numeric calibration values.
* The remainder of the file is composed of fixed-size 128 byte blocks.  Each
  block starts with two 32-bit integers (``tag`` and ``channel_id``) followed by
  six 20-byte sample groups.  Every sample group stores a time delta (in
  milliseconds) and four little-endian ``float32`` readings.
* The ``tag`` value identifies the semantic meaning of the samples.  In the
  example file the following tags were observed:

  ``0x0603`` (1539)
      Primary time-series samples (voltage + other channels).
  ``0x0103`` (259) and ``0x0203`` (515)
      Short configuration/sample blocks that appear at section boundaries.
  ``0x0002``
      Marks the end of a channel/section.  Sample payloads in these blocks are
      empty and are ignored by this parser.

The parser below extracts the metadata, counts block types, and reconstructs the
primary data stream (``tag == 0x0603``).  The remaining block types are
preserved so that callers can inspect them if needed.
"""

from __future__ import annotations

import argparse
import csv
import struct
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

BLOCK_SIZE = 128
HEADER_SIZE = 0x1000
SAMPLES_PER_BLOCK = 6
SAMPLE_STRUCT = struct.Struct("<Iffff")
TAG_NAMES = {
    0x0603: "data_points",
    0x0103: "segment_config_a",
    0x0203: "segment_config_b",
    0x0002: "section_terminator",
}


@dataclass
class SampleRecord:
    """Represents a decoded sample from a LANHE data block."""

    block_index: int
    tag: int
    channel_id: int
    delta_ms: int
    elapsed_s: float
    values: Sequence[float]

    @property
    def tag_name(self) -> str:
        return TAG_NAMES.get(self.tag, f"unknown_{self.tag:#06x}")


class LanheReader:
    """Parser for LANHE ``*.ccs`` files."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._data = self.path.read_bytes()
        if len(self._data) < HEADER_SIZE:
            raise ValueError("File is too small to contain the LANHE header")

        self.metadata = self._parse_metadata()
        self.block_counts = self._summarize_blocks()
        self.samples = list(self._decode_samples())

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _read_string(self, offset: int, length: int) -> str:
        raw = self._data[offset : offset + length]
        raw = raw.split(b"\x00", 1)[0]
        return raw.decode("utf-8", errors="ignore").strip()

    def _parse_metadata(self) -> dict[str, str | list[dict[str, str]]]:
        """Extract human readable metadata from the fixed header."""

        def field(key: str, offset: int, length: int) -> str:
            return self._read_string(offset, length)

        meta: dict[str, str | list[dict[str, str]]] = {
            "group_name": field("group_name", 0x10, 0x40),
            "computer": field("computer", 0x58, 0x20),
            "project_guid": field("project_guid", 0x78, 0x40),
            "test_name": field("test_name", 0xB0, 0x40),
            "start_date": field("start_date", 0xF0, 0x20),
            "column_signature": field("column_signature", 0x1F0, 0x80),
            "software_version": field("software_version", 0x2D0, 0x20),
            "equipment_id": field("equipment_id", 0x360, 0x20),
            "connection": field("connection", 0x444, 0x20),
            "firmware_version": field("firmware_version", 0x464, 0x10),
            "controller_version": field("controller_version", 0x474, 0x10),
            "hardware_date": field("hardware_date", 0x484, 0x10),
            "manufacturer": field("manufacturer", 0x494, 0x20),
            "calibration_date": field("calibration_date", 0x4D4, 0x10),
            "serial_number": field("serial_number", 0x4E4, 0x20),
            "program_guid": field("program_guid", 0x704, 0x30),
            "program_name": field("program_name", 0x734, 0x40),
            "program_description": field("program_description", 0x774, 0x40),
        }

        meta["operator_log"] = self._parse_operator_log()
        return meta

    def _parse_operator_log(self) -> list[dict[str, str]]:
        """Best-effort decoding of the operator/timestamp entries in the header."""

        log_entries: list[dict[str, str]] = []
        entry_start = 0x504
        entry_size = 0x40
        end = 0x700

        while entry_start + 0x20 <= end:
            chunk = self._data[entry_start : entry_start + 0x20]
            if not chunk.strip(b"\x00"):
                break

            parts = chunk.split(b"\x00", 1)
            user_bytes = parts[0]
            remainder = parts[1] if len(parts) > 1 else b""
            user = user_bytes.decode("utf-8", errors="ignore").strip()

            remainder = remainder.lstrip(b"\x00")
            timestamp = remainder.split(b"\x00", 1)[0].decode(
                "utf-8", errors="ignore",
            ).strip()

            log_entries.append({"user": user, "timestamp": timestamp})
            entry_start += entry_size

        return log_entries

    # ------------------------------------------------------------------
    # Block/sample decoding
    # ------------------------------------------------------------------
    def _summarize_blocks(self) -> Counter[tuple[int, int]]:
        counts: Counter[tuple[int, int]] = Counter()
        for block_index, offset in self._block_offsets():
            tag, channel = struct.unpack_from("<II", self._data, offset)
            counts[tag, channel] += 1
        return counts

    def _block_offsets(self) -> Iterator[tuple[int, int]]:
        payload = self._data[HEADER_SIZE:]
        block_count = len(payload) // BLOCK_SIZE
        base = HEADER_SIZE
        for block_index in range(block_count):
            yield block_index, base + block_index * BLOCK_SIZE

    def _decode_samples(self) -> Iterator[SampleRecord]:
        channel_elapsed: dict[tuple[int, int], int] = defaultdict(int)

        for block_index, offset in self._block_offsets():
            tag, channel = struct.unpack_from("<II", self._data, offset)
            if tag == 0x0002:
                # Terminator blocks do not carry sample payloads.
                continue

            block_payload_offset = offset + 8
            for sample_idx in range(SAMPLES_PER_BLOCK):
                base = block_payload_offset + sample_idx * SAMPLE_STRUCT.size
                dt, v1, v2, v3, v4 = SAMPLE_STRUCT.unpack_from(self._data, base)
                if dt == 0 and v1 == 0 and v2 == 0 and v3 == 0 and v4 == 0:
                    continue
                channel_elapsed[tag, channel] += dt
                elapsed_s = channel_elapsed[tag, channel] / 1000.0
                yield SampleRecord(
                    block_index=block_index,
                    tag=tag,
                    channel_id=channel,
                    delta_ms=dt,
                    elapsed_s=elapsed_s,
                    values=(v1, v2, v3, v4),
                )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def iter_samples(
        self, tag_filter: int | None = None, channel_filter: int | None = None,
    ) -> Iterator[SampleRecord]:
        """Yield decoded :class:`SampleRecord` objects that match the filters.

        Parameters
        ----------
        tag_filter:
            If provided, only records whose ``tag`` matches this value are
            returned.
        channel_filter:
            If provided, restricts the stream to the specified ``channel_id``.

        Notes
        -----
        The method operates on the cached ``samples`` list built during
        initialization, so iterating multiple times does not require re-parsing
        the binary payload.  This makes it cheap to pull different tag/channel
        subsets for downstream analysis or previews.
        """
        for record in self.samples:
            if tag_filter is not None and record.tag != tag_filter:
                continue
            if channel_filter is not None and record.channel_id != channel_filter:
                continue
            yield record

    def export_csv(
        self,
        destination: Path | str,
        tag_filter: int | None = 0x0603,
        channel_filter: int | None = None,
    ) -> None:
        """Dump the decoded samples to a CSV file."""

        dest_path = Path(destination)
        fieldnames = [
            "block_index",
            "tag",
            "tag_name",
            "channel_id",
            "delta_ms",
            "elapsed_s",
            "value1",
            "value2",
            "value3",
            "value4",
        ]
        with dest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.iter_samples(tag_filter=tag_filter, channel_filter=channel_filter):
                v1, v2, v3, v4 = record.values
                writer.writerow(
                    {
                        "block_index": record.block_index,
                        "tag": record.tag,
                        "tag_name": record.tag_name,
                        "channel_id": record.channel_id,
                        "delta_ms": record.delta_ms,
                        "elapsed_s": f"{record.elapsed_s:.3f}",
                        "value1": f"{v1:.9f}",
                        "value2": f"{v2:.9f}",
                        "value3": f"{v3:.9f}",
                        "value4": f"{v4:.9f}",
                    },
                )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def format_metadata(metadata: dict[str, str | list[dict[str, str]]]) -> str:
    """Return a human readable version of the parsed LANHE metadata.

    Examples
    --------
    >>> print(format_metadata({
    ...     "group_name": "Cathode study",
    ...     "test_name": "Cycle-01",
    ...     "operator_log": [{"user": "Alice", "timestamp": "2024-05-01"}]
    ... }))
      Group/Project: Cathode study
      Test name: Cycle-01
      Operator log:
        - Alice @ 2024-05-01
    """
    ordered_keys = [
        ("group_name", "Group/Project"),
        ("test_name", "Test name"),
        ("start_date", "Start date"),
        ("computer", "Host"),
        ("project_guid", "Project GUID"),
        ("column_signature", "Header signature"),
        ("software_version", "Software version"),
        ("equipment_id", "Equipment"),
        ("connection", "Connection"),
        ("firmware_version", "Firmware"),
        ("controller_version", "Controller"),
        ("hardware_date", "Hardware date"),
        ("manufacturer", "Manufacturer"),
        ("calibration_date", "Calibration"),
        ("serial_number", "Serial number"),
        ("program_guid", "Program GUID"),
        ("program_name", "Program name"),
        ("program_description", "Program description"),
    ]
    lines: list[str] = []
    for key, label in ordered_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            lines.append(f"  {label}: {value}")
    operators = metadata.get("operator_log")
    if isinstance(operators, list) and operators:
        lines.append("  Operator log:")
        for entry in operators:
            user = entry.get("user", "<unknown>")
            timestamp = entry.get("timestamp", "")
            lines.append(f"    - {user} @ {timestamp}")
    return "\n".join(lines)


def format_block_summary(counts: Counter[tuple[int, int]]) -> str:
    """Describe how often each ``(tag, channel)`` pair appears in the file.

    Examples
    --------
    >>> from collections import Counter
    >>> counts = Counter({(0x0603, 1): 2, (0x0002, 1): 1})
    >>> print(format_block_summary(counts))
    Tag/Channel usage:
      tag 0x0002 (section_terminator), channel 0x0001: 1 blocks
      tag 0x0603 (data_points), channel 0x0001: 2 blocks
    """
    lines = ["Tag/Channel usage:"]
    for (tag, channel), count in sorted(counts.items()):
        name = TAG_NAMES.get(tag, f"unknown_{tag:#06x}")
        lines.append(
            f"  tag {tag:#06x} ({name}), channel {channel:#06x}: {count} blocks",
        )
    return "\n".join(lines)


def preview_samples(reader: LanheReader, limit: int = 5) -> str:
    """Format the first few decoded samples (default five) for display."""
    rows: list[str] = []
    for idx, record in enumerate(reader.iter_samples(tag_filter=0x0603)):
        if idx >= limit:
            break
        v1, v2, v3, v4 = record.values
        rows.append(
            f"  t={record.elapsed_s:8.3f}s, Δt={record.delta_ms:5d} ms, "
            f"values=({v1:.9f}, {v2:.9f}, {v3:.9f}, {v4:.9f})",
        )
    if not rows:
        rows.append("  (no samples decoded)")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Parse LANHE *.ccs electrochemistry files and print their contents.",
    )
    parser.add_argument("file", type=Path, help="Path to the *.ccs file")
    parser.add_argument(
        "--csv",
        type=Path,
        help="If provided, dump the decoded samples (tag 0x0603) to this CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of data rows to show in the stdout preview (default: 5).",
    )
    args = parser.parse_args(argv)

    reader = LanheReader(args.file)
    print("\n=== Metadata ===")
    print(format_metadata(reader.metadata))
    print("\n=== Block summary ===")
    print(format_block_summary(reader.block_counts))
    print("\n=== Sample preview (tag 0x0603) ===")
    print(preview_samples(reader, limit=args.limit))

    if args.csv:
        reader.export_csv(args.csv)
        print(f"\nWrote CSV data to {args.csv}")


if __name__ == "__main__":
    main()
