"""Validate a rail submission CSV against the Info Kit's stated format.

Rail and Door. ACV and SHM are deferred.

Checks, per Rail_Corrugation_Info_Kit.md section 3 (lines 109-118):

1. Filename is exactly ``rail_predictions.csv``.
2. Header is exactly ``file_id,prediction``.
3. Every ``prediction`` is one of Normal / Side I / Side II.
4. No empty rows, no empty fields.
5. Row count equals the number of Test files (68).

Also checks that ``file_id`` values are unique, since one row per file cannot
hold duplicates.

Exits 0 if every check passes, 1 otherwise, printing each violation.

    python scripts/validate_submission.py <path-to-csv>

The example submission at reference/04_Example_Submission/rail_predictions.csv
is an illustrative placeholder with 5 rows, not 68, and with placeholder
file_ids. Pass --expect-rows 5 (or --allow-any-rows) to check its format
without failing on the row count.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXPECTED_FILENAME = "rail_predictions.csv"
EXPECTED_HEADER = ["file_id", "prediction"]
VALID_PREDICTIONS = {"Normal", "Side I", "Side II"}
EXPECTED_ROWS = 68  # Test1.csv .. Test68.csv


def validate(
    path: Path, expected_rows: int | None = EXPECTED_ROWS
) -> list[str]:
    """Return a list of violation messages; empty means valid."""
    violations: list[str] = []

    if not path.is_file():
        return [f"File does not exist: {path}"]

    if path.name != EXPECTED_FILENAME:
        violations.append(
            f"Filename is {path.name!r}, expected {EXPECTED_FILENAME!r}"
        )

    # newline="" per csv module guidance, so embedded newlines are handled.
    # utf-8-sig strips a leading BOM if one is present and is identical to
    # utf-8 otherwise. Without it a BOM-prefixed file fails the header check
    # with an invisible difference ("﻿file_id" vs "file_id"), which is a
    # miserable thing to debug on submission day.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return violations + ["File is empty: no header, no rows"]

    header = rows[0]
    if header != EXPECTED_HEADER:
        violations.append(
            f"Header is {header!r}, expected {EXPECTED_HEADER!r}"
        )

    data_rows = rows[1:]

    if expected_rows is not None and len(data_rows) != expected_rows:
        violations.append(
            f"Row count is {len(data_rows)}, expected {expected_rows} "
            f"(one row per Test file)"
        )

    seen: dict[str, int] = {}
    for offset, row in enumerate(data_rows):
        line_no = offset + 2  # 1-based, header is line 1

        if not row or all(field.strip() == "" for field in row):
            violations.append(f"Line {line_no}: empty row")
            continue

        if len(row) != 2:
            violations.append(
                f"Line {line_no}: has {len(row)} field(s), expected 2 -- {row!r}"
            )
            continue

        file_id, prediction = row

        if file_id.strip() == "":
            violations.append(f"Line {line_no}: empty file_id")
        elif file_id in seen:
            violations.append(
                f"Line {line_no}: duplicate file_id {file_id!r} "
                f"(first seen on line {seen[file_id]})"
            )
        else:
            seen[file_id] = line_no

        if prediction.strip() == "":
            violations.append(f"Line {line_no}: empty prediction")
        elif prediction not in VALID_PREDICTIONS:
            violations.append(
                f"Line {line_no}: prediction {prediction!r} is not one of "
                f"{sorted(VALID_PREDICTIONS)}"
            )

    return violations



# --- Door ------------------------------------------------------------------

DOOR_FILENAME = "door_predictions.csv"
DOOR_HEADER = ["start_time", "end_time", "prediction"]
DOOR_LABELS = {"Normal", "Abnormal resistance"}


def _door_time(value: str) -> float | None:
    """Parse the native Year-M-D-H-M-S-ms format to epoch seconds, or None."""
    try:
        parts = [int(x) for x in value.strip().split("-")]
        if len(parts) != 7:
            return None
        from datetime import datetime

        y, mo, d, h, mi, s, ms = parts
        return datetime(y, mo, d, h, mi, s, ms * 1000).timestamp()
    except (ValueError, OverflowError):
        return None


def validate_door(path: Path) -> list[str]:
    """Validate a door_predictions.csv (Info Kit section 3).

    Unlike rail there is no fixed row count: Test is one continuous stream and
    the number of cycles is whatever the model finds. So this checks structure
    and internal consistency rather than a count.
    """
    violations: list[str] = []

    if not path.is_file():
        return [f"File does not exist: {path}"]

    if path.name != DOOR_FILENAME:
        violations.append(f"Filename is {path.name!r}, expected {DOOR_FILENAME!r}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return violations + ["File is empty: no header, no rows"]

    if rows[0] != DOOR_HEADER:
        violations.append(f"Header is {rows[0]!r}, expected {DOOR_HEADER!r}")

    spans: list[tuple[float, float, int]] = []
    for offset, row in enumerate(rows[1:]):
        line_no = offset + 2

        if not row or all(f.strip() == "" for f in row):
            violations.append(f"Line {line_no}: empty row")
            continue
        if len(row) < 3:
            violations.append(
                f"Line {line_no}: has {len(row)} field(s), expected at least 3 -- {row!r}"
            )
            continue

        start_raw, end_raw, prediction = row[0], row[1], row[2]

        if prediction not in DOOR_LABELS:
            violations.append(
                f"Line {line_no}: prediction {prediction!r} is not one of "
                f"{sorted(DOOR_LABELS)}"
            )

        start = _door_time(start_raw)
        end = _door_time(end_raw)
        if start is None:
            violations.append(f"Line {line_no}: unparseable start_time {start_raw!r}")
        if end is None:
            violations.append(f"Line {line_no}: unparseable end_time {end_raw!r}")
        if start is not None and end is not None:
            if end <= start:
                violations.append(
                    f"Line {line_no}: end_time is not after start_time "
                    f"({start_raw!r} -> {end_raw!r})"
                )
            else:
                spans.append((start, end, line_no))

    # Overlapping predicted segments cannot both be right: matching is
    # one-to-one, so an overlap guarantees at least one false positive.
    spans.sort()
    for (s1, e1, l1), (s2, e2, l2) in zip(spans, spans[1:]):
        if s2 < e1:
            violations.append(
                f"Line {l2}: segment overlaps the one on line {l1} "
                f"(previous ends after this one starts)"
            )

    if not spans and len(rows) > 1:
        violations.append("No valid segments parsed")

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a rail_predictions.csv submission file."
    )
    parser.add_argument("path", type=Path, help="Path to the submission CSV")
    parser.add_argument(
        "--subsystem",
        choices=("rail", "door", "auto"),
        default="auto",
        help="Which schema to check (default: inferred from the filename)",
    )
    parser.add_argument(
        "--expect-rows",
        type=int,
        default=EXPECTED_ROWS,
        help=f"Expected data row count (default: {EXPECTED_ROWS})",
    )
    parser.add_argument(
        "--allow-any-rows",
        action="store_true",
        help="Skip the row-count check entirely",
    )
    args = parser.parse_args(argv)

    subsystem = args.subsystem
    if subsystem == "auto":
        subsystem = "door" if "door" in args.path.name.lower() else "rail"

    if subsystem == "door":
        violations = validate_door(args.path)
    else:
        expected = None if args.allow_any_rows else args.expect_rows
        violations = validate(args.path, expected_rows=expected)

    if violations:
        print(f"INVALID: {args.path} ({len(violations)} violation(s))")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print(f"VALID: {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
