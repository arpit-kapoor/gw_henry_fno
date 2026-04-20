from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def append_result_row(csv_path: Path, row: dict[str, object], fieldnames: Iterable[str]) -> None:
    """Append one result row to CSV, writing header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
