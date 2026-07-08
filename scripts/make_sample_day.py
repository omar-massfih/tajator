"""Regenerate tests/data/spy_sample_day.csv — a deterministic synthetic SPY day.

Scripted so the strategy sees exactly one clean setup: a fast late-morning
selloff into the premarket low (call entry), a bounce through the EMAs
(scale-outs), then a fade back to break-even (runner exit).
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DAY = datetime(2026, 6, 15, tzinfo=ET)  # a Monday

# (time "HH:MM", close price) waypoints; linear interpolation per minute between them.
WAYPOINTS = [
    ("08:00", 500.00),
    ("08:20", 500.80),  # premarket high
    ("08:50", 499.20),  # premarket low
    ("09:29", 500.90),
    ("09:30", 501.00),  # open ~0.34% above the premarket low — a tradable distance
    ("10:00", 502.00),  # first high (HOD)
    ("10:20", 500.80),
    ("10:40", 501.95),  # double top vs 10:00 high
    ("11:04", 500.50),  # orderly selloff...
    ("11:10", 499.30),  # ...turns fast into the premarket low -> call setup
    ("12:00", 501.50),  # bounce through 9 EMA and 50 EMA/VWAP
    ("13:30", 499.15),  # slow fade back through break-even -> runner exit
    ("16:00", 500.30),  # quiet drift into the close
]


def minute_path() -> list[tuple[datetime, float]]:
    points = [(DAY.replace(hour=int(t[:2]), minute=int(t[3:])), p) for t, p in WAYPOINTS]
    out: list[tuple[datetime, float]] = []
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        steps = int((t1 - t0).total_seconds() // 60)
        for i in range(steps):
            frac = i / steps
            t = t0 + timedelta(minutes=i)
            # tiny deterministic wiggle so bars aren't perfectly flat
            wiggle = 0.03 * math.sin(i * 1.7)
            out.append((t, round(p0 + (p1 - p0) * frac + wiggle, 2)))
    out.append(points[-1])
    return out


def main() -> None:
    path = Path(__file__).parents[1] / "tests" / "data" / "spy_sample_day.csv"
    rows = minute_path()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        prev = rows[0][1]
        for t, close in rows:
            high, low = max(prev, close) + 0.02, min(prev, close) - 0.02
            w.writerow([t.isoformat(), prev, round(high, 2), round(low, 2), close, 1000])
            prev = close
    print(f"wrote {len(rows)} bars to {path}")


if __name__ == "__main__":
    main()
