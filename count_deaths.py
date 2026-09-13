"""Small Part I log-inspection helper retained for teammate compatibility."""

import csv
from pathlib import Path

for path in (
    Path("part1/logs/level1_sarsa.csv"),
    Path("part1/logs/level1_q_learning.csv"),
):
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    deaths = sum(int(row["died"]) for row in rows)
    successes = sum(int(row["success"]) for row in rows)
    print(f"{path}: deaths={deaths}/{len(rows)}, successes={successes}/{len(rows)}")
