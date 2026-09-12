import csv

log_files = [
    "logs/level1_sarsa.csv", "logs/level1_q_learning.csv",
    "logs/level2_sarsa.csv", "logs/level2_q_learning.csv",
    "logs/level3_sarsa.csv", "logs/level3_q_learning.csv",
]
for f in log_files:
    with open(f, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        deaths = sum(1 for r in rows if r["died"] == "1")
        print(f"{f}: {deaths}/{len(rows)} deaths")