import csv

for f in ["logs/level1_sarsa.csv", "logs/level1_q_learning.csv"]:
    with open(f, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        deaths = sum(1 for r in rows if r["died"] == "1")
        print(f"{f}: {deaths}/1000 deaths")