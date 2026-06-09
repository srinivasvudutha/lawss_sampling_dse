import re
import csv

input_path = "Pasted.txt"
output_path = "output.csv"

def sim_to_seconds(sim_str):
    """
    Converts strings like '13m 19s' -> 799
    """
    m = re.search(r"(\d+)m\s*(\d+)", sim_str)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None

rows = []

with open(input_path, "r") as f:
    for line in f:
        # Skip irrelevant lines
        if "[" not in line:
            continue

        try:
            run_part = re.search(r"\[(\s*\d+)/(\d+)\]", line)
            iu = re.search(r"IU=([\d.]+)", line)
            T = re.search(r"T=([\d.]+)", line)
            seed = re.search(r"seed=(\d+)", line)
            nodes = re.search(r"nodes=\s*(\d+)", line)
            sim = re.search(r"sim=([\dm\s]+)", line)
            dist = re.search(r"dist=\s*([\d.]+)", line)
            wall = re.search(r"\(([\d.]+)s wall\)", line)

            # Only keep fully valid rows
            if all([run_part, iu, T, seed, nodes, sim, dist, wall]):
                rows.append([
                    int(run_part.group(1)),          # run
                    int(run_part.group(2)),          # total
                    float(iu.group(1)),              # IU
                    float(T.group(1)),               # T
                    int(seed.group(1)),              # seed
                    int(nodes.group(1)),             # nodes
                    sim_to_seconds(sim.group(1)),    # sim_time (seconds)
                    float(dist.group(1)),            # distance_m
                    float(wall.group(1))             # wall_s
                ])

        except Exception:
            # Skip bad lines instead of crashing
            continue

# Write CSV
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run", "total", "IU", "T", "seed",
        "nodes", "sim_time_s", "distance_m", "wall_s"
    ])
    writer.writerows(rows)

print(f"Done. Wrote {len(rows)} rows to {output_path}")