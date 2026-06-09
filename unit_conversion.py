import pandas as pd

INPUT_CSV = "jan_one_data.csv"
OUTPUT_CSV = "jan_one_data_wind_mps.csv"
MPH_TO_MPS = 0.44704

df = pd.read_csv(INPUT_CSV)

for col in list(df.columns):
    if ("Avg Wind Speed" in col or "Peak Wind Speed" in col) and "[MPH]" in col:
        new_col = col.replace("[MPH]", "[m/s]")
        df[new_col] = pd.to_numeric(df[col], errors="coerce") * MPH_TO_MPS
        df.drop(columns=[col], inplace=True)

# Keep turbulence intensity unchanged
ordered = []
for col in pd.read_csv(INPUT_CSV, nrows=0).columns:
    if ("Avg Wind Speed" in col or "Peak Wind Speed" in col) and "[MPH]" in col:
        ordered.append(col.replace("[MPH]", "[m/s]"))
    else:
        ordered.append(col)

df = df[ordered]
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved: {OUTPUT_CSV}")