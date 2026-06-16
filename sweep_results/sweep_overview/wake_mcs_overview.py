import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# CHOOSE YOUR COLORMAP HERE
# Options include: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'YlGn_r'
# ==========================================
colormap_choice = "plasma"
output_filename = f"wake_mcs_overview_magma.png"

# 1. Define the updated axes data from the image
turbulence_intensity = [
    0.25,
    0.26,
    0.27,
    0.28,
    0.29,
    0.30,
    0.31,
    0.32,
    0.33,
    0.34,
    0.35,
    0.36,
    0.37,
    0.38,
    0.39,
    0.40,
]
integral_time_scale = [7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0]

# 2. Transcribe the new grid matrix data
data = [
    [8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 14, 14],  # 7.0
    [8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 12, 12, 13, 14],  # 6.5
    [7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12],  # 6.0
    [7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12],  # 5.5
    [7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11],  # 5.0
    [7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11],  # 4.5
    [7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11],  # 4.0
    [7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10],  # 3.5
    [7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10],  # 3.0
]

# 3. Create DataFrame
df = pd.DataFrame(
    data, index=integral_time_scale, columns=turbulence_intensity
)

# 4. Format string labels to append the "%" sign
annot_labels = df.astype(str) + "%"

# 5. Initialize the plot layout
fig, ax = plt.subplots(figsize=(14, 6))

# 6. Generate the heatmap
sns.heatmap(
    df,
    annot=annot_labels,
    fmt="",
    cmap=colormap_choice,
    linewidths=0.5,
    linecolor="black",
    cbar=True,
    cbar_kws={"label": r"Epsilon $\epsilon$ (%)"},
    ax=ax,
)

# 7. Labels, updated title, and visual formatting
plt.title(
    "Monte Carlo: Optimal Epsilon for Wake Drones",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
plt.xlabel("Turbulence Intensity", fontsize=12, fontweight="bold", labelpad=10)
plt.ylabel("Integral Time Scale", fontsize=12, fontweight="bold", labelpad=10)

# 8. Clean layout optimizations and save image
plt.tight_layout()
plt.savefig(output_filename, dpi=300)
plt.close()

print(f"Successfully saved Wake Simulation figure as: {output_filename}")