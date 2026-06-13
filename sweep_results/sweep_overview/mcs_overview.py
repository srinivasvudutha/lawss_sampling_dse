import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# CHOOSE YOUR COLORMAP HERE
# Options include: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'YlGn_r'
# ==========================================
colormap_choice = "plasma"
output_filename = f"inlet_mcs_overview_heatmap_plasma.png"

# 1. Define the axes data
turbulence_intensity = [
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.20,
]
integral_time_scale = [7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0]

# 2. Recreate the matrix data (percentages)
data = [
    [5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9],  # 7.0
    [5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9],  # 6.5
    [5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9],  # 6.0
    [5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8],  # 5.5
    [5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8],  # 5.0
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8],  # 4.5
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7],  # 4.0
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7],  # 3.5
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7],  # 3.0
]

# 3. Create DataFrame
df = pd.DataFrame(
    data, index=integral_time_scale, columns=turbulence_intensity
)

# 4. Create string annotations with the '%' symbol
annot_labels = df.astype(str) + "%"

# 5. Initialize the plot layout using subplots
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
    cbar_kws={"label": "Convergence Limit (%)"},
    ax=ax,
)

# 7. Add Labels and Formatting
plt.title(
    "INLET MONTE CARLO SIMULATION 6 MIN CONVERGENCE LIMIT",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
plt.xlabel("Turbulence Intensity", fontsize=12, fontweight="bold", labelpad=10)
plt.ylabel("Integral Time Scale", fontsize=12, fontweight="bold", labelpad=10)

# 8. Optimize and Save Figure (dpi=300 sets a high print quality)
plt.tight_layout()
plt.savefig(output_filename, dpi=300)
plt.close()

print(f"Successfully saved figure as: {output_filename}")