import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Data Preparation
# -----------------------------------------------------------------------------
materials = ['CFRP', 'Aluminium', 'Lithium Ion', 'Titanium', 'Copper']

# Raw values extracted from the table (using 0.0 for '-')
energy_primary = [355.10, 11.20, 1033.20, 66.36, 4.59]
energy_recycled = [0.0, 0.9, 0.0, 31.0, 0.757]

gwp_primary = [17.0, 0.6, 101.7, 3.53, 0.157]
gwp_recycled = [0.0, 0.03, 0.0, 1.64, 0.0260]

# Consistent color mapping for each material across all charts
material_colors = {
    'CFRP': '#264653',
    'Aluminium': '#f4a261',
    'Lithium Ion': '#2a9d8f',
    'Titanium': '#e9c46a',
    'Copper': '#e76f51'
}

# Helper function to filter out missing data and sort from largest to smallest
def prepare_pie_data(labels, values):
    filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
    filtered.sort(key=lambda x: x[1], reverse=True)  # Sort descending
    
    lbls = [x[0] for x in filtered]
    vals = [x[1] for x in filtered]
    clrs = [material_colors[l] for l in lbls]
    return lbls, vals, clrs

# -----------------------------------------------------------------------------
# 2. Plot Configuration & Layout
# -----------------------------------------------------------------------------
# Create a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='white')
axes = axes.flatten()

titles = [
    r"Energy to Produce (Primary) [$\text{MJ}$]",
    r"Energy to Produce (Recycled) [$\text{MJ}$]",
    r"GWP (Primary) [$\text{kgCO}_2$]",
    r"GWP (Recycled) [$\text{kgCO}_2$]"
]

all_metrics = [energy_primary, energy_recycled, gwp_primary, gwp_recycled]

# -----------------------------------------------------------------------------
# 3. Generating the Pie Charts
# -----------------------------------------------------------------------------
for ax, title, raw_values in zip(axes, titles, all_metrics):
    lbls, vals, clrs = prepare_pie_data(materials, raw_values)
    
    # Slightly explode the largest contributor for visual emphasis
    explode = [0.05] + [0] * (len(vals) - 1)
    
    wedges, texts, autotexts = ax.pie(
        vals,
        labels=lbls,
        autopct='%1.1f%%',
        startangle=140,
        colors=clrs,
        explode=explode,
        pctdistance=0.75,
        textprops=dict(color="black", fontsize=11)
    )
    
    # Text styling inside the pie slices
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(10)
        
    # Text styling outside the pie slices
    for text in texts:
        text.set_weight('medium')
        text.set_fontsize(11)
        
    ax.set_title(title, fontsize=14, pad=15, weight='bold', color='#1c1c1c')

# Global Title Layout
plt.suptitle('Environmental Impact Breakdown Per Drone', fontsize=18, weight='bold', y=0.98, color='#1c1c1c')
plt.tight_layout()

# Save the visualization
plt.savefig('environmental_impact_breakdown.png', dpi=300, bbox_inches='tight')