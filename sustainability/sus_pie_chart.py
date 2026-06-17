import matplotlib.pyplot as plt
import numpy as np

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

# Threshold below which a slice is considered "small" -> gets a leader-line
# label outside the pie instead of a label/percent crammed inside it.
SMALL_SLICE_THRESHOLD = 0.06  # 6%


def prepare_pie_data(labels, values):
    """Filter out missing data and sort from largest to smallest."""
    filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
    filtered.sort(key=lambda x: x[1], reverse=True)

    lbls = [x[0] for x in filtered]
    vals = [x[1] for x in filtered]
    clrs = [material_colors[l] for l in lbls]
    return lbls, vals, clrs


# -----------------------------------------------------------------------------
# 2. Plot Configuration & Layout
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor='white')

titles = [
    r"Energy to Produce (Primary) [$\mathbf{MJ}$]",
    r"GWP (Primary) [$\mathbf{kgCO_2}$]",
]

all_metrics = [energy_primary, gwp_primary]

# -----------------------------------------------------------------------------
# 3. Generating the Pie Charts
# -----------------------------------------------------------------------------
for ax, title, raw_values in zip(axes, titles, all_metrics):
    lbls, vals, clrs = prepare_pie_data(materials, raw_values)
    total = sum(vals)
    fractions = [v / total for v in vals]

    # Explode the largest contributor slightly for visual emphasis
    explode = [0.05] + [0] * (len(vals) - 1)

    # Don't pass `labels` to ax.pie at all -- we place every label ourselves
    # afterwards, which lets us route small-slice labels outward on leader
    # lines so nothing overlaps regardless of how thin the wedge is.
    wedges, autotexts = ax.pie(
        vals,
        labels=None,
        autopct=None,
        startangle=140,
        colors=clrs,
        explode=explode,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
    )

    # ---- Place percentage + name labels ----
    # Large slices: percentage centered inside the wedge, name as a
    # standard outer label.
    # Small slices: both percentage and name pushed outside the pie on a
    # short leader line, stacked so they never collide with one another.
    small_slice_idx = [i for i, f in enumerate(fractions) if f < SMALL_SLICE_THRESHOLD]

    for i, (wedge, lbl, frac) in enumerate(zip(wedges, lbls, fractions)):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        x, y = np.cos(ang), np.sin(ang)

        if i not in small_slice_idx:
            # Percentage inside the slice
            r_pct = 0.7 * (1 + (0.05 if i == 0 else 0))  # account for explode
            ax.text(
                x * r_pct, y * r_pct, f"{frac * 100:.1f}%",
                ha='center', va='center',
                color='white', weight='bold', fontsize=11,
            )
            # Name outside the slice
            r_lbl = 1.12
            ha = 'left' if x >= 0 else 'right'
            ax.text(
                x * r_lbl, y * r_lbl, lbl,
                ha=ha, va='center', fontsize=11.5, weight='medium', color='#1c1c1c',
            )

    # ---- Small slices: leader-line annotations, vertically de-collided ----
    if small_slice_idx:
        # Sort small slices by their angular position so we can stack their
        # labels top-to-bottom on whichever side they fall on without
        # letting two labels land on the same line.
        side_groups = {'left': [], 'right': []}
        for i in small_slice_idx:
            wedge = wedges[i]
            ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
            x, y = np.cos(ang), np.sin(ang)
            side = 'right' if x >= 0 else 'left'
            side_groups[side].append((i, ang, x, y))

        for side, items in side_groups.items():
            # Order top to bottom so stacked text reads naturally
            items.sort(key=lambda t: -t[3])
            n = len(items)
            for k, (i, ang, x, y) in enumerate(items):
                lbl = lbls[i]
                frac = fractions[i]

                # Anchor point on the wedge edge
                r_anchor = 1.0
                ax_, ay_ = x * r_anchor, y * r_anchor

                # Spread stacked labels vertically, away from center
                y_text = 1.32 - k * (0.5 if n > 1 else 0) if False else None
                # Simple vertical fan-out around the slice's natural y,
                # clamped and spaced to avoid overlap.
                base_y = 1.25
                spread = 0.22
                if n == 1:
                    y_text = y * base_y
                else:
                    y_text = base_y - k * spread if y >= 0 else -(base_y - (n - 1 - k) * spread)

                x_text = 1.38 if side == 'right' else -1.38
                ha = 'left' if side == 'right' else 'right'

                ax.annotate(
                    f"{lbl} ({frac * 100:.1f}%)",
                    xy=(ax_, ay_),
                    xytext=(x_text, y_text),
                    ha=ha, va='center',
                    fontsize=10.5, weight='medium', color='#1c1c1c',
                    arrowprops=dict(
                        arrowstyle='-', color='#888888', linewidth=0.9,
                        shrinkA=0, shrinkB=3,
                        connectionstyle='angle3'
                    ),
                )

    ax.set_title(title, fontsize=15, pad=20, weight='bold', color='#1c1c1c')
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.55, 1.55)

# Global Title Layout
plt.suptitle('Environmental Impact Breakdown Per Drone', fontsize=19, weight='bold', y=0.99, color='#1c1c1c')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the visualization
plt.savefig('environmental_impact_breakdown.png', dpi=600, bbox_inches='tight')
print("Saved.")