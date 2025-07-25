import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import seaborn as sns

# Colors
color_codes = sns.color_palette("colorblind", 6)
cl1 = color_codes[0]
cl2 = color_codes[1]

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fs = 16

# Triangle vertices
triangle_vertices = np.array([[0, 0], [0.5, 1], [1, 0]])

# Panel 1: Whole triangle
ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.1, 1.1)
ax1.set_aspect('equal')

triangle1 = Polygon(triangle_vertices, facecolor='lightgray', edgecolor='black', linewidth=1.5)
ax1.add_patch(triangle1)
ax1.text(0.5, 0.3, 'T', fontsize=fs, ha='center', va='center')
ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)

# Panel 2: Horizontal division
ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-0.1, 1.1)
ax2.set_aspect('equal')

lower_vertices = np.array([[0, 0], [0.25, 0.5], [0.75, 0.5], [1, 0]])
lower_region = Polygon(lower_vertices, facecolor=cl1, edgecolor=cl1, linewidth=1, alpha=0.7)
ax2.add_patch(lower_region)

upper_vertices = np.array([[0.25, 0.5], [0.5, 1], [0.75, 0.5]])
upper_region = Polygon(upper_vertices, facecolor=cl2, edgecolor=cl2, linewidth=1, alpha=0.7)
ax2.add_patch(upper_region)

triangle2 = Polygon(triangle_vertices, facecolor='none', edgecolor='black', linewidth=1.5)
ax2.add_patch(triangle2)
ax2.plot([0, 1], [0.5, 0.5], 'k--', linewidth=1)
ax2.text(0.5, 0.25, r'$T_{X_{0,\alpha}}$', fontsize=fs, ha='center', va='center')
ax2.text(0.5, 0.75, r'$T_{X_{1,\alpha}}$', fontsize=fs, ha='center', va='center')
ax2.text(1.07, 0.55, r'$G^{-1}(\alpha)$', fontsize=fs, ha='right', va='center')
ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)

# Panel 3: Vertical division
ax3.set_xlim(-0.1, 1.1)
ax3.set_ylim(-0.1, 1.1)
ax3.set_aspect('equal')

left_vertices = np.array([[0, 0], [0.3, 0.6], [0.3, 0]])
left_region = Polygon(left_vertices, facecolor=cl1, edgecolor=cl1, linewidth=1, alpha=0.7)
ax3.add_patch(left_region)

right_vertices = np.array([[0.3, 0], [0.3, 0.6], [0.5, 1], [1, 0]])
right_region = Polygon(right_vertices, facecolor=cl2, edgecolor=cl2, linewidth=1, alpha=0.7)
ax3.add_patch(right_region)

triangle3 = Polygon(triangle_vertices, facecolor='none', edgecolor='black', linewidth=1.5)
ax3.add_patch(triangle3)
ax3.plot([0.3, 0.3], [0, 1], 'k--', linewidth=1)
ax3.text(0.2, 0.2, r'$T_{Y_{0,\alpha}}$', fontsize=fs, ha='center', va='center')
ax3.text(0.65, 0.2, r'$T_{Y_{1,\alpha}}$', fontsize=fs, ha='center', va='center')
ax3.text(0.2, 0.98, r'$F^{-1}(\alpha)$', fontsize=16, ha='left', va='bottom')
ax3.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='both', which='major', labelsize=fs-2)

# Add panel labels
for i, ax in enumerate([ax1, ax2, ax3]):
    letter = chr(ord('a') + i)
    ax.text(-0.08, 1.05, letter + ')', transform=ax.transAxes, fontsize=16)

plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.25)
plt.savefig('./triangle_visualization.pdf', format='pdf', bbox_inches='tight', dpi=300)
#plt.show()