# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Create the L03 binary place-value diagram: uv run scripts/binary_place_values.py."""
from pathlib import Path
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 2.8))
ax.axis("off")
table = ax.table(
    cellText=[
        ["Place", "2³", "2²", "2¹", "2⁰", "·", "2⁻¹", "2⁻²", "2⁻³"],
        ["Weight", "8", "4", "2", "1", "·", "1/2", "1/4", "1/8"],
        ["Digit", "1", "1", "0", "1", "·", "1", "0", "1"],
    ], cellLoc="center", loc="center", colWidths=[.14] + [.095] * 8,
)
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 1.8)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#d4dce2")
    if row == 0 or col == 0:
        cell.set_facecolor("#edf3f7")
    if row == 2 and col in [1, 2, 4, 6, 8]:
        cell.set_facecolor("#d7e8d3")
ax.set_title("The binary point separates positive and negative powers of two", fontsize=14)
fig.text(.5, .06, "1101.101₂ = 8 + 4 + 1 + 1/2 + 1/8 = 13.625₁₀", ha="center", fontsize=15)
fig.tight_layout(rect=[0, .12, 1, 1])
fig.savefig(Path(__file__).resolve().parents[1] / "assets/L03-binary-place-values.png", dpi=180)
