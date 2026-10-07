"""True-vector endpoint and validation schematic; illustrations are conceptual."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    Circle,
    Polygon,
    FancyArrowPatch,
    Rectangle,
)

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
fig = plt.figure(figsize=(11, 8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 110)
ax.set_ylim(0, 80)
ax.axis("off")
ink = "#18324A"
blue = "#DDECF6"
yellow = "#FFF0BE"
grey = "#E7E7E7"


def text(x, y, s, **kw):
    defaults = {"color": ink, "ha": "center", "va": "center"}
    defaults.update(kw)
    ax.text(x, y, s, **defaults)


def box(x, y, w, h, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.5,rounding_size=1",
            facecolor=color,
            edgecolor=ink,
            lw=0.8,
        )
    )


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=ink, lw=1.2
        )
    )


def token(x, y, study, shape="circle", excluded=False):
    if shape == "circle":
        patch = Circle((x, y), 1.9, facecolor="white", edgecolor=ink, lw=1)
    else:
        patch = Polygon(
            [(x, y + 2.1), (x - 2, y - 1.6), (x + 2, y - 1.6)],
            facecolor="white",
            edgecolor=ink,
            lw=1,
        )
    ax.add_patch(patch)
    text(x, y, study, fontsize=7)
    if excluded:
        ax.plot([x - 2.1, x + 2.1], [y - 2, y + 2.2], color="#A03D32", lw=1.5)


text(5, 76, "A", ha="left", fontweight="bold", fontsize=14)
text(11, 76, "Endpoint definition", ha="left", fontweight="bold", fontsize=14)
for x, w, label in [
    (5, 23, "Initial test material (F)"),
    (43, 24, "Terminal digesta (D)"),
    (82, 23, "Isolated micellar fraction (M)"),
]:
    box(x, 51, w, 20, "#F7FAFC")
    text(x + w / 2, 53.5, label, fontsize=10)
# Vector food symbols: leaf, tomato and carrot.
ax.add_patch(
    Polygon(
        [(10, 61), (12, 68), (21, 68), (18, 62)],
        facecolor="#79A96B",
        edgecolor=ink,
        lw=0.8,
    )
)
ax.plot([11, 20], [62, 67], color=ink, lw=0.7)
ax.add_patch(Circle((22, 61.5), 3.2, facecolor="#D86551", edgecolor=ink, lw=0.8))
ax.add_patch(
    Polygon([(13, 59), (16, 56), (18, 63)], facecolor="#E9A651", edgecolor=ink, lw=0.8)
)
for x, y in [(12, 65), (17, 65), (23, 64)]:
    ax.add_patch(Circle((x, y), 0.55, facecolor="#ECC35D", edgecolor=ink, lw=0.5))
# Beaker and separated fraction, all geometry remains vector.
ax.add_patch(Rectangle((49, 58), 12, 10, facecolor="none", edgecolor=ink, lw=1.2))
ax.add_patch(Rectangle((49.3, 58.3), 11.4, 6.4, facecolor="#F0D7A4", edgecolor="none"))
for x, y in [(51, 61), (55, 60), (59, 63)]:
    ax.add_patch(Circle((x, y), 0.6, facecolor="#B88645", edgecolor="none"))
ax.add_patch(
    Polygon(
        [(89, 69), (99, 69), (99, 59), (94, 56), (89, 59)],
        facecolor="#E5F2F8",
        edgecolor=ink,
        lw=1.1,
    )
)
for x, y in [(92, 65), (96, 65), (94, 61)]:
    ax.add_patch(Circle((x, y), 1.1, facecolor="#F3C750", edgecolor=ink, lw=0.6))
    for dx, dy in [(0, 1.3), (1.3, 0), (0, -1.3), (-1.3, 0)]:
        ax.add_patch(Circle((x + dx, y + dy), 0.24, facecolor=ink))
arrow(29, 63, 41, 63)
text(35, 68, "In vitro\ndigestion", fontsize=9)
arrow(68, 63, 80, 63)
text(74, 68, "Fraction\nisolation", fontsize=9)
box(27, 43, 56, 5, "#F3F6F8")
text(55, 45.5, "Bioaccessibility (%) = 100 × M / F", fontweight="bold", fontsize=12)
text(
    55,
    41,
    "M and F are amounts of the matching carotenoid or defined analyte pool.",
    fontsize=9,
)
text(5, 38.5, "B", ha="left", fontweight="bold", fontsize=14)
text(11, 38.5, "Validation designs", ha="left", fontweight="bold", fontsize=14)
for x, label, c in [
    (39, "Training", blue),
    (63, "Test", yellow),
    (87, "Excluded", grey),
]:
    box(x, 31.8, 20, 4, c)
    text(x + 10, 33.8, label, fontweight="bold")
rows = [
    (
        "Random rows",
        [("S1", "circle"), ("S2", "triangle"), ("S3", "circle")],
        [("S1", "circle"), ("S2", "triangle")],
        [],
    ),
    (
        "Held-out studies",
        [("S2", "circle"), ("S2", "triangle"), ("S3", "circle"), ("S3", "triangle")],
        [("S1", "circle"), ("S1", "triangle")],
        [],
    ),
    (
        "Held-out matrix category",
        [("S1", "triangle"), ("S2", "triangle"), ("S3", "triangle")],
        [("S1", "circle"), ("S2", "circle")],
        [],
    ),
    (
        "Matrix and contributing\nstudies withheld",
        [("S3", "triangle")],
        [("S1", "circle"), ("S2", "circle")],
        [("S1", "triangle"), ("S2", "triangle")],
    ),
]
for i, (label, train, test, excluded) in enumerate(rows):
    y = 28 - i * 6.3
    text(5, y, label, ha="left", fontsize=10)
    for x, items, c in [(39, train, blue), (63, test, yellow), (87, excluded, grey)]:
        box(x, y - 2.4, 20, 4.8, c)
        for j, (study, shape) in enumerate(items):
            token(x + 10 + (j - (len(items) - 1) / 2) * 4.5, y, study, shape, c == grey)
token(8, 1.7, "", "circle")
token(13, 1.7, "", "triangle")
text(17, 1.7, "Matrix categories; S1–S3 = study identifiers.", ha="left", fontsize=9)
text(
    105, 1.7, "Illustrative partitions, not actual fold counts.", ha="right", fontsize=9
)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(OUT / f"Figure_1.{ext}", dpi=350, facecolor="white")
plt.close(fig)
