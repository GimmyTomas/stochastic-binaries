"""Shared matplotlib style for the figure scripts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Mathematica-like colors used in the paper's figures
COLORS = ["#5E81B5", "#E19C24", "#8FB032", "#EB6235", "#8778B3",
          "#C56E1A", "#5D9EC7", "#FFBF00"]


def setup():
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.prop_cycle": plt.cycler(color=COLORS),
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })
    return plt
