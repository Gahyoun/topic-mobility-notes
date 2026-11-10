"""Figure style: 170 mm width, Arial, small size hierarchy (K-STEM figure convention)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = 170 / 25.4
FS = 7
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS + 0.5,
    "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1, "legend.fontsize": FS - 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
})
# subject palette (colour-blind safe, Okabe-Ito)
C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
SUBJ = {"physics": "#0072B2", "chemistry": "#E69F00", "biology": "#009E73", "mathematics": "#CC79A7"}
GREY = "#8C8C8C"
