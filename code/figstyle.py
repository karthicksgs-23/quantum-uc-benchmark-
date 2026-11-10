import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import matplotlib as mpl
PAL=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948"]
INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"
def apply():
    mpl.rcParams.update({"font.family":"Times New Roman","font.size":10,"axes.edgecolor":INK2,"axes.labelcolor":INK,
      "xtick.color":INK2,"ytick.color":INK2,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,
      "axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":2,"legend.frameon":False,"savefig.dpi":600,
      "figure.facecolor":"white","axes.facecolor":"white"})
