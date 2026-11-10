import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/") + "/"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import figstyle as fs; fs.apply()
plt.rcParams["axes.grid"]=False
fig,ax=plt.subplots(figsize=(6.6,6.4)); ax.set_xlim(0,100); ax.set_ylim(-1,100); ax.axis("off")
FILL={"data":"#cde2fb","model":"#fde3d6","qubo":"#d3f0e4","eval":"#f0efec"}
EDGE={"data":fs.PAL[0],"model":fs.PAL[1],"qubo":fs.PAL[2],"eval":"#52514e"}
LW=1.0; AC=fs.INK2
def box(x0,y0,x1,y1,text,kind,size=9.5):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0,rounding_size=1.6",fc=FILL[kind],ec=EDGE[kind],lw=1.3))
    ax.text((x0+x1)/2,(y0+y1)/2,text,ha="center",va="center",fontsize=size,color=fs.INK,linespacing=1.2)
def line(pts):
    xs,ys=zip(*pts); ax.plot(xs,ys,color=AC,lw=LW,solid_capstyle="butt")
def arrow(pts):
    line(pts[:-1]+[pts[-1]])
    (x1,y1),(x2,y2)=pts[-2],pts[-1]
    ax.annotate("",xy=(x2,y2),xytext=(x1+(x2-x1)*0.999,y1+(y2-y1)*0.999),arrowprops=dict(arrowstyle="-|>",color=AC,lw=LW,mutation_scale=11,shrinkA=0,shrinkB=0))
# data row
box(4,88,30,98,"German grid data\n(2019, hourly)","data")
box(37,88,63,98,"Renewable share\n0%, 33.8%, 50%","data")
box(70,88,96,98,"Net load and\ntest days","data")
arrow([(30,93),(37,93)]); arrow([(63,93),(70,93)])
# model
box(30,72,70,81,"10-unit commitment model","model",10)
arrow([(83,88),(83,76.5),(70,76.5)])
# bus to solvers
xs=[11,30.5,50,69.5,89]
line([(50,72),(50,65)]); line([(11,65),(89,65)])
for x in xs: arrow([(x,65),(x,59)])
labels=["MILP\n(exact)","Priority list\nand GA","Direct\nQUBO","AL-QUBO\nloop","QAOA"]
kinds=["eval","eval","qubo","qubo","qubo"]
for x,l,k in zip(xs,labels,kinds): box(x-8.5,47,x+8.5,59,l,k)
# samplers
box(40,32,80,40,"SA, SQA, tabu","qubo")
arrow([(50,47),(50,40)]); arrow([(69.5,47),(69.5,40)])
# evaluation
box(4,15,46,24,"Repair and\neconomic dispatch","eval")
box(54,15,96,24,"Metrics and\nstatistics","eval")
arrow([(11,47),(11,24)]); arrow([(30.5,47),(30.5,24)]); arrow([(44,32),(44,24)])
arrow([(89,47),(89,24)])
arrow([(46,19.5),(54,19.5)])
box(54,0,96,8,"Results","eval",10)
arrow([(75,15),(75,8)])
fig.savefig(ROOT + "figures/Fig1_workflow.png",bbox_inches="tight",pad_inches=0.12)
print("ok")
