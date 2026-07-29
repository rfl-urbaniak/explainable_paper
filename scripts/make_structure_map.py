"""
Paper-structure map: a single overview figure placed once, early in the paper (the
intro), with no per-section "you are here" highlighting. Layered design, chronological
top->bottom, fill-width grid per tier, appendices as a 2-row grid with letter tags
(A-H).

The `current=("overview",)` mode renders every non-appendix card the same neutral way
(no done/current/ahead distinction, no legend); the appendix tier keeps its distinct
dashed "supporting register" styling, since that's a structural category, not a
progress state. The underlying per-stage highlighting machinery
(`current=("tier",T)` / `("card",T,C)` / `("cards",T,S)`) is still here in case a
future revision wants it back, but STAGES below only produces the one overview image
now.

Color-blind safe: each status is encoded redundantly by color + border line-style + symbol
(verified in grayscale).

Run:  python3 scripts/make_structure_map.py
Output: figures/structure_map_<slug>.png
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------- palette ----------
TEAL="#1F9AA6"; GOLD="#F08A00"; INK="#1d2730"
DONE_FILL="#e1f3f4"; DONE_EDGE=TEAL
CUR_FILL ="#fff2e0"; CUR_EDGE =GOLD
FUT_FILL ="#f3f5f6"; FUT_EDGE ="#cad2d8"; FUT_TXT="#9aa6ae"
APP_FILL ="#edeef7"; APP_EDGE ="#8b80b8"; APP_TXT="#5f5880"
BAND_FILL="#f7f9fa"; BAND_EDGE="#e7ecef"
BAND_CUR ="#fff8ee"; BAND_CUR_EDGE="#f6dcb4"
TIER_TXT ="#7d8a94"

plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":220,
                     "savefig.bbox":"tight","figure.facecolor":"white",
                     "pdf.fonttype":42,"ps.fonttype":42})   # embed TrueType (no Type-3) for publishing

SOLID="solid"; DOTTED=(0,(1.4,1.6)); DASHED=(0,(5,2.3))
FLAT_FILL="#f3f5f6"; FLAT_EDGE="#c3ccd2"
def st(s):
    return {"done":(DONE_FILL,DONE_EDGE,INK   ,1.9,SOLID),
            "cur" :(CUR_FILL ,CUR_EDGE ,INK   ,3.0,SOLID),
            "app" :(APP_FILL ,APP_EDGE ,APP_TXT,1.7,DASHED),
            "fut" :(FUT_FILL ,FUT_EDGE ,FUT_TXT,1.6,DOTTED),
            "flat":(FLAT_FILL,FLAT_EDGE,INK   ,1.6,SOLID)}[s]

def badge(ax,x,y,status,s=170):
    f=max(6.0,(s**0.5)*0.6)
    if status=="done":
        ax.scatter([x],[y],s=s,color=TEAL,zorder=7,edgecolors="white",linewidths=1.3)
        ax.text(x,y,"✓",ha="center",va="center",color="white",fontsize=f,fontweight="bold",zorder=8)
    elif status=="cur":
        ax.scatter([x],[y],s=s+25,color=GOLD,zorder=7,edgecolors="white",linewidths=1.5)
        ax.text(x,y,"▶",ha="center",va="center",color="white",fontsize=f*0.8,zorder=8)
    elif status=="fut":
        ax.scatter([x],[y],s=s-40,facecolors="white",edgecolors=FUT_EDGE,linewidths=1.8,zorder=7)

def letter_tag(ax,x,y,text):
    w=0.30+0.16*len(text); h=0.34
    ax.add_patch(FancyBboxPatch((x-0.02,y-h+0.02),w,h,boxstyle="round,pad=0.01,rounding_size=0.06",
                 linewidth=0,facecolor=APP_EDGE,zorder=7))
    ax.text(x-0.02+w/2,y-h/2+0.02,text,ha="center",va="center",color="white",
            fontsize=7.5,fontweight="bold",zorder=8)

# ---------- content (defined once; statuses computed per stage) ----------
# each card = (title, tag)   tag only used for appendix letters
TIERS = [
 ("Inspirations","",
   [[("Actual Causality",None),
     ("Probability of Necessity\n& Sufficiency",None)]]),
 ("Definitions","& Running Example",
   [[("Variable\nSelection Dist.",None),
     ("Alternative\nValue Dist.",None),
     ("Necessity &\nSufficiency\nMeasure",None),
     ("Causal Impact",None)]]),
 ("Evaluation","Preview",
   [[("Preview of Empirical & Comparative Results",None)]]),
 ("Positioning","",
   [[("vs SHAP",None),
     ("vs Causal\nSHAP",None),
     ("vs Prob. of\nNecessity &\nSufficiency",None),
     ("vs Actual\nCausality",None),
     ("vs Prob. of\nActual\nCausality",None)]]),
 ("Appendices","Supporting Detail",
   [[("Running-Example\nComputations","A–C"),
     ("Synthetic Eval,\nContinuous Variables","D"),
     ("vs Differential\nCausal Effect","E")],
    [("Scaling Up\nvs Exact AC","F"),
     ("Dynamical SIR Model","G"),
     ("Real-World\nValuation Model","H")]]),
]
APP_TI = next(i for i,(n,*_) in enumerate(TIERS) if n=="Appendices")

def status_of(ti, ci, current):
    """current is ('tier', T) -> whole tier highlighted; ('card', T, C) -> one card
    highlighted; ('cards', T, S) -> the cards whose indices are in the set/tuple S;
    ('overview',) -> no highlighting, every non-appendix card rendered the same way."""
    if ti==APP_TI: return "app"
    if current[0]=="overview": return "flat"
    if current[0]=="tier":
        T=current[1]
        return "done" if ti<T else ("cur" if ti==T else "fut")
    if current[0]=="cards":
        _,T,S = current
        if ti<T: return "done"
        if ti>T: return "fut"
        if ci in S: return "cur"
        return "done" if ci < min(S) else "fut"
    _,T,C = current
    if ti<T: return "done"
    if ti>T: return "fut"
    return "done" if ci<C else ("cur" if ci==C else "fut")

# ---------- geometry ----------
W=12.6; CH=1.02; GAP=0.40; PADV=0.22; INNERGAP=0.30; TIERGAP=0.55; APP_GAP=1.45
LABELX=2.20; CX0=3.00; CX1=CX0+W; MIDX=(CX0+CX1)/2
def band_h(rows):
    R=len(rows); return R*CH + (R-1)*INNERGAP + 2*PADV

# ---------- one figure for a given stage ----------
def make_map(current, slug):
    fig, ax = plt.subplots()

    def draw_card(x,y,w,title,status,tag):
        fc,ec,tc,lw,ls = st(status)
        ax.add_patch(FancyBboxPatch((x,y),w,CH,boxstyle="round,pad=0.02,rounding_size=0.12",
                     linewidth=lw,edgecolor=ec,facecolor=fc,linestyle=ls,zorder=4))
        fn = 8.7 if w>=3.3 else (8.1 if w>=2.6 else 7.4)   # shrink text on narrow cards
        is_app = (status=="app" and tag)
        ty = (y+0.28) if is_app else (y+CH/2)              # appendix: title in lower strip, below the tag
        ax.text(x+w/2,ty,title,ha="center",va="center",fontsize=fn,
                fontweight="bold",color=tc,zorder=6,linespacing=1.08)
        if is_app: letter_tag(ax,x+0.10,y+CH-0.04,tag)
        elif status!="flat": badge(ax,x+0.07,y+CH-0.07,status,s=150)

    cur_y=0.0; band_box=[]
    for ti,(name,desc,rows) in enumerate(TIERS):
        bh=band_h(rows); b_top=cur_y; b_bot=cur_y-bh
        band_box.append((b_top,b_bot))
        is_app=(ti==APP_TI)
        flat=[c for row in rows for c in row]
        stats=[status_of(ti,ci,current) for ci in range(len(flat))]
        active=("cur" in stats)
        ax.add_patch(FancyBboxPatch((CX0-0.30,b_bot),W+0.60,bh,
                     boxstyle="round,pad=0.02,rounding_size=0.09",
                     linewidth=1.3,edgecolor=(BAND_CUR_EDGE if active else ("#c9c3e0" if is_app else BAND_EDGE)),
                     facecolor=(BAND_CUR if active else ("#f6f6fb" if is_app else BAND_FILL)),zorder=1))
        midb=(b_top+b_bot)/2
        ax.text(LABELX, midb+0.12, name, ha="right", va="center", fontsize=11,
                fontweight="bold", color=(GOLD if active else (APP_EDGE if is_app else TIER_TXT)))
        ax.text(LABELX, midb-0.24, desc, ha="right", va="center", fontsize=7.3,
                style="italic", color="#aeb8bf")
        ci=0
        for r,row in enumerate(rows):
            n=len(row); cw=(W-(n-1)*GAP)/n
            ry=b_top - PADV - r*(CH+INNERGAP) - CH
            for j,(title,tag) in enumerate(row):
                draw_card(CX0+j*(cw+GAP), ry, cw, title, status_of(ti,ci,current), tag); ci+=1
        nxt_app=(ti+1<len(TIERS) and ti+1==APP_TI)
        cur_y=b_bot-(APP_GAP if nxt_app else TIERGAP)

    for ti in range(len(TIERS)-1):
        if ti+1==APP_TI:
            b_bot=band_box[ti][1]; app_top=band_box[ti+1][0]; dvy=(b_bot+app_top)/2
            ax.plot([CX0-0.30,CX1+0.30],[dvy,dvy],ls=(0,(2,3)),color="#cfd6db",lw=1.1,zorder=1)
            ax.text(CX0-0.30, dvy+0.16, "SUPPORTING MATERIAL — APPENDICES  (Pointed to in the Overview)",
                    ha="left",va="bottom",fontsize=8,fontweight="bold",color=APP_TXT)
            # down-arrow routed to the right of the label text so no letters overlap it
            xA=CX1-1.9
            ax.add_patch(FancyArrowPatch((xA,b_bot),(xA,app_top),arrowstyle="-|>",
                         mutation_scale=12,color=APP_EDGE,lw=1.5,linestyle=(0,(3,2)),zorder=2))
        else:
            ax.add_patch(FancyArrowPatch((MIDX,band_box[ti][1]-0.04),(MIDX,band_box[ti+1][0]+0.04),
                         arrowstyle="-|>",mutation_scale=16,color="#b7c1c8",lw=2.4,zorder=2))

    top=band_box[0][0]; HEADX=CX0-0.30
    if current[0]!="overview":
        ly=top+1.18; bw_,bh_=0.58,0.40; pitch=2.05
        items=[("done","Done"),("cur","Current"),("fut","Ahead")]
        for k,(status,lab) in enumerate(items):
            bx=(CX1-(len(items)*pitch-(pitch-bw_-0.78)))+k*pitch
            fc,ec,tc,lwd,ls=st(status)
            ax.add_patch(FancyBboxPatch((bx,ly-bh_/2),bw_,bh_,boxstyle="round,pad=0.01,rounding_size=0.07",
                         linewidth=min(lwd,2.2),edgecolor=ec,facecolor=fc,linestyle=ls,zorder=6))
            badge(ax,bx+0.05,ly+bh_/2-0.05,status,s=95)
            ax.text(bx+bw_+0.10,ly,lab,ha="left",va="center",fontsize=8.0,color="#5a6873")
    # (narrative "Now: ..." line intentionally omitted -- it lives in the LaTeX caption)

    x0,x1=-0.1,CX1+0.4; y0,y1=band_box[-1][1]-0.45,top+1.5
    ax.set_xlim(x0,x1); ax.set_ylim(y0,y1); ax.axis("off")
    SCALE=0.56; fig.set_size_inches((x1-x0)*SCALE,(y1-y0)*SCALE)
    figs_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"figures")
    out=os.path.join(figs_dir, f"structure_map_{slug}.pdf")          # vector PDF for the paper
    fig.savefig(out)
    fig.savefig(os.path.join(figs_dir, f"structure_map_{slug}.png")) # PNG for quick preview
    plt.close(fig)
    return out

# ---------- the stages (one figure now: the overview) ----------
STAGES = [
 dict(slug="00_overview", current=("overview",)),
]

if __name__=="__main__":
    want=set(sys.argv[1:])
    for s in STAGES:
        if want and s["slug"] not in want: continue
        print("wrote", make_map(s["current"], s["slug"]))
