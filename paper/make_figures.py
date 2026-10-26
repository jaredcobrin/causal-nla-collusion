"""All figures for the paper. Numbers copied from paper_evidence/RESULTS.md and results/*.md; example sentences read from saved outputs."""
import json, textwrap, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R = "../paper_evidence/results"; OUT = "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, GREY, GREEN, ORANGE = "#c0392b", "#2c6fbb", "#8a8a8a", "#2e8b57", "#e08e0b"

def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#2c6fbb", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)
def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10, lw=1.2, color=color))

# ---------- Figure 1: original NLA ----------
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.4, 2.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
def nbox(x, w, text, fc, ec, bold=False):
    ax.add_patch(FancyBboxPatch((x, 1.85), w, 1.25, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, 2.475, text, ha="center", va="center", fontsize=8.1, fontweight="bold" if bold else "normal", linespacing=1.3)
nbox(0.05, 1.7, "Activation  h\nfrom the\ntarget model", "#fff4e0", ORANGE)
nbox(2.25, 1.6, "Verbalizer\n(AV)", "#eef3fb", BLUE, True)
nbox(4.35, 2.1, "Explanation\nin natural\nlanguage", "#f6f6f6", GREY)
nbox(6.8, 1.7, "Reconstructor\n(AR)", "#eef3fb", BLUE, True)
nbox(8.8, 1.15, "Rebuilt\nactivation \u0125", "#fff4e0", ORANGE)
for x1, x2 in ((1.75, 2.25), (3.85, 4.35), (6.45, 6.8), (8.5, 8.8)): arrow(ax, x1, 2.475, x2, 2.475)
ax.add_patch(FancyBboxPatch((0.05, 0.25), 9.9, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#e8f5ee", ec=GREEN, lw=1.5))
ax.text(5.0, 0.65, "Reward for the verbalizer:  how close is the rebuilt activation \u0125 to the original h ?", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#1d6b44")
arrow(ax, 0.9, 1.85, 0.9, 1.07, GREEN); arrow(ax, 9.375, 1.85, 9.375, 1.07, GREEN)
plt.savefig(f"{OUT}/fig1_original_nla.png", dpi=220, bbox_inches="tight"); plt.close()
plt.rcParams.update({"font.family": "serif"})

# ---------- Figure 3: side by side ----------
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.4, 4.3)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
def sbox(x, y, w, h, text, fc, ec, fs=8.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)
ax.text(0.05, 5.95, "Original NLA", fontsize=10, fontweight="bold")
sbox(0.05, 4.55, 3.0, 1.0, "Warm start:\nsummaries by Claude", "#f6f6f6", GREY); sbox(3.5, 4.55, 3.0, 1.0, "Joint RL", "#eef3fb", BLUE, bold=True); sbox(6.95, 4.55, 3.0, 1.0, "Reward:\nreconstruction", "#e8f5ee", GREEN, bold=True)
arrow(ax, 3.05, 5.05, 3.5, 5.05); arrow(ax, 6.5, 5.05, 6.95, 5.05)
ax.plot([0.05, 9.95], [4.2, 4.2], color="#bbb", lw=0.9, ls="--")
ax.text(0.05, 3.85, "This work", fontsize=10, fontweight="bold")
sbox(0.05, 2.45, 3.0, 1.0, "1. Find probe\ndirections", "#fff4e0", ORANGE); sbox(3.5, 2.45, 3.0, 1.0, "2. Train on direction\ndescriptions (SFT)", "#eef3fb", BLUE); sbox(6.95, 2.45, 3.0, 1.0, "3. RL on unlabeled\nactivations", "#eef3fb", BLUE, bold=True)
arrow(ax, 3.05, 2.95, 3.5, 2.95); arrow(ax, 6.5, 2.95, 6.95, 2.95); arrow(ax, 8.45, 2.45, 8.45, 1.75, GREEN)
ax.text(0.05, 1.95, "Reward terms (weights)", fontsize=8.6, color="#1d6b44", fontweight="bold", va="center")
for i, (t, w) in enumerate((("reconstruction", "0.40"), ("consistency", "0.25"), ("behaviour", "0.15"), ("concept\nagreement", "0.20"))):
    sbox(0.05 + i * 2.5, 0.2, 2.35, 1.2, f"{t}\n{w}", "#e8f5ee", GREEN, fs=8.4)
plt.savefig(f"{OUT}/fig3_side_by_side.png", dpi=220, bbox_inches="tight"); plt.close()
plt.rcParams.update({"font.family": "serif"})

# ---------- Figure 2: example outputs (real saved outputs) ----------
L = lambda p: [json.loads(l)["text"].replace("\n", " ") for l in open(p)]
I = json.load(open("../paper_evidence/data/short_info.json")); 
sft = L(f"{R}/gen/ev_short_sft.jsonl"); a2 = L(f"{R}/gen/ev_short_A2s11_100.jsonl")
off = [t.replace("\n", " ") for t in json.load(open("../official_nla/outputs/av_outputs.json"))["short"]["text"]]
def trunc(s, n): return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " ..."
ex = [(0, "collusive"), (20, "honest twin"), (3, "collusive"), (23, "honest twin")]
fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0.0, 9.7, "Figure 2. What the verbalizers say about the same activations", fontsize=9.5, fontweight="bold")
y = 9.1
for idx, kind in ex:
    ax.text(0.0, y, f'[{kind}]  "{I["texts"][idx]}"', fontsize=8.3, fontweight="bold", color=RED if kind == "collusive" else BLUE); y -= 0.42
    for name, S, col in (("SFT", sft, "#555"), ("RL + collusion data", a2, "#222"), ("Official NLA", off, GREEN)):
        s = trunc(S[idx], 190 if name == "Official NLA" else 150); lines = textwrap.wrap(s, 88)[:2]
        ax.text(0.15, y, name + ":", fontsize=7.4, color=col, fontweight="bold")
        for k, ln in enumerate(lines): ax.text(2.9, y - 0.3 * k, ln, fontsize=7.4, color=col)
        y -= 0.3 * len(lines) + 0.18
    y -= 0.2
plt.savefig(f"{OUT}/fig2_examples.png", dpi=220, bbox_inches="tight"); plt.close()

# ---------- Figure 4: hand-written 50 ----------
models = ["SFT", "RL, public\ntext (B)", "RL, public\ntext (re-run)", "RL + honest\nconv. only", "RL +\ncollusion\ndata"]
coll = [[2], [2], [9], [12, 5, 2], [13, 15, 15]]; hon = [[2], [0], [1], [4, 2, 3], [3, 2, 4]]
fig, ax = plt.subplots(figsize=(6.4, 3.0)); w = 0.36
for i in range(5):
    mc, mh = sum(coll[i]) / len(coll[i]), sum(hon[i]) / len(hon[i])
    ax.bar(i - w / 2, mc, w, color=RED, label="collusive sentences flagged" if i == 0 else None); ax.bar(i + w / 2, mh, w, color=BLUE, label="honest twins flagged" if i == 0 else None)
    if len(coll[i]) > 1:
        ax.scatter([i - w / 2] * 3, coll[i], color="white", edgecolor="k", s=14, zorder=3); ax.scatter([i + w / 2] * 3, hon[i], color="white", edgecolor="k", s=14, zorder=3)
ax.set_xticks(range(5)); ax.set_xticklabels(models, fontsize=7.5); ax.set_ylabel("flagged, out of 20"); ax.set_ylim(0, 20); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
plt.savefig(f"{OUT}/fig4_hand50.png", dpi=220, bbox_inches="tight"); plt.close()

# ---------- Figure 5: locked set ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
names = ["SFT", "RL, public\ntext (B)", "RL, public\ntext\n(re-run)", "RL + honest\nconv. only", "RL +\ncollusion\ndata"]; ba = [0.515, 0.506, 0.519, 0.553, 0.589]; lo = [0.490, 0.492, 0.497, 0.535, 0.566]; hi = [0.543, 0.518, 0.543, 0.572, 0.612]
cols = [GREY, GREY, GREY, BLUE, RED]
a.bar(range(5), ba, color=cols, yerr=[[b_ - l for b_, l in zip(ba, lo)], [h - b_ for b_, h in zip(ba, hi)]], capsize=3)
a.scatter([3] * 3, [0.553, 0.578, 0.528], color="white", edgecolor="k", s=14, zorder=3); a.scatter([4] * 3, [0.571, 0.582, 0.613], color="white", edgecolor="k", s=14, zorder=3)
a.axhline(0.5, color="k", lw=0.8, ls="--"); a.text(4.45, 0.503, "chance", fontsize=7, ha="right"); a.set_ylim(0.45, 0.65); a.set_xticks(range(5)); a.set_xticklabels(names, fontsize=6.8); a.set_ylabel("balanced accuracy")
a.set_title("(a) 359 cue-free pairs, 12 unseen types", fontsize=8.5, loc="left")
cells = ["collusive\nno cue", "honest\nno cue", "collusive\nwith cue", "honest\nwith cue"]
data = {"SFT": [21, 18, 29, 22], "RL + honest conv. only": [20, 9, 27, 15], "RL + collusion data": [38, 21, 56, 32]}
for k, (nm, v) in enumerate(data.items()):
    b.bar([i + (k - 1) * 0.27 for i in range(4)], v, 0.27, label=nm, color=[GREY, BLUE, RED][k])
b.set_xticks(range(4)); b.set_xticklabels(cells, fontsize=6.8); b.set_ylabel("% flagged"); b.legend(frameon=False, fontsize=6.8)
b.set_title("(b) intent vs cue words", fontsize=8.5, loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_locked.png", dpi=220, bbox_inches="tight"); plt.close()

# ---------- Figure 6: held-out concepts (blind-judged, genuine cases) ----------
cn = ["music\n(49)", "law\n(37)", "vehicles\n(54)", "German\n(30)", "mental\nhealth (44)", "hedging\n(38)"]; bb = [19 / 49, 19 / 37, 14 / 54, 10 / 30, 7 / 44, 0]
fig, ax = plt.subplots(figsize=(6.0, 2.8))
ax.bar(range(6), [100 * v for v in bb], 0.6, color=RED, label="after RL (run B)")
ax.axhline(5, color="#555", ls="--", lw=1.2); ax.text(5.45, 6.2, "before RL (SFT): about 5%", fontsize=7.5, ha="right", color="#444")
for i, v in enumerate(bb): ax.text(i, 100 * v + 1.2, f"{100 * v:.0f}%", ha="center", fontsize=7.5)
ax.set_xticks(range(6)); ax.set_xticklabels(cn, fontsize=7.5); ax.set_ylabel("% of genuine cases\nwhere the concept is named"); ax.set_ylim(0, 62); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
plt.savefig(f"{OUT}/fig6_heldout.png", dpi=220, bbox_inches="tight"); plt.close()
print("figures done:", sorted(os.listdir(OUT)))
