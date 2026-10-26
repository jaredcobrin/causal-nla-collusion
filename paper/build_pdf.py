"""draft_v2.md -> figures inserted -> HTML in the Apart template layout -> PDF (Chrome headless). python build_pdf.py"""
import re, subprocess, html, os
src = open("draft_v4.md").read()
src = re.sub(r"<!--.*?-->", "", src, flags=re.S)
cap = lambda t: f"\n\n*{t}*\n\n"
import json
L = lambda p: [json.loads(l)["text"].replace("\n", " ") for l in open(p)]
I = json.load(open("../paper_evidence/data/short_info.json"))
sft = L("../paper_evidence/results/gen/ev_short_sft.jsonl"); a2 = L("../paper_evidence/results/gen/ev_short_A2s11_100.jsonl")
off = [t.replace("\n", " ") for t in json.load(open("../official_nla/outputs/av_outputs.json"))["short"]["text"]]
def cut(s, n): s = s.replace("|", "/"); return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " ..."
rows = ["| Input sentence | SFT | RL + collusion data (seed 11) | Official NLA (excerpt) |", "|---|---|---|---|"]
for idx, kind in ((0, "collusive"), (20, "honest twin")):
    rows.append(f"| *{kind}:* \"{I['texts'][idx]}\" | {cut(sft[idx], 80)} | {cut(a2[idx], 110)} | {cut(off[idx], 120)} |")
tbl = "\n".join(rows) + "\n\n*Table 1. What the models write for the same activations (excerpts). Our verbalizers write one short sentence; the official NLA writes a longer explanation.*"
src = src.replace("{{EXAMPLES_TABLE}}", tbl)
_ct = "../paper_evidence/results/core_transfer_report.md"
if os.path.exists(_ct):
    src = src.replace("{{CORE_TABLE}}", open(_ct).read() + "\n\n*Table 6. Final models on the Rose-style data: share of agent messages flagged, and balanced accuracy of the flag against the colluder/honest role (95% CI by scenario).*")
else:
    src = src.replace("{{CORE_TABLE}}", "")
src = src.replace("{{CORE_RESULT}}", open("core_sentence.txt").read() if os.path.exists("core_sentence.txt") else "see Table 7.")
CAP = {
 "fig1_original_nla": "**Figure 1.** The original natural language autoencoder (Anthropic, 2026). An activation from a target model is turned into a natural-language explanation by the verbalizer (AV); the reconstructor (AR) rebuilds the activation from that text alone. The reward for the AV is how close the rebuilt activation is to the original. Both are first fine-tuned on a summarisation task (summaries of the text written by Claude); RL then trains the AV, with a KL penalty.",
 "fig3_side_by_side": "**Figure 2.** How the original NLA and this work are trained. This work replaces the summary warm start with probe directions and adds reward terms that point back at the activation or the directions. No collusion label appears in the reward.",
 "fig4_hand50": "**Figure 3.** The 50 simple contrast examples: how many of the 20 collusive sentences and the 20 honest twins each model flags. Dots are individual seeds.",
 "fig6_heldout": "**Figure 4.** Held-out concepts the verbalizer was never taught to name: positions where the output names the concept, SFT vs B (in brackets: positions picked for each concept).",
 "fig5_locked": "**Figure A1.** The larger controlled test (Appendix A). (a) Balanced accuracy on 359 cue-free pairs; dots are seeds, dashed line is chance. (b) Share flagged for collusive and honest sentences with and without cue words.",
}
for k, v in CAP.items(): src = src.replace(f"![](figures/{k}.png)", f"![{v}](figures/{k}.png)")
# split title / authors / abstract / body
m = re.search(r"^# (.+)$", src, re.M); title = m.group(1)
ab = re.search(r"## Abstract\s+(.*?)\n## 1\.", src, re.S).group(1).strip()
body = src[src.index("## 1. Introduction"):]
open("_body.md", "w").write(body)
bodyhtml = subprocess.run(["pandoc", "-f", "gfm", "-t", "html5", "_body.md"], capture_output=True, text=True, check=True).stdout
abhtml = subprocess.run(["pandoc", "-f", "gfm", "-t", "html5"], input=ab, capture_output=True, text=True, check=True).stdout
def figs(h):
    def f(m):
        cap = " ".join(html.unescape(m.group(2)).split()); cap = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(cap, quote=False)); return f'<figure><img src="{m.group(1)}"><figcaption>{cap}</figcaption></figure>'
    return re.sub(r'<img src="([^"]+)"\s+alt="([^"]*)"\s*/?>', f, h, flags=re.S)
mark = lambda s: re.sub(r"\[((?:Jared:|to fill|LessWrong,)[^\]]*)\]", r'<mark>[\1]</mark>', s)
bodyhtml, abhtml = mark(figs(bodyhtml)), mark(abhtml)
css = """
@page { size: Letter; margin: 0.75in 0.95in 0.8in 0.95in; @bottom-center { content: counter(page); font-size: 9pt; color:#555; } }
body { font-family: 'Old Standard TT','Times New Roman',Times,serif; font-size: 10pt; line-height: 1.22; color:#000; }
h1.title { font-size: 19pt; text-align:center; margin: 0 0 8pt 0; line-height:1.15; border-top: 3pt solid #000; padding-top: 10pt; }
.authors { text-align:center; margin: 6pt 0 2pt 0; font-size: 10.5pt; } .with { text-align:center; margin: 6pt 0; font-size: 10.5pt; }
.rule { border-top: 1pt solid #000; border-bottom: 1pt solid #000; height: 3pt; margin: 8pt 0 10pt 0; }
h3.abs { text-align:center; font-size: 13pt; margin: 6pt 0 6pt 0; }
.abstract { margin: 0 24pt 10pt 24pt; text-align: justify; font-size: 10pt; line-height:1.24; }
h2 { font-size: 13.5pt; margin: 13pt 0 4pt 0; page-break-after: avoid; } h3 { font-size: 11pt; margin: 9pt 0 3pt 0; }
p { margin: 0 0 5pt 0; text-align: justify; } li { margin-bottom: 2pt; text-align: justify; }
table { border-collapse: collapse; margin: 4pt auto 5pt auto; font-size: 8.4pt; line-height:1.18; page-break-inside: avoid; }
th, td { border-top: 0.6pt solid #888; border-bottom: 0.6pt solid #888; padding: 2pt 6pt; text-align: left; } th { background:#f1f1f1; border-top: 1.2pt solid #000; }
figure { margin: 6pt 0; page-break-inside: avoid; } figcaption { font-size: 8.6pt; font-style: normal; text-align: justify; margin: 2pt 18pt 0 18pt; line-height:1.2; } img { display:block; margin: 4pt auto; max-width: 100%; } figure:has(img[src*='fig1']) img { width: 78%; } figure:has(img[src*='fig3']) img { width: 66%; } figure:has(img[src*='fig4']) img, figure:has(img[src*='fig6']) img { width: 54%; } figure:has(img[src*='fig5']) img { width: 66%; }
mark { background: #fff0a0; } .fn { font-size: 8.5pt; color:#333; border-top: 0.5pt solid #999; margin-top: 8pt; padding-top: 3pt; }
code { font-size: 8.6pt; }
"""
authors = '<div class="authors"><b>Jared Cobrin</b><br>Independent</div><div class="with"><b>With</b> Apart Research</div>'
doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>
<h1 class="title">{html.escape(title)}<sup>1</sup></h1>{authors}<div class="fn" style="text-align:center;border:0;margin:2pt 0 0 0"><sup>1</sup> Research conducted at the AI Collusion Research Sprint, October 2026</div><div class="rule"></div>
<h3 class="abs">Abstract</h3><div class="abstract">{abhtml}</div>
{bodyhtml}
</body></html>"""
open("paper.html", "w").write(doc)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={os.getcwd()}/paper_draft.pdf", f"file://{os.getcwd()}/paper.html"], capture_output=True, timeout=120)
pdf = open("paper_draft.pdf", "rb").read(); print("pages:", len(re.findall(rb"/Type\s*/Page[^s]", pdf)), "size KB:", len(pdf) // 1024)
