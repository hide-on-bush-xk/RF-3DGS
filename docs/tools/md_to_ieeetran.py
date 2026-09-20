"""Convert docs/paper1_draft.md into an IEEEtran LaTeX body (pdflatex).

Handles the subset of Markdown the draft uses: # title, ## / ### sections,
paragraphs, bullet and numbered lists, pipe tables preceded by a bold
"Table N. caption" line (rendered as table* with booktabs), **bold**, *italic*,
`code`, [text](url), and the Unicode symbols the draft uses (degrees, plus-minus,
arrows, Greek letters, sub/superscripts). Everything else passes through.

    python docs/tools/md_to_ieeetran.py docs/paper1_draft.md > body.tex
"""

from __future__ import annotations

import re
import sys

UNICODE = [
    ("≈", r"$\approx$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("×", r"$\times$"), ("±", r"$\pm$"), ("→", r"$\rightarrow$"),
    ("−", r"$-$"), ("Σ", r"$\Sigma$"), ("θ₃dB", r"$\theta_{3\mathrm{dB}}$"), ("θ", r"$\theta$"), ("σ", r"$\sigma$"), ("α", r"$\alpha$"),
    ("μ", r"$\mu$"), ("λ", r"$\lambda$"), ("τ", r"$\tau$"), ("ε", r"$\varepsilon$"), ("∈", r"$\in$"), ("²", r"$^2$"), ("³", r"$^3$"),
    ("⁴", r"$^4$"), ("ᵢ", r"$_i$"), ("₀", r"$_0$"), ("°", r"$^\circ$"), ("…", r"\ldots{}"), ("§", r"\S{}"), ("‑", "-"), ("–", "--"),
    ("—", "---"), ("“", "``"), ("”", "''"), ("‘", "`"), ("’", "'"), ("\u00a0", "~"), ("\u202f", "~"), ("·", r"$\cdot$"), ("∞", r"$\infty$"),
]
SPECIALS = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def inline(text):
    """Escape LaTeX specials, map Unicode, then apply the inline markup."""
    # protect code spans first
    codes = []
    def keep(m):
        codes.append(m.group(1)); return f"\x00{len(codes) - 1}\x00"
    text = re.sub(r"`([^`]+)`", keep, text)
    for a, b in SPECIALS:
        text = text.replace(a, b)
    for a, b in UNICODE:
        text = text.replace(a, b)
    # collapse adjacent math: "$\pm$0.5" fine; "$^\circ$ / " fine
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w\\])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\\emph{\1}", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", lambda m: r"\href{%s}{%s}" % (m.group(2).replace("\\_", "_").replace("\\%", "%"), m.group(1)), text)
    def restore(m):
        c = codes[int(m.group(1))]
        for a, b in SPECIALS:
            c = c.replace(a, b)
        return r"\texttt{%s}" % c
    text = re.sub("\x00(\\d+)\x00", restore, text)
    return text


def table(caption, header, rows):
    """table* spanning both columns; text-heavy columns wrap (tabularx X), short numeric ones stay natural width."""
    ncol = len(header)
    widest = [max([len(header[k])] + [len(r[k]) if k < len(r) else 0 for r in rows]) for k in range(ncol)]
    spec = "".join(r">{\raggedright\arraybackslash}X" if w > 16 else "c" for w in widest)
    spec = spec if "X" in spec else "l" + "c" * (ncol - 1)
    size = r"\scriptsize" if ncol >= 6 or sum(widest) > 150 else r"\footnotesize"
    env = "tabularx" if "X" in spec else "tabular"
    begin = r"\begin{tabularx}{\textwidth}{%s}" % spec if env == "tabularx" else r"\begin{tabular}{%s}" % spec
    out = [r"\begin{table*}[!t]", r"\caption{%s}" % inline(caption), r"\centering" + size, begin, r"\toprule",
           " & ".join(inline(c) for c in header) + r" \\", r"\midrule"]
    for r in rows:
        r = r + [""] * (ncol - len(r))
        out.append(" & ".join(inline(c) for c in r[:ncol]) + r" \\")
    out += [r"\bottomrule", r"\end{%s}" % env, r"\end{table*}", ""]
    return out


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md):
    lines = md.splitlines()
    out = [r"% generated from docs/paper1_draft.md by docs/tools/md_to_ieeetran.py; edit the Markdown or this file, not both"]
    i = 0; title = None; note = None; pending_caption = None; in_list = None; para = []

    def flush_para():
        nonlocal para
        if para:
            out.append(inline(" ".join(para))); out.append(""); para = []

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{%s}" % in_list); out.append(""); in_list = None

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# ") and title is None:
            title = ln[2:].strip(); i += 1; continue
        if ln.startswith("*Draft") and note is None:
            note = ln.strip("*").strip(); i += 1; continue
        if ln.startswith("## "):
            flush_para(); close_list(); h = ln[3:].strip(); h = re.sub(r"^\d+\.\s*", "", h)
            if h.lower().startswith("abstract"):
                out.append(r"\begin{abstract}"); abstract_open = True
                # the abstract is the paragraph(s) until the next ## ; collect them
                j = i + 1; paras = []
                while j < len(lines) and not lines[j].startswith("## "):
                    if lines[j].strip():
                        paras.append(lines[j].strip())
                    j += 1
                out.append(inline(" ".join(paras))); out.append(r"\end{abstract}"); out.append("")
                out.append(r"\begin{IEEEkeywords}"); out.append("Radio radiance fields, 3D Gaussian splatting, ray tracing, channel modelling, reproducibility."); out.append(r"\end{IEEEkeywords}"); out.append("")
                i = j; continue
            if h.lower().startswith("tables and figures"):
                out.append("% " + h); j = i + 1
                while j < len(lines) and not lines[j].startswith("## "):
                    out.append("% " + lines[j]); j += 1
                i = j; continue
            out.append(r"\section{%s}" % inline(h)); out.append(""); i += 1; continue
        if ln.startswith("### "):
            flush_para(); close_list(); h = re.sub(r"^\d+\.\d+\s*", "", ln[4:].strip())
            out.append(r"\subsection{%s}" % inline(h)); out.append(""); i += 1; continue
        m = re.match(r"^\*\*(Table \d+\.\s*.+)\*\*\s*$", ln)
        if m:
            flush_para(); close_list(); pending_caption = m.group(1); i += 1; continue
        if ln.startswith("|"):
            flush_para(); close_list()
            header = cells(ln); i += 1
            if i < len(lines) and re.match(r"^\|\s*-", lines[i]):
                i += 1
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i])); i += 1
            out += table(pending_caption or "", header, rows); pending_caption = None; continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m and not ln.startswith("**"):
            flush_para(); kind = "enumerate" if m.group(2)[0].isdigit() else "itemize"
            if in_list != kind:
                close_list(); out.append(r"\begin{%s}" % kind); in_list = kind
            out.append(r"\item " + inline(m.group(3))); i += 1; continue
        if not ln.strip():
            flush_para(); close_list(); i += 1; continue
        para.append(ln.strip()); i += 1
    flush_para(); close_list()
    return title, note, "\n".join(out)


PREAMBLE = r"""\documentclass[journal,twocolumn]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\newcommand{\MISSING}[1]{\textcolor{red}{[MISSING: #1]}}
"""


def main():
    md = open(sys.argv[1], encoding="utf-8").read()
    title, note, body = convert(md)
    print(PREAMBLE)
    print(r"\begin{document}")
    print(r"\title{%s}" % inline(title or "Untitled"))
    print(r"\author{Ke~Xu%")
    print(r"\thanks{Draft generated from the working repository; " + inline(note or "") + "}}")
    print(r"\maketitle")
    print(body)
    print(r"\nocite{*}   % the draft names its references in prose; list every entry of refs.bib until \cite keys are placed")
    print(r"\bibliographystyle{IEEEtran}")
    print(r"\bibliography{refs}")
    print(r"\end{document}")


if __name__ == "__main__":
    main()
