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

# Unicode the draft uses -> LaTeX. Order matters within this list: the longer
# "θ₃dB" must be replaced before the bare "θ", or the subscript would be orphaned.
UNICODE = [
    ("≈", r"$\approx$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("×", r"$\times$"), ("±", r"$\pm$"), ("→", r"$\rightarrow$"),
    ("−", r"$-$"), ("Σ", r"$\Sigma$"), ("θ₃dB", r"$\theta_{3\mathrm{dB}}$"), ("θ", r"$\theta$"), ("σ", r"$\sigma$"), ("α", r"$\alpha$"),
    ("μ", r"$\mu$"), ("λ", r"$\lambda$"), ("τ", r"$\tau$"), ("ε", r"$\varepsilon$"), ("∈", r"$\in$"), ("²", r"$^2$"), ("³", r"$^3$"),
    ("⁴", r"$^4$"), ("ᵢ", r"$_i$"), ("₀", r"$_0$"), ("°", r"$^\circ$"), ("…", r"\ldots{}"), ("§", r"\S{}"), ("‑", "-"), ("–", "--"),
    ("—", "---"), ("“", "``"), ("”", "''"), ("‘", "`"), ("’", "'"), ("\u00a0", "~"), ("\u202f", "~"), ("·", r"$\cdot$"), ("∞", r"$\infty$"),
    ("′", r"$'$"),
]
# LaTeX's own special characters. The backslash MUST be first: escaping it after
# the others would re-escape the backslashes those replacements just introduced.
SPECIALS = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def inline(text):
    """Escape LaTeX specials, map Unicode, then apply the inline markup."""
    # protect code spans first
    # Code spans are lifted out and replaced by \x00N\x00 placeholders, so the
    # escaping below cannot touch their contents. \x00 is safe because it cannot
    # appear in the source Markdown.
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
    # Italic is deliberately fussy: the lookarounds require the asterisks to sit
    # outside word characters and to hug non-space text, so a literal * in prose
    # and the ** already consumed above are both left alone.
    text = re.sub(r"(?<![\w\\])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\\emph{\1}", text)
    # URLs had their _ and % escaped by SPECIALS; \href needs them raw again.
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", lambda m: r"\href{%s}{%s}" % (m.group(2).replace("\\_", "_").replace("\\%", "%"), m.group(1)), text)
    def restore(m):
        # Code spans are escaped now -- after the markup passes -- so their
        # contents are typeset verbatim rather than interpreted.
        c = codes[int(m.group(1))]
        for a, b in SPECIALS:
            c = c.replace(a, b)
        return r"\texttt{%s}" % c
    text = re.sub("\x00(\\d+)\x00", restore, text)
    return text


def table(caption, header, rows):
    """table* spanning both columns; text-heavy columns wrap (tabularx X), short numeric ones stay natural width."""
    ncol = len(header)
    caption = re.sub(r"^Table \d+\.\s*", "", caption)      # IEEEtran numbers the table itself
    # Column width heuristic: measured in characters over the header and every row.
    widest = [max([len(header[k])] + [len(r[k]) if k < len(r) else 0 for r in rows]) for k in range(ncol)]
    # Wider than 16 characters -> a wrapping X column; otherwise centred fixed width.
    spec = "".join(r">{\raggedright\arraybackslash}X" if w > 16 else "c" for w in widest)
    # With no X column tabularx has nothing to stretch, so fall back to a plain
    # tabular with a left-aligned first column.
    spec = spec if "X" in spec else "l" + "c" * (ncol - 1)
    # Shrink the type for tables that are wide in either sense.
    size = r"\scriptsize" if ncol >= 6 or sum(widest) > 150 else r"\footnotesize"
    env = "tabularx" if "X" in spec else "tabular"
    begin = r"\begin{tabularx}{\textwidth}{%s}" % spec if env == "tabularx" else r"\begin{tabular}{%s}" % spec
    out = [r"\begin{table*}[!t]", r"\caption{%s}" % inline(caption), r"\centering" + size, begin, r"\toprule",
           " & ".join(inline(c) for c in header) + r" \\", r"\midrule"]
    for r in rows:
        # Short rows are padded and long ones truncated, so a malformed table in
        # the Markdown still produces valid LaTeX rather than a compile error.
        r = r + [""] * (ncol - len(r))
        out.append(" & ".join(inline(c) for c in r[:ncol]) + r" \\")
    out += [r"\bottomrule", r"\end{%s}" % env, r"\end{table*}", ""]
    return out


def cells(line):
    """Split one Markdown pipe-table row into stripped cell strings."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md):
    """Walk the Markdown line by line and emit the LaTeX body.

    Returns (title, note, body). A hand-rolled state machine rather than a
    Markdown library: `para`, `in_list` and `pending_caption` hold the state
    between lines, and flush_para / close_list are what terminate a block.
    """
    lines = md.splitlines()
    out = [r"% generated from docs/paper1_draft.md by docs/tools/md_to_ieeetran.py; edit the Markdown or this file, not both"]
    i = 0; title = None; note = None; pending_caption = None; in_list = None; para = []

    def flush_para():
        """Emit the buffered paragraph. Lines are joined with spaces, so a
        Markdown soft wrap does not become a LaTeX line break."""
        nonlocal para
        if para:
            out.append(inline(" ".join(para))); out.append(""); para = []

    def close_list():
        """Close whichever list environment is open, if any."""
        nonlocal in_list
        if in_list:
            out.append(r"\end{%s}" % in_list); out.append(""); in_list = None

    while i < len(lines):
        ln = lines[i]
        # The first "# " line is the paper title, not a section.
        if ln.startswith("# ") and title is None:
            title = ln[2:].strip(); i += 1; continue
        # The "*Draft ...*" line becomes the author \thanks note.
        if ln.startswith("*Draft") and note is None:
            note = ln.strip("*").strip(); i += 1; continue
        if ln.startswith("## "):
            # Strip any manual "3. " numbering: LaTeX numbers sections itself.
            flush_para(); close_list(); h = ln[3:].strip(); h = re.sub(r"^\d+\.\s*", "", h)
            if h.lower().startswith("abstract"):
                out.append(r"\begin{abstract}"); abstract_open = True
                # the abstract is the paragraph(s) until the next ## ; collect them
                # Consumed here with its own scan, then `i` jumps past it, because
                # the abstract environment does not tolerate blank-line paragraph
                # breaks the way body text does.
                j = i + 1; paras = []
                while j < len(lines) and not lines[j].startswith("## "):
                    if lines[j].strip():
                        paras.append(lines[j].strip())
                    j += 1
                out.append(inline(" ".join(paras))); out.append(r"\end{abstract}"); out.append("")
                # Keywords are hard-coded here rather than read from the Markdown.
                out.append(r"\begin{IEEEkeywords}"); out.append("Radio radiance fields, 3D Gaussian splatting, ray tracing, channel modelling, reproducibility."); out.append(r"\end{IEEEkeywords}"); out.append("")
                i = j; continue
            if h.lower().startswith("tables and figures"):
                # This section is a working note, not part of the paper: the whole
                # block is emitted as LaTeX comments so it survives but never prints.
                out.append("% " + h); j = i + 1
                while j < len(lines) and not lines[j].startswith("## "):
                    out.append("% " + lines[j]); j += 1
                i = j; continue
            out.append(r"\section{%s}" % inline(h)); out.append(""); i += 1; continue
        if ln.startswith("### "):
            # Same numbering strip as above, for the "3.1 " subsection form.
            flush_para(); close_list(); h = re.sub(r"^\d+\.\d+\s*", "", ln[4:].strip())
            out.append(r"\subsection{%s}" % inline(h)); out.append(""); i += 1; continue
        # A bold "Table N. caption" line is held until the table that follows it.
        m = re.match(r"^\*\*(Table \d+\.\s*.+)\*\*\s*$", ln)
        if m:
            flush_para(); close_list(); pending_caption = m.group(1); i += 1; continue
        if ln.startswith("|"):
            flush_para(); close_list()
            header = cells(ln); i += 1
            # Skip the |---|---| separator row if present.
            if i < len(lines) and re.match(r"^\|\s*-", lines[i]):
                i += 1
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i])); i += 1
            out += table(pending_caption or "", header, rows); pending_caption = None; continue
        # List item: "- ", "* " or "1. ". The **-guard stops a bold line that
        # happens to start with * from being read as a bullet.
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m and not ln.startswith("**"):
            flush_para(); kind = "enumerate" if m.group(2)[0].isdigit() else "itemize"
            # Switching bullet kind closes the old environment and opens a new one.
            # Note indentation (group 1) is captured but ignored: nested lists are
            # flattened into the enclosing one.
            if in_list != kind:
                close_list(); out.append(r"\begin{%s}" % kind); in_list = kind
            out.append(r"\item " + inline(m.group(3))); i += 1; continue
        # A blank line terminates whatever block was open.
        if not ln.strip():
            flush_para(); close_list(); i += 1; continue
        para.append(ln.strip()); i += 1
    # Anything still buffered when the input ends.
    flush_para(); close_list()
    return title, note, "\n".join(out)


# Fixed preamble. \MISSING is the draft's marker for a result that has not been
# measured yet; it prints in red so an unfilled slot cannot reach a reader unnoticed.
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
    """Read the Markdown named on argv[1] and print a complete .tex to stdout."""
    md = open(sys.argv[1], encoding="utf-8").read()
    title, note, body = convert(md)
    print(PREAMBLE)
    print(r"\begin{document}")
    print(r"\title{%s}" % inline(title or "Untitled"))
    # The trailing % suppresses the newline LaTeX would otherwise insert here.
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
