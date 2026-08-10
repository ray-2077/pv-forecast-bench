"""One-off conversion script for the Overleaf bundle (paper/overleaf/).
Reads paper/draft/0N_*.md, applies ONLY the conversion rules given for
this task (headers, *italic*, and %/&/_/# escaping), and writes
paper/overleaf/sections/0N_*.tex. Does not touch numbered lists, does
not insert floats (done separately by hand for placement judgement),
does not reword/reformat/improve any prose. Not part of the normal
build pipeline - written for this one bundle-assembly task.

Usage:
    python scripts/convert_sections_to_latex.py
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRAFT_DIR = REPO_ROOT / "paper" / "draft"
OUT_DIR = REPO_ROOT / "paper" / "overleaf" / "sections"

# file stem -> (title-cased \section title)
SECTION_TITLES = {
    "01_introduction": "Introduction",
    "02_related_work": "Related Work",
    "03_data": "Data and Preprocessing",
    "04_methodology": "Methodology",
    "05_experimental_setup": "Experimental Setup",
    "06_results": "Results",
    "07_limitations": "Limitations and Threats to Validity",
    "08_conclusion": "Conclusion",
}

TOP_HEADING_RE = re.compile(r"^#?\s*[IVX]+\.\s+[A-Z][A-Z \-]*$")
# (?:##)? / (?:###)? - the WHOLE marker is optional (0 or 2/3 '#'
# characters), not "1 or 2" - 01-04/06-08 prefix subsections with "##"
# and subsubsections with "###"; 05_experimental_setup.md uses no
# markdown '#' marker at all for either. A bare "##?" (single mandatory
# '#') matched the former but not the latter - fixed here.
SUBSECTION_RE = re.compile(r"^(?:##)?\s*([A-Z])\.\s+(.+)$")
SUBSUBSECTION_RE = re.compile(r"^(?:###)?\s*(\d+)\)\s+(.+)$")


def escape_specials(text):
    # Order matters only in that we must not re-escape a backslash we
    # just inserted - none of the four target characters appear inside
    # any \textit{...} command or [CITE ...]/[VERIFY ...] bracket in
    # this draft (checked by hand before writing this script), so a
    # single global pass per character is safe.
    text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    text = text.replace("_", r"\_")
    text = text.replace("#", r"\#")
    return text


def convert_italics(text):
    # *phrase* -> \textit{phrase}, DOTALL so a span crossing a markdown
    # line-wrap (soft-wrapped prose, no blank line inside) still matches.
    return re.sub(r"\*([^*]+)\*", r"\\textit{\1}", text, flags=re.DOTALL)


def convert_file(stem):
    src = DRAFT_DIR / f"{stem}.md"
    text = src.read_text(encoding="utf-8")
    lines = text.split("\n")

    out_lines = []
    skipped_top_heading = False
    for line in lines:
        stripped = line.strip()

        # Top-level heading: "# I. INTRODUCTION" (or, in 05, the same
        # text with no leading "# " at all) - replace with \section{...}
        # exactly once, using the hand-verified title mapping above
        # (not derived by regex, to avoid any title-casing mistake).
        if not skipped_top_heading and TOP_HEADING_RE.match(stripped):
            out_lines.append(f"\\section{{{SECTION_TITLES[stem]}}}")
            skipped_top_heading = True
            continue

        m = SUBSECTION_RE.match(stripped)
        if m:
            out_lines.append(f"\\subsection{{{m.group(2)}}}")
            continue

        m = SUBSUBSECTION_RE.match(stripped)
        if m:
            out_lines.append(f"\\subsubsection{{{m.group(2)}}}")
            continue

        out_lines.append(line)

    body = "\n".join(out_lines)
    body = convert_italics(body)
    body = escape_specials(body)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{stem}.tex"
    out_path.write_text(body, encoding="utf-8")
    print(f"wrote {out_path} ({len(body)} bytes, heading converted: {skipped_top_heading})")


def main():
    for stem in SECTION_TITLES:
        convert_file(stem)


if __name__ == "__main__":
    main()
