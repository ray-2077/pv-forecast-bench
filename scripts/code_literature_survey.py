"""Interactive evidence extractor for the PV forecasting literature survey.

PURPOSE
-------
Feeds results/literature_survey.csv, which feeds the paper's Related Work
section and survey table. Part of the paper's claim is about what this
literature FAILS TO REPORT (night-hour handling, proper baselines, skill
score, variance across seeds, code availability). That means the tool has
to preserve "the paper never said" as a distinct, honest signal, not paper
over it with a guess.

WHY THIS IS AN EVIDENCE EXTRACTOR, NOT A CLASSIFIER
----------------------------------------------------
For every judgement column (night_hours_excluded, baseline_used,
skill_score_reported, weather_source, split_type, variance_reported,
code_available) this script never decides the value. It only:
  1. searches the extracted text for trigger terms specific to that column
  2. prints every match with surrounding context and a page number
  3. prints "NO MATCHES FOUND" explicitly when there are none
  4. prompts the user for the value, showing the allowed options
  5. defaults to not_stated on Enter (where not_stated is a valid value
     for that column - see NOT_STATED DEFAULTS below)

"not_stated" is itself a finding: if a paper never says whether it
excluded night hours, that absence is data about the state of reporting
practice in this literature, not a gap to be filled in. A classifier that
infers "probably excluded night hours, it mentions daylight somewhere"
would quietly convert an absence-of-reporting result into a
presence-of-practice result, and the paper's central claim would be
built on fabricated codings. There is deliberately NO auto-classify mode,
not even behind an opt-in flag. If you want that, this is the wrong tool.

NOT_STATED DEFAULTS
--------------------
Five columns (night_hours_excluded, baseline_used, weather_source,
split_type, code_available) list not_stated as an explicit allowed value.
For those, pressing Enter with no input records not_stated.
Two columns (skill_score_reported, variance_reported) do NOT list
not_stated as allowed - the coder is reading the paper's own results
section, and "does this paper contain a skill-score number" is something
you can always determine, so Enter re-prompts instead of silently
defaulting for those two.

RESUMABILITY
------------
The source of truth for "already coded" is results/literature_survey.csv:
a paper is skipped only once its citation_key is a row in that file. But
citation_key is NOT the filename (publisher PDF filenames like
"1-s2.0-S1364032122006566-main.pdf" carry no citation information), so a
lookup from filename -> citation_key is needed to skip on re-run. That
mapping is reconstructed from evidence/<citation_key>.txt, which records
"SOURCE_FILE: <filename>" for the PDF it was built from. A file is only
treated as done if BOTH its evidence log exists AND that evidence log's
citation_key is present in the CSV - so a run interrupted mid-paper
(evidence log partially written, CSV row never appended) is correctly
retried from scratch rather than silently skipped.
evidence/ is gitignored (see .gitignore) but is load-bearing for this
skip logic on the machine doing the coding - don't delete it between
sessions if you want resumability. results/literature_survey.csv is the
tracked audit trail; evidence/ is the reproducibility trail behind it.

SCHEMA
------
The CSV header is read from disk and checked against EXPECTED_COLUMNS
below. This script never adds, renames, or reorders columns - if the
on-disk header does not match, it exits loudly rather than guessing.

USAGE
-----
    python scripts/code_literature_survey.py
    python scripts/code_literature_survey.py --dry-run   # rehearse, write nothing
    python scripts/code_literature_survey.py --limit 1    # stop after 1 new paper
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Paper titles/filenames routinely carry Unicode punctuation (non-breaking
# hyphens, ligatures, accented author names) that crashes print() under the
# legacy codepages some Windows terminals default to. Reconfigure early so a
# coding session never dies mid-paper on an unlucky character.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS_DIR = REPO_ROOT / "data" / "papers"
DEFAULT_CSV_PATH = REPO_ROOT / "results" / "literature_survey.csv"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "evidence"

MIN_EXTRACTED_CHARS = 1000
CONTEXT_CHARS = 150  # each side of a match -> ~300 characters total

EXPECTED_COLUMNS = [
    "citation_key",
    "year",
    "venue",
    "dataset",
    "night_hours_excluded",
    "baseline_used",
    "skill_score_reported",
    "weather_source",
    "split_type",
    "n_seeds",
    "variance_reported",
    "code_available",
    "key_claim",
    "notes",
    "evidence_level",
    "leakage_flag",
]

TRIGGER_TERMS = {
    "night_hours_excluded": [
        "night", "nighttime", "night-time", "daylight", "daytime", "zenith",
        "solar elevation", "sunrise", "sunset", "zero output", "non-zero",
        "clear-sky", "diurnal", "24-hour", "24 hour", "irradiance > 0",
        "GHI > 0",
    ],
    "baseline_used": [
        "persistence", "naive", "baseline", "benchmark", "reference forecast",
        "smart persistence", "climatology", "persistence model",
        "reference model",
    ],
    "skill_score_reported": [
        "skill score", "forecast skill", "improvement over",
        "relative to persistence", "SS", "nRMSE improvement",
        "improvement over persistence",
    ],
    "weather_source": [
        "measured", "observed", "NWP", "numerical weather", "forecast weather",
        "exogenous", "meteorological input", "ground station", "pyranometer",
        "satellite", "reanalysis", "ERA5", "MERRA", "GFS", "weather API",
    ],
    "split_type": [
        "train test split", "train-test split", "chronological", "random",
        "shuffle", "k-fold", "cross-validation", "hold-out", "holdout",
        "80/20", "70/30", "walk-forward", "expanding window",
        "leave-one-out", "stratified",
    ],
    "variance_reported": [
        "seed", "random seed", "repeated", "standard deviation", "average of",
        "runs", "+/-", "confidence interval", "mean +-", "std dev",
        "error bars", "multiple runs",
    ],
    "code_available": [
        "github", "code available", "publicly available", "reproducib",
        "data availability", "open source", "open-source", "repository",
        "zenodo", "supplementary material",
    ],
}

# Free-text hint terms for the "dataset" column. Not a restricted-vocabulary
# judgement column - just a nudge to help the human write the description.
DATASET_HINT_TERMS = [
    "dataset", "data set", "PV plant", "solar farm", "station", "site",
    "records", "samples", "period", "resolution", "sampling interval",
]

ALLOWED_VALUES = {
    "night_hours_excluded": ["yes", "no", "partial", "not_stated"],
    "baseline_used": [
        "persistence", "climatology", "convex", "own_components", "other_ML",
        "none", "not_stated",
    ],
    "skill_score_reported": ["yes", "no"],
    "weather_source": [
        "measured", "NWP_forecast", "reanalysis", "both", "none", "not_stated",
    ],
    "split_type": ["chronological", "random", "k-fold", "rolling", "not_stated"],
    "variance_reported": ["yes", "no"],
    "code_available": ["yes", "no", "not_stated"],
    "evidence_level": ["quoted", "summary_only"],
    "leakage_flag": ["none", "suspected", "documented"],
}

JUDGEMENT_COLUMNS = [
    "night_hours_excluded",
    "baseline_used",
    "skill_score_reported",
    "weather_source",
    "split_type",
    "variance_reported",
    "code_available",
]

DOI_RE = re.compile(r'\b10\.\d{4,9}/[^\s"<>]+')
ARXIV_RE = re.compile(r'arXiv:\s?(\d{4}\.\d{4,5})(v\d+)?', re.IGNORECASE)


# ---------------------------------------------------------------------------
# text extraction
# ---------------------------------------------------------------------------

def extract_text_pdf(path: Path):
    if pdfplumber is None:
        raise RuntimeError(
            "pdfplumber is not installed. Run: pip install pdfplumber"
        )
    full_text_parts = []
    page_offsets = []
    offset = 0
    with pdfplumber.open(path) as pdf:
        metadata = dict(pdf.metadata or {})
        for i, page in enumerate(pdf.pages, start=1):
            # default x_tolerance (~3) merges adjacent words into one blob on
            # some two-column / tightly-kerned academic layouts (verified on
            # this batch: energies-17-03877.pdf loses ALL word spacing at the
            # default setting). x_tolerance=1 fixed every file tested without
            # over-splitting normally-spaced PDFs.
            page_text = page.extract_text(x_tolerance=1) or ""
            page_offsets.append((offset, i))
            full_text_parts.append(page_text)
            offset += len(page_text) + 1  # +1 for the join newline below
    full_text = "\n".join(full_text_parts)
    return full_text, page_offsets, metadata


def extract_text_txt(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "\f" in raw:
        pages = raw.split("\f")
    else:
        pages = [raw]
    full_text_parts = []
    page_offsets = []
    offset = 0
    for i, page_text in enumerate(pages, start=1):
        page_offsets.append((offset, i))
        full_text_parts.append(page_text)
        offset += len(page_text) + 1
    full_text = "\n".join(full_text_parts)
    return full_text, page_offsets, {}


def load_paper_text(path: Path):
    if path.suffix.lower() == ".pdf":
        return extract_text_pdf(path)
    if path.suffix.lower() == ".txt":
        return extract_text_txt(path)
    raise ValueError(f"unsupported file type: {path.suffix}")


def page_for_offset(pos: int, page_offsets):
    page = page_offsets[0][1] if page_offsets else 1
    for start, pnum in page_offsets:
        if start <= pos:
            page = pnum
        else:
            break
    return page


# ---------------------------------------------------------------------------
# trigger search
# ---------------------------------------------------------------------------

def compile_trigger_pattern(term: str) -> re.Pattern:
    if term.isupper() and term.isalpha() and len(term) <= 4:
        # short all-caps abbreviations (SS, NWP, ...) - case sensitive and
        # word-bounded to cut down on false positives from ordinary words
        return re.compile(r"\b" + re.escape(term) + r"\b")
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    if term[0].isalnum() and term[-1].isalnum():
        pattern = r"\b" + escaped + r"\b"
    else:
        pattern = escaped
    return re.compile(pattern, re.IGNORECASE)


def find_matches(text: str, page_offsets, terms):
    hits = []
    for term in terms:
        pattern = compile_trigger_pattern(term)
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            ctx_start = max(0, start - CONTEXT_CHARS)
            ctx_end = min(len(text), end + CONTEXT_CHARS)
            snippet = re.sub(r"\s+", " ", text[ctx_start:ctx_end]).strip()
            hits.append({
                "term": term,
                "page": page_for_offset(start, page_offsets),
                "start": start,
                "snippet": snippet,
            })
    hits.sort(key=lambda h: h["start"])
    return hits


def print_matches(label: str, terms, hits):
    print(f"\n--- {label} ---")
    print(f"trigger terms: {', '.join(terms)}")
    if not hits:
        print("NO MATCHES FOUND")
        return
    for h in hits:
        print(f"  [p.{h['page']}] (matched '{h['term']}'): ...{h['snippet']}...")


# ---------------------------------------------------------------------------
# best-effort guesses for citation_key / year / venue - always overridable
# ---------------------------------------------------------------------------

def guess_year(filename_stem: str, text: str):
    m = re.search(r"(19|20)\d{2}", filename_stem)
    if m:
        return m.group(0)
    candidates = re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", text[:4000])
    if candidates:
        return Counter(candidates).most_common(1)[0][0]
    return None


def guess_doi(text: str):
    m = DOI_RE.search(text)
    if not m:
        return None
    return m.group(0).rstrip(".,;)")


def guess_arxiv(text: str):
    m = ARXIV_RE.search(text)
    return m.group(1) if m else None


def guess_venue(text: str, doi, arxiv_id):
    keywords = [
        "journal", "proceedings", "conference", "transactions", "ieee",
        "elsevier", "springer", "mdpi", "energies", "energy", "solar",
        "renewable", "applied", "sensors", "nature",
    ]
    for line in text[:3000].splitlines():
        s = line.strip()
        if 3 < len(s) < 120 and any(k in s.lower() for k in keywords):
            return s
    if doi:
        return f"DOI:{doi}"
    if arxiv_id:
        return f"arXiv:{arxiv_id}"
    return None


def guess_citation_key(text: str, filename_stem: str, year):
    head = text[:2000]
    idx_abstract = head.lower().find("abstract")
    search_region = head[:idx_abstract] if idx_abstract > 0 else head
    name_match = re.search(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", search_region)
    surname = name_match.group(2) if name_match else None

    slug_source = None
    for line in text.splitlines():
        s = line.strip()
        if len(s) > 15 and not s.lower().startswith(("abstract", "keywords")):
            slug_source = s
            break

    slug = ""
    if slug_source:
        stop = {
            "a", "an", "the", "of", "for", "and", "in", "on", "using",
            "with", "based", "to", "review", "survey", "via", "from",
        }
        words = [w.lower() for w in re.findall(r"[A-Za-z]+", slug_source)
                 if w.lower() not in stop][:3]
        slug = "".join(w[:6] for w in words)

    guess = f"{(surname or 'unknown').lower()}{year or 'xxxx'}{slug or 'paper'}"
    guess = re.sub(r"[^a-z0-9]", "", guess)
    return guess or re.sub(r"[^a-z0-9]", "", filename_stem.lower())


def extract_abstract_tail(text: str, n_sentences: int = 3):
    lower = text.lower()
    start = lower.find("abstract")
    if start == -1:
        fallback = text[:1200]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", fallback) if s.strip()]
        return None, sentences[-n_sentences:]

    search_from = start + len("abstract")
    end = len(text)
    for marker in ("keywords", "1. introduction", "1 introduction",
                   "index terms", "\nintroduction"):
        pos = lower.find(marker, search_from)
        if pos != -1:
            end = min(end, pos)
    abstract = text[search_from:end].strip()
    if len(abstract) < 50:
        fallback = text[:1200]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", fallback) if s.strip()]
        return None, sentences[-n_sentences:]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if s.strip()]
    return abstract, sentences[-n_sentences:]


# ---------------------------------------------------------------------------
# prompting
# ---------------------------------------------------------------------------

def prompt_with_default(prompt_text: str, default, required: bool) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt_text}{suffix}: ").strip()
        if raw:
            return raw
        if default:
            return default
        if not required:
            return ""
        print("A value is required here - please type one.")


def prompt_choice(column: str, allowed, allow_blank_default: bool) -> str:
    default_note = " (Enter = not_stated)" if allow_blank_default else " (no default - type one)"
    while True:
        raw = input(f"{column} [{'/'.join(allowed)}]{default_note}: ").strip()
        if raw == "":
            if allow_blank_default:
                return "not_stated"
            print(f"'{column}' requires an explicit answer - not_stated is not "
                  f"a valid value for this column. Type one of: {', '.join(allowed)}")
            continue
        if raw in allowed:
            return raw
        lower_map = {a.lower(): a for a in allowed}
        if raw.lower() in lower_map:
            return lower_map[raw.lower()]
        print(f"'{raw}' is not an allowed value. Allowed: {', '.join(allowed)}")


# ---------------------------------------------------------------------------
# per-paper coding session
# ---------------------------------------------------------------------------

def code_one_paper(path: Path, text: str, page_offsets, metadata: dict, done_keys: set):
    ev = [f"SOURCE_FILE: {path.name}", f"EXTRACTED_CHARS: {len(text)}",
          f"PDF_METADATA: {metadata}", ""]

    year_guess = guess_year(path.stem, text)
    doi_guess = guess_doi(text)
    arxiv_guess = guess_arxiv(text)
    key_guess = guess_citation_key(text, path.stem, year_guess)

    print(f"\nGuessed citation_key: '{key_guess}' "
          f"(from filename/title heuristics - verify against the actual paper)")
    while True:
        citation_key = prompt_with_default("citation_key", key_guess, required=True)
        if citation_key in done_keys:
            print(f"'{citation_key}' is already coded in the CSV. Enter a different key.")
            continue
        break
    ev.append(f"CITATION_KEY: {citation_key}")

    print(f"Guessed year: {year_guess or '(no guess)'}")
    year = prompt_with_default("year", year_guess, required=True)
    ev.append(f"year -> {year}")

    if doi_guess:
        print(f"Found possible DOI: {doi_guess}")
    if arxiv_guess:
        print(f"Found possible arXiv id: {arxiv_guess}")
    venue_guess = guess_venue(text, doi_guess, arxiv_guess)
    print(f"Guessed venue: {venue_guess or '(no guess)'}")
    venue = prompt_with_default("venue (include DOI/arXiv id if useful)", venue_guess, required=True)
    ev.append(f"venue -> {venue}")

    dataset_hits = find_matches(text, page_offsets, DATASET_HINT_TERMS)
    print_matches("dataset (free text - hints only, not a restricted vocabulary)",
                  DATASET_HINT_TERMS, dataset_hits)
    dataset = prompt_with_default(
        "dataset (describe what data the paper used)", None, required=False
    ) or "not_stated"
    ev.append(f"dataset -> {dataset}")
    for h in dataset_hits:
        ev.append(f"  hint p.{h['page']} ('{h['term']}'): {h['snippet']}")

    row = {"citation_key": citation_key, "year": year, "venue": venue, "dataset": dataset}

    for column in JUDGEMENT_COLUMNS:
        hits = find_matches(text, page_offsets, TRIGGER_TERMS[column])
        print_matches(column, TRIGGER_TERMS[column], hits)
        allow_blank = "not_stated" in ALLOWED_VALUES[column]
        value = prompt_choice(column, ALLOWED_VALUES[column], allow_blank)
        row[column] = value
        ev.append(f"\n[{column}]")
        if hits:
            for h in hits:
                ev.append(f"  p.{h['page']} ('{h['term']}'): {h['snippet']}")
        else:
            ev.append("  NO MATCHES FOUND")
        ev.append(f"  -> CODED AS: {value}")

    n_seeds = prompt_with_default(
        "n_seeds (number of seeds/runs the variance claim above is based on, "
        "or leave blank for not_stated)", None, required=False
    ) or "not_stated"
    row["n_seeds"] = n_seeds
    ev.append(f"\nn_seeds -> {n_seeds}")

    abstract, tail_sentences = extract_abstract_tail(text)
    print("\n--- abstract, last up to 3 sentences (for key_claim) ---")
    if abstract is None:
        print("(no 'Abstract' heading found - showing the tail of the first "
              "~1200 characters instead)")
    if tail_sentences:
        for s in tail_sentences:
            print(f"  {s}")
    else:
        print("  (nothing extracted)")
    key_claim = ""
    while not key_claim:
        key_claim = input(
            "key_claim - your one-sentence summary (required, write it "
            "yourself - you must be able to defend it): "
        ).strip()
    row["key_claim"] = key_claim
    ev.append(f"\nABSTRACT_TAIL: {tail_sentences}")
    ev.append(f"KEY_CLAIM: {key_claim}")

    strengths = input(
        "What does this paper do WELL (reports skill score? excludes night "
        "hours? reports variance across seeds? releases code?) - describe, "
        "or leave blank: "
    ).strip()
    ev.append(f"STRENGTHS: {strengths}")

    general_notes = input("Any other notes (free text, or leave blank): ").strip()
    ev.append(f"GENERAL_NOTES: {general_notes}")

    notes_parts = []
    if strengths:
        notes_parts.append(f"STRENGTHS: {strengths}")
    if general_notes:
        notes_parts.append(general_notes)
    row["notes"] = " | ".join(notes_parts)

    # Every row this script produces is backed by an evidence/<key>.txt log
    # of verbatim quotes with page numbers - that is the tool's whole design
    # (see WHY THIS IS AN EVIDENCE EXTRACTOR above). "summary_only" only
    # applies to rows entered some other way, before this column existed.
    row["evidence_level"] = "quoted"

    # leakage_flag is NOT auto-detected - it has no trigger terms and no
    # prompt step of its own yet, unlike the JUDGEMENT_COLUMNS above. It
    # defaults to "none" here only so the CSV stays well-formed; that is a
    # placeholder, not a finding, and violates this file's own "no
    # auto-classify" principle if left unreviewed. Read the evidence log by
    # hand and correct to "suspected" or "documented" before treating this
    # value as coded.
    row["leakage_flag"] = "none"

    return row, "\n".join(ev) + "\n"


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def read_csv_header_and_keys(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


def validate_row(row: dict) -> None:
    """Refuse to write a row with a value outside ALLOWED_VALUES for any
    judgement column. This is the mistake that produced the 2026-08-06
    literature_survey.csv free-text entries like "80/20 chronology not
    stated" and "not stated" (space, not the allowed "not_stated") - those
    rows were entered before this tool's ALLOWED_VALUES existed and had to
    be normalized by hand afterward. Raises ValueError instead of writing,
    so a bad value is caught at write time, not discovered later during a
    CSV-wide audit.
    """
    for column, allowed in ALLOWED_VALUES.items():
        if column not in row:
            continue
        value = row[column]
        if value not in allowed:
            raise ValueError(
                f"Refusing to write row for '{row.get('citation_key', '?')}': "
                f"column '{column}' has value {value!r}, which is not in "
                f"ALLOWED_VALUES[{column!r}] = {allowed}. Fix the value "
                f"(free text belongs in 'notes', not in a judgement column) "
                f"and try again."
            )


def append_row(csv_path: Path, header, row: dict):
    validate_row(row)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(row)


def find_source_done_set(evidence_dir: Path, done_keys: set):
    source_done = set()
    if not evidence_dir.exists():
        return source_done
    for ev_path in evidence_dir.glob("*.txt"):
        content = ev_path.read_text(encoding="utf-8", errors="replace")
        src_m = re.search(r"^SOURCE_FILE:\s*(.+)$", content, re.MULTILINE)
        key_m = re.search(r"^CITATION_KEY:\s*(.+)$", content, re.MULTILINE)
        if src_m and key_m and key_m.group(1).strip() in done_keys:
            source_done.add(src_m.group(1).strip())
    return source_done


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    p.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    p.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    p.add_argument("--dry-run", action="store_true",
                    help="run the full search-and-prompt flow but write nothing")
    p.add_argument("--limit", type=int, default=None,
                    help="stop after this many NEW papers (for testing)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.csv_path.exists():
        sys.exit(f"ERROR: {args.csv_path} does not exist.")
    if not args.papers_dir.exists():
        sys.exit(f"ERROR: {args.papers_dir} does not exist.")

    header, existing_rows = read_csv_header_and_keys(args.csv_path)
    if header != EXPECTED_COLUMNS:
        sys.exit(
            "ERROR: CSV header does not match the schema this script expects.\n"
            f"  on disk:  {header}\n"
            f"  expected: {EXPECTED_COLUMNS}\n"
            "Refusing to write - fix the mismatch or update EXPECTED_COLUMNS "
            "to match a deliberate schema change."
        )
    done_keys = {r["citation_key"] for r in existing_rows}

    candidate_files = sorted(
        p for p in args.papers_dir.iterdir()
        if p.suffix.lower() in {".pdf", ".txt"}
    )
    source_done = find_source_done_set(args.evidence_dir, done_keys)

    to_process = [f for f in candidate_files if f.name not in source_done]
    skipped_done = len(candidate_files) - len(to_process)

    print(f"Found {len(candidate_files)} candidate files in {args.papers_dir}")
    print(f"Skipping {skipped_done} already coded (citation_key present in "
          f"{args.csv_path.name}).")
    if args.limit is not None:
        to_process = to_process[: args.limit]
        print(f"--limit set: processing at most {args.limit} new paper(s).")
    print(f"{len(to_process)} to process this run.\n")

    skipped_extraction = 0
    processed = 0
    total = len(to_process)

    for idx, path in enumerate(to_process, start=1):
        print("=" * 70)
        print(f"paper {idx} of {total}: {path.name}")
        print("=" * 70)

        try:
            text, page_offsets, metadata = load_paper_text(path)
        except Exception as e:
            print(f"!! EXTRACTION FAILED for {path.name}: {e}")
            print("!! Skipping. It will be retried next run.")
            skipped_extraction += 1
            continue

        if len(text.strip()) < MIN_EXTRACTED_CHARS:
            print(f"!! Extracted only {len(text.strip())} characters "
                  f"(< {MIN_EXTRACTED_CHARS}).")
            print("!! Looks like a scanned PDF or a failed extraction, not "
                  "real text - refusing to code this from nothing.")
            print("!! Skipping rather than writing a row of fabricated "
                  "not_stated values.")
            skipped_extraction += 1
            continue

        row, evidence_text = code_one_paper(path, text, page_offsets, metadata, done_keys)

        if args.dry_run:
            print("\n[DRY RUN] would append this row (nothing written):")
            for col in header:
                print(f"  {col}: {row[col]}")
        else:
            append_row(args.csv_path, header, row)
            args.evidence_dir.mkdir(exist_ok=True)
            (args.evidence_dir / f"{row['citation_key']}.txt").write_text(
                evidence_text, encoding="utf-8"
            )
            done_keys.add(row["citation_key"])
            print(f"\nWrote row for '{row['citation_key']}' to "
                  f"{args.csv_path.name} and evidence/{row['citation_key']}.txt")

        processed += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  candidates found:            {len(candidate_files)}")
    print(f"  skipped (already coded):     {skipped_done}")
    print(f"  skipped (extraction failed): {skipped_extraction}")
    print(f"  processed this run:          {processed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
