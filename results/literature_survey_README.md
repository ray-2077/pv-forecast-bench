# results/literature_survey.csv - schema and coding rules

Companion document for `results/literature_survey.csv`. Read this before
adding, editing, or citing a row.

## Purpose

This CSV is the coded output of a verbatim-quote literature survey of PV
power forecasting papers, built specifically to measure how this
literature reports (or fails to report) evaluation protocol: night-hour
handling, reference-forecast choice, skill scoring, weather-input source,
split methodology, seed variance, code availability, and data-leakage
risk. It is the evidence behind this paper's Related Work section and
its central claim that reported accuracy in this literature is at least
as attributable to evaluation choices as to architecture.

The coding tool is `scripts/code_literature_survey.py`. Its own
docstring is the authoritative source if this document and the script
ever disagree - this file is a summary of it, not a replacement.

## Durability model

`data/papers/` (the source PDFs) is gitignored - they are copyrighted
and not ours to redistribute. `results/literature_survey.csv` and the
`evidence/<citation_key>_audit.md` files ARE version-controlled and are
the only durable record of how each row was coded. If a PDF is ever
lost, the corresponding audit file's citation (authors, year, title,
venue, and DOI/arXiv id where available) is what makes the paper
re-obtainable, and its verbatim quotes (with page numbers) are what
makes every coded field independently checkable without the original
file. Treat `evidence/` as load-bearing, not disposable.

## Schema

14 original columns plus 3 added 2026-08-06/08 (`evidence_level`,
`leakage_flag`, `doi`). Exact column order, as enforced by
`EXPECTED_COLUMNS` in `scripts/code_literature_survey.py`:

| Column | Type | Meaning |
|---|---|---|
| `citation_key` | free text, unique | `authorYEARshortname`, e.g. `mayer2022physmlhybrid`. Primary key; also the audit filename stem (`evidence/<citation_key>_audit.md`). |
| `year` | free text | Publication year, as stated by the paper. |
| `venue` | free text | Journal/conference, volume, article number or page range. |
| `dataset` | free text (not restricted vocabulary) | Descriptive summary of the paper's data source - site, technology, period, resolution. A nudge list of hint terms exists (`DATASET_HINT_TERMS`) but this column is never validated against a fixed list. |
| `night_hours_excluded` | restricted | Does the paper state that night hours are excluded from evaluation, included, or does it never say? |
| `baseline_used` | restricted | What reference/comparator the paper's own evaluation uses. |
| `skill_score_reported` | restricted, yes/no only | Does the paper report a skill score against ANY reference forecast? |
| `weather_source` | restricted | Where the paper's weather input data comes from. |
| `split_type` | restricted | How the paper describes its train/test(/val) split methodology. |
| `n_seeds` | free text (not restricted vocabulary) | Number of random seeds/repeated runs stated, or `not_stated`. |
| `variance_reported` | restricted, yes/no only | Does the paper report any run-to-run (seed) variance statistic? |
| `code_available` | restricted | Does the paper state that code is available? |
| `key_claim` | free text | One-sentence paraphrase of the paper's own headline result, with page citation. |
| `notes` | free text, must be under 300 characters when written by the interactive tool | Anything else load-bearing: flags, cross-paper connections, format caveats. Not restricted vocabulary, no hard length cap when edited outside the tool (several rows exceed 300 characters after later corrections were appended - see git history), but keep it short by convention. |
| `evidence_level` | restricted | See "evidence_level" below. |
| `leakage_flag` | restricted | See "leakage_flag" below. |
| `doi` | free text (not restricted vocabulary) | The paper's own DOI (bare form, e.g. `10.3390/en17071781`, no `https://doi.org/` prefix), or the literal string `not_stated`. Populated 2026-08-08 from each audit file's `DOI:` line - see "doi" below. This is what `scripts/build_bibtex.py` (not yet written - see `paper/WRITING_BRIEF.md` Section 9, GAPS) will read to build the paper's `.bib` file. |

## Allowed values (restricted-vocabulary columns)

Exact lists from `ALLOWED_VALUES` in `scripts/code_literature_survey.py`.
A value outside these lists will be rejected: `append_row()` calls
`validate_row()` first and raises `ValueError` naming the offending
column and value rather than writing a bad row (added 2026-08-07, after
several rows were found with non-conforming free text left over from
before this validation existed - see git history for
`faa0e86`/`3b91a74`).

- `night_hours_excluded`: `yes`, `no`, `partial`, `not_stated`
- `baseline_used`: `persistence`, `climatology`, `convex`, `own_components`,
  `other_ML`, `none`, `not_stated`
  - `convex` means the optimal convex combination of climatology and
    persistence, per Yang et al. (2020), Solar Energy 210:20-37 - the
    same reference construction this project's own headline metric uses.
    Distinct from `persistence` and `climatology` alone.
- `skill_score_reported`: `yes`, `no`
- `weather_source`: `measured`, `NWP_forecast`, `reanalysis`, `both`,
  `none`, `not_stated`
  - `measured` = station observation at or near the site.
  - `NWP_forecast` = a forecast available at issue time.
  - `reanalysis` = historical model products such as MERRA-2 or ERA5 -
    distinct from both of the above.
- `split_type`: `chronological`, `random`, `k-fold`, `rolling`,
  `not_stated`
- `variance_reported`: `yes`, `no`
- `code_available`: `yes`, `no`, `not_stated`
- `evidence_level`: `quoted`, `summary_only` - see below.
- `leakage_flag`: `none`, `suspected`, `documented` - see below.

## The core coding rule: not_stated is a finding, not a gap

Five columns (`night_hours_excluded`, `baseline_used`, `weather_source`,
`split_type`, `code_available`) allow `not_stated` as an explicit,
first-class value. This is deliberate, not a placeholder for missing
work.

Part of this paper's claim is about what this literature FAILS to
report. If a paper never says whether it excluded night hours, that
absence is itself data about the state of reporting practice - not a
gap to be filled in by inference. A classifier or a coder that infers
"probably excluded night hours, it mentions daylight somewhere" would
quietly convert an absence-of-reporting result into a
presence-of-practice result, and this survey's central claim would be
built on fabricated codings.

Consequently:
- Every `not_stated` coding in this CSV should be backed, in the row's
  audit file, by an EXHAUSTIVE CHECK statement - a list of the specific
  terms searched for and a confirmation that none were found describing
  the paper's own work (not a cited related-work sentence describing a
  different study). This is what makes `not_stated` defensible as a
  finding rather than an admission that the coder didn't look hard
  enough.
- Two columns (`skill_score_reported`, `variance_reported`) do NOT allow
  `not_stated` - whether a paper's results section contains a skill-score
  number or a variance statistic is something a coder can always
  determine by reading it, so these are forced to `yes`/`no`.
- Never infer a value from what "most papers like this one" probably do,
  from an abstract's vague language, or from a cited OTHER paper's stated
  practice. Code only what THIS paper's own text states about THIS
  paper's own evaluation.

## `evidence_level`: quoted vs summary_only

- `quoted` = an `evidence/<citation_key>_audit.md` file exists, with a
  verbatim quote and page number behind every coded judgement field.
  Every row written by `scripts/code_literature_survey.py` gets this
  value automatically and unconditionally - the tool's whole design is
  that it never writes a row without producing the matching evidence log
  first (see its module docstring, "WHY THIS IS AN EVIDENCE EXTRACTOR,
  NOT A CLASSIFIER").
- `summary_only` = the row was entered some other way, before this
  distinction existed, with no verbatim-quote audit file and (for the
  two rows currently in this state) no locatable source PDF either. Any
  count or claim drawn from this CSV that needs to survive a reviewer
  asking "show me the quote" should be restricted to `evidence_level=
  quoted` rows, or should say explicitly that it includes 2 unaudited
  rows.
- As of 2026-08-07: 25 of 27 rows are `quoted`; 2
  (`xu2025lstmxgboosteemdso`, `energyeng2025cnnlstmcascade`) are
  `summary_only`.

## `leakage_flag`: none / suspected / documented

Tracks whether a paper's own described methodology contains a data-
leakage pattern (Kapoor & Narayanan 2023 taxonomy, Patterns 4:100804),
independent of the seven judgement columns above.

- `documented` = the paper's OWN TEXT, quoted verbatim in its audit
  file, states the leaky procedure directly - e.g. a stated sequence
  where a decomposition or feature-selection step runs before the
  train/test split, or a stated input-feature list that includes a
  quantity algebraically derived from the prediction target. This is
  the strict bar: `documented` is only used when the paper's own
  sentence, not the coder's inference, describes the leak.
  - As of 2026-08-07, three rows: `bhutta2024hcrnhcln` (Performance
    Ratio, target-derived, used as 1 of 3 input features),
    `li2022eemdssalstm` (EEMD decomposition stated to run before the
    split), `zhou2024cnnlstmattnbayes` (two DKASC target-derived columns
    used as input features - the same columns this project's own
    `src/data/loader.py` drops for the same reason).
- `suspected` = a leakage RISK is identifiable from what the paper
  states, but the paper does not confirm the leaky detail directly - for
  example, a stated k-fold cross-validation step on time-series data
  where the paper never says whether the folds are chronological or
  randomly shuffled. Do not upgrade `suspected` to `documented` without
  a verbatim quote that removes the ambiguity.
- `none` = no leakage pattern identified from the paper's stated
  methodology. This is the default; it does NOT mean the paper's
  methodology was reviewed exhaustively for every possible leakage mode,
  only that none of the specific patterns checked were found stated.
  `leakage_flag` currently has no trigger-term list or prompt step of
  its own in the coding tool (unlike the seven judgement columns above)
  - see the comment on `row["leakage_flag"] = "none"` in
  `scripts/code_literature_survey.py` - so a coder should read the
  evidence log by hand before treating a `none` value as a considered
  judgement rather than an unreviewed default.

## `doi`

The paper's own DOI, bare form (no `https://doi.org/` prefix, no
trailing punctuation) - e.g. `10.3390/en17071781`. Added 2026-08-08 as
part of a durability audit: `data/papers/` is gitignored, so
`evidence/<citation_key>_audit.md`'s citation line (authors, year,
title, venue) is the only durable, version-controlled way to re-obtain a
paper if its PDF is ever lost, and a DOI makes that lookup a single
step instead of a journal-plus-volume-plus-page search.

- Every one of the 25 `evidence_level=quoted` audit files has an exact
  `DOI: https://doi.org/<doi>` line, added right before the `## Coded
  fields` heading. 19 of the 25 needed this added from scratch (the DOI
  was found in the source PDF's own header/footer or citation block on
  every one of the 19 - none needed the `not_stated` fallback in
  practice); the other 6 already had a DOI quoted somewhere in prose and
  got the same clean line added for consistent machine extraction.
- The `doi` column in this CSV is populated FROM those audit-file `DOI:`
  lines, by direct extraction, not retyped by hand - see
  `scripts/code_literature_survey.py`'s docstring or git history
  (2026-08-08) for the exact regex if reproducing this.
- The 2 `evidence_level=summary_only` rows (`xu2025lstmxgboosteemdso`,
  `energyeng2025cnnlstmcascade`) have no audit file to extract a DOI
  from, and their DOIs are NOT recorded anywhere else in this repo - the
  `doi` column is `not_stated` for both, for that reason, not because
  the papers themselves lack a DOI.
- If a future paper's own text genuinely never states a DOI (no arXiv id
  either), record the literal string `not_stated` in both the audit
  file's `DOI:` line and this column - do not leave the field blank, and
  do not guess a DOI from the venue/volume/page alone.

## Where to look next

- `scripts/code_literature_survey.py` - the coding tool and the
  authoritative schema definition.
- `evidence/<citation_key>_audit.md` - the verbatim-quote audit trail
  behind every `quoted` row.
- `paper/literature_notes.md` - narrative notes on the small set of
  methodology/framework papers (Yang et al. 2020, Kapoor & Narayanan
  2023, etc.) that this survey is built against, as distinct from the 27
  surveyed hybrid-architecture papers coded in this CSV.
- `paper/WRITING_BRIEF.md`, Section 8 (Citation Plan) - which papers from
  this survey are cited where in the paper, and for what specific claim.
