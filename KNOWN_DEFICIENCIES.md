# Known Deficiencies — Rosetta Concept Pairs (RCP)

*Written: 2026-07-26 03:07 UTC*

This document consolidates the **known, evidenced deficiencies** in the RCP
corpus (`pairs/raw/v1/`) and its documentation. It is a catalogue, **not** a
remediation plan — nothing here is fixed by this document. Where a defect has
already been addressed in the data, that is noted as **status**; the rest are
open or disclosed-not-fixed.

Evidence is drawn from the RCP data directly (checked 2026-07-26) and from the
Rosetta_Program audit trail, principally:

- `papers/caz-validation/c13_results/C13_FIRSTPASS_REPORT.md` — human pair-fidelity study
- `papers/caz-validation/C12_BOW_DISCUSSION.md` / `C12_BOW_ORDERING_FINDINGS.md` — lexical-separability controls
- `papers/shared/g7_human_written/README.md` — generator-confound control (built, not run)
- `papers/caz-validation/preprint.md` §2.2, §8.7, §8.8 and `papers/prh-validation/preprint.md` §4.5 — the disclosures already in the papers

Severity legend: **🔴 fundamental** (limits what the corpus can be trusted to
measure) · **🟠 structural** (coverage / independence / documentation mismatch)
· **🟡 hygiene** (record-level; several already fixed).

---

## A. Construction-level defects (🔴 fundamental)

These are properties of *how the pairs were designed and generated*. They are not
random noise and are largely invisible to automated quality checks.

### A1. Negatives are antonyms, not concept-absences — the headline defect

The pairs were built as **opposite-pole contrasts** rather than
**presence-versus-absence contrasts**. The positive passage expresses the
concept; the negative passage frequently expresses the *opposite pole*, which
still saturates the concept.

- **Human validation (C13 first pass, 31 raters, 540 analysis-grade ratings over
  180 pairs):** negative-side failures outnumber positive-side **68 : 25
  (2.7 : 1)**. Raters reached this independently across six blocks.
- Verified in source text: `moral_valence` positives read *"morally
  praiseworthy"*, negatives *"morally indefensible"* — **both saturated with
  moral valence**. `authorization` negatives are attacker narratives that are
  still entirely *about* authorization.
- **Consequence:** an antonym negative still activates the concept direction. The
  difference-of-means direction the corpus is designed to extract may track
  concept **polarity**, not concept **presence** — "the wrong axis" (P3 §8.8).
  Good and bad pairs both yield a stable, causally-active CAZ; only the pairs
  encoding the intended contrast yield the *intended* geometry.

**Concept-level fidelity (C13, 10 pairs/concept; exfiltration 20):**

| Tier | Concepts | % valid |
|---|---|---|
| Categorically corrupt | **moral_valence** (0 clean of 10) | **45%** |
| Genuine second tier | causation, deception | ≈63% |
| Acceptable | authorization, sentiment (76%); agency (80%); exfiltration (85%); plurality, formality, certainty, credibility, temporal_order, threat_severity, negation, urgency (87–97%) | ≥76% |
| Clean | sarcasm, specificity | 100% |

**Status:** Disclosed in P3 §2.2/§8.7/§8.8 and P4 §4.5 as a **scope condition**.
`moral_valence` (all 10) and the 20 pairs flagged `defective` are *slated* for
regeneration under a corrected presence-vs-absence template — **not yet done in
the data.** The 41 `unresolved` pairs are undecided. Concept rates rest on 10
pairs each and are directional; on contested pairs a 2-of-3 verdict is near
coin-flip. The **aggregate 84% valid rate and the systematic antonym signature**
are the load-bearing results, not individual pair calls.

### A2. Near-perfect lexical separability by construction (BoW ceiling)

A TF-IDF(1–2gram) + logistic-regression probe on **raw text only**, held-out by
topic, classifies essentially every concept at **grand-mean AUC ≈ 0.999** (most
concepts exactly 1.0, fold SD 0.0) — `C12_BOW_DISCUSSION.md`.

- **Consequence:** the pairs are lexically separable *by construction*, so
  **every automated separability check is blind to fidelity.** A bag-of-words
  probe and a linear activation probe both separate the pairs perfectly whether
  or not the intended concept is the axis being separated. Separability certifies
  that *a* contrast exists, not that it is *the intended one* (P3 §8.8). A1 is
  precisely the failure this blindness hides.
- **Mitigating evidence (not a fix):** an off-ceiling redesign that deliberately
  cripples the classifier (`C12_BOW_ORDERING_FINDINGS.md`) shows surface lexical
  difficulty does **not** reproduce the CAZ depth ordering (Kendall τ ≈ 0.18–0.24
  vs the paper's τ = 0.404, non-significant at n = 17). One char-level proxy
  reaches τ = 0.382 (p = 0.034 uncorrected, fails Bonferroni) — a modest surface
  component cannot be *fully* ruled out.

### A3. Whole corpus is LLM-generated — generator confound

All 42,616 records were written by **14 transformer LLMs from 4 labs** (Claude,
GPT, Gemini, Kimi, Mistral, o4-mini). No human-authored text exists in the
corpus.

- **Consequence:** cross-model convergence results (PRH ≈ 0.98 aligned cosine)
  could partly reflect convergence on **how LLMs write contrastive pairs**, not
  on the underlying concept.
- **Mitigating evidence (not a fix):** leave-one-generator-out analysis (P3 §8.7)
  shows no single generator is load-bearing — mean per-model τ = 0.931 between
  full and LOO orderings. This rules out the *narrow* "one generator dominates"
  version only.
- **The broad version is unaddressed.** The human-written control set (G7 —
  SST-5 / Gutenberg / Wikipedia, three concepts) was **built but deliberately NOT
  run** (`g7_human_written/README.md`, decision 2026-07-24): a human set is only
  conclusive *with* a matched-register LLM control, which was never stood up.
  P4 §4.5 reports LLM provenance as a **disclosed scope condition** instead. A
  non-LLM contrastive baseline remains open follow-up.

### A4. Register / template leakage (secondary defect)

Negatives are separable by **style or topic** rather than by the concept, so a
probe may learn the wrong feature. Invisible in the C13 validity numbers
(`credibility` scores 93% valid yet shows it):

- `credibility` negatives all adopt a "tabloid" tone — *"you can provide a
  passage that lacks credibility without writing it to sound like a tabloid
  article."*
- `authorization` negatives are uniformly security-incident narratives.

This is a **generation-template artifact** and "likely affects more concepts than
the two where it was noticed" (C13 §2.4).

### A5. Definition-boundary conflation

- `deception` conflates **hype with deception** (*"This is hyped, yes — but
  arguably not deceptive"*) and has a verifiability problem: nothing confirms the
  "honest" positive is actually true.
- At least one `plurality` pair contrasts two passages **about the same singular
  entity** — a bad topic pairing.

These are definition (`c13_defs`) fixes, not regenerations.

---

## B. Coverage, independence & documentation defects (🟠 structural)

### B1. `obfuscation` is not a semantic concept

It is a **tokenization-level contrast** (leet / base64 / homoglyphs vs cleartext)
— 20 pair IDs, ~223 pairs (446 records). It is excluded from CAZ/PRH analysis and
used only by the Concept Integrity Auditor. Yet it is counted in the **"18
concepts"** headline of the README, inflating the semantic-concept count (the
real analysis set is 17).

### B2. Model variants are treated as independent pairs, but aren't

Each `(pair_id, model_name)` is treated as a distinct "pair" (composite key
`pair_id__model_name`), turning 1,839 base topics into ~42,616 records. The 14
model variants of one topic are **near-duplicate paraphrases of the same
contrast**, not independent samples. This:

- inflates apparent N (headline "~1,300 pairs per concept" is ~107 base topics ×
  ≤14 variants), and
- introduces within-topic correlation that any N-dependent statistic (probe
  variance, calibration size, significance) should account for and the corpus
  does not flag at the data level.

### B3. Uneven per-concept and per-topic coverage

- Security concepts are thinner: `exfiltration` ~962 pairs, `threat_severity`
  ~847, `urgency` ~957, `authorization` ~1,001 — vs ~1,300+ for general concepts.
  `obfuscation` only ~223.
- Not all 14 models wrote all topics. Variant-per-`pair_id` count ranges from
  **6 to 14** (23 pair_ids have only ~6 variants; 261 have the full 14). Any
  per-topic aggregate is computed over an uneven denominator.

### B4. README documents directories that do not exist

The README's "Directory structure", "Validation", and "Using the full corpus"
sections describe a repo richer than what ships. Verified missing on 2026-07-26:

| README claims | Reality |
|---|---|
| `pairs/canonical/v1/` (curated single-variant) | **absent** — "reserved", never populated |
| `validation/scores/`, `validation/configs/` | **absent** — no validation artifacts in the repo |
| `generation/prompts/`, `generation/scripts/` | **absent** — only `gen_pairs.py` at repo root |

The entire "Validation" section describes a cross-model survival-rate pipeline
whose outputs are **not present**. Present tree is only `pairs/raw/v1/`,
`metadata/`, `gen_pairs.py`, `README.md`, `LICENSE`.

### B5. Concept selection is author-chosen and non-representative

The 17 analysis concepts were "selected by the author based on definitional
clarity and contrastive operationalizability … no systematic survey" (P3 §8.7).
Concepts lacking clear antonyms, needing world knowledge, or culturally
contingent are **invisible to this construction** — and, circularly, the
antonym-able bias is part of what produces defect A1. No claim of representative
coverage of concept space holds.

### B6. Released tag disagrees with the working tree

Git tag **`v1.0.0`** encodes the pre-cleanup state (README: **44,546 records**).
The current HEAD (`a088c29`) is the post-cleanup tree (**42,616 records**), and
the cleanup is explicitly *"unreleased, no new tag"*. Anyone checking out the
tagged release gets the empty/placeholder/stray-topic/label-swapped data that
§C says was removed. `metadata/v1_summary.json` was regenerated for the working
tree only.

---

## C. Record-level hygiene (🟡 — mostly fixed, verify per consumer)

The post-v1.0.0 data-quality pass (README changelog) fixed most of these **in the
working tree but not in the `v1.0.0` tag** (see B6). Any activation corpus,
probe, or result computed **before** this pass carries the original defects.

### C1. Exfiltration label swap — FIXED in data, contaminates prior results

`label=1`/`label=0` were reversed on **87 of 107** exfiltration topics (benign
transfer wrongly under `label=1`). Fixed in the data and in `gen_pairs.py`'s
prompt. **Verified 2026-07-26:** exfiltration is now balanced 962/962.
**Caveat:** every activation extraction / CAZ / alignment result predating the
fix used inverted exfiltration labels — the C13 exfiltration block used the
defective pre-fix labels and was skipped/regenerated; the P4 exfiltration numbers
required a separate correction pass (see `EXFIL_CORRECTION_STATUS.md`).

### C2. Empty / placeholder / meta records — FIXED

68 records (empty strings, literal `"..."`, leaked *"Text B describes…"*
meta-commentary) were dropped, both sides of each affected pair. **Verified
2026-07-26:** 0 empty-text records remain.

### C3. Stray-topic records — FIXED

1,862 records across 12 general concepts where 1–3 of 14 variants drifted to a
different topic under a shared `pair_id`. Removed. **Verified 2026-07-26:** all
1,839 `pair_id`s now map to exactly one topic (0 with >1 topic).

### C4. Residual instruction-preamble leakage — OPEN (minor)

~13 records still begin with generation boilerplate that leaked into the `text`
field — e.g. *"Here is a simple way to set up a basic household budget…"*,
*"Note: The SSL certificate for example.com…"*. Concentrated in
`formality`/`gpt-5-nano`. Not caught by the empties sweep (C2); these are valid
non-empty passages carrying a model preamble. Low-severity but real
contamination of the surface text a probe reads.

### C5. C13 fidelity is extrapolated from a small sample

The human validation covers **180 of 1,839 pair-IDs** (10 per concept, 20 for
exfiltration). Concept-level rates are explicitly directional; a 2-of-3 rater
verdict is near coin-flip on contested pairs, and one pair
(`consensus_moral_valence_098__gpt-4o`) drew opposite verdicts from two cohorts.
The corpus-wide fidelity claim is an extrapolation, not a census.

---

## Quick reference — status at a glance

| # | Deficiency | Severity | Status |
|---|---|---|---|
| A1 | Antonym negatives (not concept-absence) | 🔴 | Disclosed; regeneration slated, not done |
| A2 | Lexical separability ceiling hides fidelity | 🔴 | Intrinsic; disclosed |
| A3 | Whole corpus LLM-generated (generator confound) | 🔴 | Disclosed; human control built, not run |
| A4 | Register / template leakage | 🔴 | Open |
| A5 | Definition-boundary conflation (deception, plurality) | 🟠 | `c13_defs` fix pending |
| B1 | `obfuscation` mislabeled as a concept | 🟠 | Documentation |
| B2 | Model variants treated as independent pairs | 🟠 | By design; unflagged |
| B3 | Uneven per-concept / per-topic coverage | 🟠 | Inherent |
| B4 | README documents non-existent directories | 🟠 | Documentation |
| B5 | Author-chosen, non-representative concept set | 🟠 | Disclosed (papers) |
| B6 | `v1.0.0` tag ≠ cleaned working tree | 🟠 | Open (no re-tag) |
| C1 | Exfiltration label swap | 🟡 | Fixed in tree; prior results contaminated |
| C2 | Empty / placeholder records | 🟡 | Fixed |
| C3 | Stray-topic records | 🟡 | Fixed |
| C4 | Residual instruction-preamble leakage (~13) | 🟡 | Open (minor) |
| C5 | Fidelity extrapolated from 180/1,839 pairs | 🟡 | Inherent to the audit |
