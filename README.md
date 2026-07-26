# Rosetta Concept Pairs

Contrastive text pairs for mechanistic interpretability research. Each pair consists of two texts on the same topic — one that strongly expresses a target concept and one that doesn't — designed to extract concept directions from transformer residual streams via difference-of-means.

> ⚠️ **Read [`KNOWN_DEFICIENCIES.md`](KNOWN_DEFICIENCIES.md) before relying on this corpus.** The single most important caveat: many negatives are *opposite-pole* (antonym) passages rather than *concept-absent* ones, so a probe may recover concept **polarity** rather than concept **presence**. `moral_valence` is categorically affected (human validation: 45% valid, 0 clean pairs of 10) and is slated for regeneration. See the [Known Deficiencies](#known-deficiencies) summary below.

## What's in the dataset (v1.0.0)

**18 concepts**, **1,839 unique pair IDs**, **14 model variants per pair topic** = **42,616 total records**.

Each pair topic was independently written by 14 diverse language models (Claude, GPT, Gemini, Kimi, Mistral, o4-mini). The models were given identical generation prompts but produced their own text — the "consensus" is in the concept labeling, not the wording. Each (pair_id, model) combination is treated as a distinct pair, giving ~1,300 usable pairs per concept. Every `pair_id` now maps to exactly one topic across all its model variants (see [Changelog](#changelog)).

| Concept | Pair IDs | Pairs | Domain | CAZ use |
|---|---|---|---|---|
| `agency` | 107 | ~1,323 | General | ✓ |
| `authorization` | 107 | ~1,001 | Security | ✓ |
| `causation` | 107 | ~1,352 | General | ✓ |
| `certainty` | 107 | ~1,330 | General | ✓ |
| `credibility` | 107 | ~1,331 | General | ✓ |
| `deception` | 107 | ~1,337 | Security | ✓ |
| `exfiltration` | 107 | ~962 | Security | ✓ |
| `formality` | 107 | ~1,305 | General | ✓ |
| `moral_valence` | 107 | ~1,312 | General | ✓ |
| `negation` | 107 | ~1,257 | General | ✓ |
| `obfuscation` | 20 | ~223 | Security | CIA only¹ |
| `plurality` | 107 | ~1,346 | General | ✓ |
| `sarcasm` | 107 | ~1,333 | General | ✓ |
| `sentiment` | 107 | ~1,442 | General | ✓ |
| `specificity` | 107 | ~1,305 | General | ✓ |
| `temporal_order` | 107 | ~1,345 | General | ✓ |
| `threat_severity` | 107 | ~847 | Security | ✓ |
| `urgency` | 107 | ~957 | Security | ✓ |

**Pair IDs** = unique base topics. **Pairs** ≈ pair_ids × model variants actually present (not all models wrote all topics, hence ~). Pair counts are from `metadata/v1_summary.json`.

¹ `obfuscation` pairs are tokenization-level contrasts (leet/base64/homoglyphs vs clear text) — not a semantic concept direction. Excluded from CAZ/PRH analysis; used by the Concept Integrity Auditor (CIA) for encoding-detection probes only.

### Generating models (v1)

claude-sonnet-4-6, claude-sonnet-4-5, claude-3-7-sonnet, claude-haiku-4-5, gpt-5.4, gpt-5-mini, gpt-5-nano, gpt-4o, gemini-3.1-pro, gemini-3-flash, gemini-2.5-pro, kimi-k2.5, mistral-large, o4-mini

## Known Deficiencies

**Full catalogue: [`KNOWN_DEFICIENCIES.md`](KNOWN_DEFICIENCIES.md).** The headline items — read before extracting directions or citing this corpus:

**Fundamental (limit what the pairs can be trusted to measure):**

- **Antonym negatives, not concept-absences.** Negatives frequently express the *opposite pole* instead of *absence*, so they still activate the concept. Human validation (C13, 540 ratings over 180 pairs) found negative-side failures outnumber positive-side **2.7 : 1**; overall 84% pair-valid, but `moral_valence` **45% valid with 0 clean pairs of 10**, and `causation`/`deception` ≈63%. Fourteen of 17 concepts validate ≥76%. `moral_valence` and the 20 `defective` pairs are slated for regeneration under a corrected presence-vs-absence template — **not yet done in the data.**
- **Lexical separability by construction.** A bag-of-words probe classifies the pairs at grand-mean AUC ≈ 0.999. Any automated separability check (BoW, linear probe) therefore certifies only that *a* contrast exists — not that it is the *intended* concept. This is why the antonym defect above is invisible to automated QC. (An off-ceiling control confirms surface difficulty does *not* reproduce the CAZ depth ordering.)
- **Entirely LLM-generated (generator confound).** All records are written by 14 transformer LLMs. Cross-model convergence results may partly reflect how LLMs write contrastive pairs. Leave-one-generator-out shows no single generator is load-bearing; the broader "all generators are LLMs" concern is disclosed, not resolved (a human-written control was built but not run).
- **Register/template leakage:** negatives sometimes separable by style/topic (e.g. `credibility` negatives all "tabloid"-toned, `authorization` negatives all incident narratives), so a probe may learn register instead of concept.

**Structural:**

- `obfuscation` is a tokenization-level contrast, **not a semantic concept** — excluded from CAZ/PRH, yet counted in the "18 concepts" headline (the real analysis set is 17).
- Each `(pair_id, model_name)` is treated as an independent pair, but the 14 variants of a topic are near-duplicate paraphrases — **not independent samples**. This inflates apparent N and introduces within-topic correlation.
- **This README oversells the repo layout.** `pairs/canonical/`, `validation/`, and `generation/` (below) are **not present** — only `pairs/raw/v1/`, `metadata/`, and `gen_pairs.py` ship.
- Concept selection is author-chosen and not representative of concept space.

**Hygiene (mostly fixed in the working tree; the `v1.0.0` tag predates the fixes):**

- Exfiltration label swap (87/107 topics) — **fixed** in-tree; any activation results computed before the fix are contaminated.
- Empty/placeholder records (68) and stray-topic records (1,862) — **removed** (see [Changelog](#changelog)).
- ~13 records still carry leaked generation preamble (*"Here is a simple way to…"*) in the `text` field — open, minor.
- The **`v1.0.0` git tag encodes the pre-cleanup 44,546-record state**; the current tree is 42,616. A checkout of the tag gets the un-fixed data.

## Record schema

Each JSONL record represents one model's text for one side of one pair:

```json
{
  "pair_id": "consensus_sentiment_000",
  "label": 1,
  "domain": "consensus",
  "model_name": "claude-sonnet-4-6",
  "text": "Last Saturday's dinner at Marigold Bistro was nothing short of magical...",
  "topic": "restaurant dining experience",
  "concept": "sentiment"
}
```

| Field | Type | Description |
|---|---|---|
| `pair_id` | string | Base pair identifier. Multiple records share a pair_id across models. |
| `label` | int | `1` = high-concept (positive class), `0` = low-concept (negative class) |
| `domain` | string | Source domain (currently `"consensus"` for all v1 pairs) |
| `model_name` | string | Which LLM generated this text |
| `text` | string | The contrastive text |
| `topic` | string | The shared topic for the pair (e.g., "restaurant dining experience") |
| `concept` | string | The target concept this pair contrasts on |

When consumed via `rosetta_tools`, each (pair_id, model_name) is assigned a composite key `pair_id__model_name` and treated as an independent pair. Additional fields may appear and are captured as `metadata`.

### Label convention

`label=1` always means **the text expresses the named concept**; `label=0` means it doesn't. This is consistent across all 18 concepts — but it is *not* the same as "1 = the nice/desirable text." For concepts named after an undesirable behavior, label=1 is the undesirable one:

| Concept | `label=1` text | `label=0` text |
|---|---|---|
| `deception` | deceptive / misleading | honest / transparent |
| `exfiltration` | covert, unauthorized transfer | authorized, controlled transfer |
| `threat_severity` | critical / high severity | low severity / informational |
| `urgency` | genuine urgency | no urgency |

All other concepts read the "expected" way (e.g. `sentiment` 1=positive, `moral_valence` 1=virtuous, `credibility` 1=credible). When writing code that branches on `label`, treat it as "concept present," never assume it means "the good one."

## Directory structure

> ⚠️ **This section is aspirational, not current.** Only the lines marked *(present)* below actually ship in the repo. `pairs/canonical/`, `validation/`, and `generation/` do **not** exist — see [Known Deficiencies](#known-deficiencies).

```
Rosetta_Concept_Pairs/
  pairs/
    raw/v1/               # (present) All data — 18 JSONL files, one per concept
    canonical/v1/         # NOT PRESENT — reserved for future curated single-variant selection
  validation/             # NOT PRESENT — no validation artifacts ship in this repo
    scores/               #   (described below, but the outputs are not committed)
    configs/
  generation/             # NOT PRESENT — the only generation artifact is gen_pairs.py at the repo root
    prompts/
    scripts/
  metadata/               # (present)
    v1_summary.json       # Concept-level statistics (pair counts, record counts)
    v1_validation_split.json  # Fixed 80/20 train/validation split by base pair_id
  gen_pairs.py            # (present) generation script
  KNOWN_DEFICIENCIES.md   # (present) full deficiency catalogue
```

All data lives in `pairs/raw/v1/`. The `canonical/`, `validation/`, and `generation/` directories are described here as the intended layout but are **not populated or committed** in v1.

## Using the full corpus

For published use — including the `rcp_v1` HF extraction — use all pairs from all model variants:

```python
from rosetta_tools.dataset import load_concept_pairs
pairs = load_concept_pairs("credibility", n=2000, split="all")
# returns ~1,415 pairs (all 14 model variants × 107 base pair IDs)
```

The `split="all"` argument bypasses the internal train/validation division (which is a research tool for held-out evaluation, not relevant to downstream use). Pass a large `n` to ensure all available pairs are returned — the loader silently clamps to the number available.

## Validation

> ⚠️ **The pipeline below describes intended methodology; its outputs are not committed to this repo** (`validation/scores/` does not exist — see [Known Deficiencies](#known-deficiencies)). The pair-fidelity evidence that *does* exist is the human C13 study and the BoW controls summarized in [`KNOWN_DEFICIENCIES.md`](KNOWN_DEFICIENCIES.md), which found the antonym-negative defect that this automated separation-score pipeline is blind to.

Cross-model validation scores each pair against multiple target architectures. A pair is "consensus-validated" if the concept separates consistently across diverse models — evidence that the pair captures the concept itself, not a model-specific encoding artifact.

Validation pipeline:
1. For each (pair_id, model_variant), feed both texts through N target models
2. At each target model's CAZ peak layer, compute concept separation
3. Pairs with separation above threshold in >= K/N target models are validated
4. Survival rate = fraction of pairs that pass = dataset quality metric

Validation scores are in `validation/scores/`.

## Intended use

- **Direction extraction**: Feed pairs into a target model, compute difference-of-means at the CAZ peak layer to get the concept's direction in activation space.
- **Probe training**: Use as labeled data for linear probes on transformer hidden states.
- **Cross-model comparison**: Same pairs evaluated on different architectures reveal shared vs model-specific concept geometry (Platonic Representation Hypothesis).
- **Monitoring**: The Concept Integrity Auditor (CIA) uses these pairs to build concept probes for real-time inference monitoring.

## Relationship to other Rosetta projects

| Project | Role |
|---|---|
| **Rosetta_Concept_Pairs** (this repo) | Dataset: contrastive pairs and validation |
| **Rosetta_Tools** | Library: extraction, CAZ metrics, ablation |
| **Rosetta_Program** | Research: CAZ theory, cross-model analysis, papers |
| **Concept Integrity Auditor** | Application: real-time concept monitoring using these pairs |

## Versioning

**v1.0.1** (git tag, current): 18 concepts, 1,839 unique pair IDs, **42,616 records**. Applies the post-v1.0.0 data-quality pass (see [Changelog](#changelog)) — exfiltration label swap corrected, empty/placeholder and stray-topic records removed — and adds [`KNOWN_DEFICIENCIES.md`](KNOWN_DEFICIENCIES.md). **This is the release to use.**

**v1.0.0** (git tag): 18 concepts, 1,839 unique pair IDs, 44,546 records. Security concepts (authorization, exfiltration, threat_severity, urgency) topped up to 107 pair IDs in this release.

> ⚠️ **The `v1.0.0` tag predates the data-quality pass below** — the tagged commit still contains the exfiltration label swap, the empty/placeholder records, and the stray-topic records described in the [Changelog](#changelog). Use **`v1.0.1`** for the cleaned corpus. See [Known Deficiencies](#known-deficiencies).

## Changelog

### v1.0.1

Post-v1.0.0 data-quality pass (released as **v1.0.1**), plus the addition of [`KNOWN_DEFICIENCIES.md`](KNOWN_DEFICIENCIES.md) and the deficiency notes throughout this README:

- **exfiltration label direction**: corrected 87/107 topics where `label=1`/`label=0` had been swapped (benign transfer under label=1 instead of the malicious one). Fixed in both the data and `gen_pairs.py`'s generation prompt so future top-ups don't reintroduce it.
- **Empty/placeholder records removed**: 68 records across `authorization`, `causation`, `deception`, `exfiltration`, `moral_valence`, `specificity`, `threat_severity`, `urgency` — failed generations that landed as empty strings, literal `"..."` placeholders, or leaked meta-commentary (e.g. "Text B describes..."). Both sides of the affected `(pair_id, model_name)` were dropped to keep every pair complete.
- **Stray-topic records removed**: 1,862 records across 12 "General"-domain concepts (`agency`, `causation`, `certainty`, `credibility`, `deception`, `formality`, `moral_valence`, `negation`, `plurality`, `sarcasm`, `specificity`, `temporal_order`) where a minority of model variants (typically 1–3 of 14) had drifted onto a different topic than the rest while sharing the same `pair_id`. Kept only the majority-topic variants; every `pair_id` now maps to exactly one topic across all its models.
- No `pair_id` was dropped entirely, so `metadata/v1_validation_split.json` is unchanged. `metadata/v1_summary.json` was regenerated. Net: 44,546 → 42,616 records (1,839 pair IDs unchanged).

## License

Apache 2.0
