# Rosetta Concept Pairs

Contrastive text pairs for mechanistic interpretability research. Each pair consists of two texts on the same topic — one that strongly expresses a target concept and one that doesn't — designed to extract concept directions from transformer residual streams via difference-of-means.

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

```
Rosetta_Concept_Pairs/
  pairs/
    raw/v1/               # All data — 18 JSONL files, one per concept
    canonical/v1/         # Reserved for future curated single-variant selection (not populated)
  validation/
    scores/               # Per-pair, per-target-model separation scores
    configs/              # Validation run configurations
  generation/
    prompts/              # Generation prompt templates per concept
    scripts/              # Multi-model generation pipeline
  metadata/
    v1_summary.json       # Concept-level statistics (pair counts, record counts)
    v1_validation_split.json  # Fixed 80/20 train/validation split by base pair_id
```

All data lives in `pairs/raw/v1/`. The `canonical/` directory is reserved for a future curation step (single best-validated model variant per pair topic) but is not populated in v1.

## Using the full corpus

For published use — including the `rcp_v1` HF extraction — use all pairs from all model variants:

```python
from rosetta_tools.dataset import load_concept_pairs
pairs = load_concept_pairs("credibility", n=2000, split="all")
# returns ~1,415 pairs (all 14 model variants × 107 base pair IDs)
```

The `split="all"` argument bypasses the internal train/validation division (which is a research tool for held-out evaluation, not relevant to downstream use). Pass a large `n` to ensure all available pairs are returned — the loader silently clamps to the number available.

## Validation

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

**v1.0.0** (git tag): 18 concepts, 1,839 unique pair IDs, 44,546 records. Security concepts (authorization, exfiltration, threat_severity, urgency) topped up to 107 pair IDs in this release.

## Changelog

Post-v1.0.0 data-quality pass (unreleased, no new tag):

- **exfiltration label direction**: corrected 87/107 topics where `label=1`/`label=0` had been swapped (benign transfer under label=1 instead of the malicious one). Fixed in both the data and `gen_pairs.py`'s generation prompt so future top-ups don't reintroduce it.
- **Empty/placeholder records removed**: 68 records across `authorization`, `causation`, `deception`, `exfiltration`, `moral_valence`, `specificity`, `threat_severity`, `urgency` — failed generations that landed as empty strings, literal `"..."` placeholders, or leaked meta-commentary (e.g. "Text B describes..."). Both sides of the affected `(pair_id, model_name)` were dropped to keep every pair complete.
- **Stray-topic records removed**: 1,862 records across 12 "General"-domain concepts (`agency`, `causation`, `certainty`, `credibility`, `deception`, `formality`, `moral_valence`, `negation`, `plurality`, `sarcasm`, `specificity`, `temporal_order`) where a minority of model variants (typically 1–3 of 14) had drifted onto a different topic than the rest while sharing the same `pair_id`. Kept only the majority-topic variants; every `pair_id` now maps to exactly one topic across all its models.
- No `pair_id` was dropped entirely, so `metadata/v1_validation_split.json` is unchanged. `metadata/v1_summary.json` was regenerated. Net: 44,546 → 42,616 records (1,839 pair IDs unchanged).

## License

Apache 2.0
