# Rosetta Concept Pairs

Contrastive text pairs for mechanistic interpretability research. Each pair consists of two texts on the same topic — one that strongly expresses a target concept and one that doesn't — designed to extract concept directions from transformer residual streams via difference-of-means.

## What's in the dataset (v1.0.0)

**18 concepts**, **1,839 unique pair IDs**, **14 model variants per pair topic** = **44,546 total records**.

Each pair topic was independently written by 14 diverse language models (Claude, GPT, Gemini, Kimi, Mistral, o4-mini). The models were given identical generation prompts but produced their own text — the "consensus" is in the concept labeling, not the wording. Each (pair_id, model) combination is treated as a distinct pair, giving ~1,400 usable pairs per concept.

| Concept | Pair IDs | Pairs | Topics | Domain | CAZ use |
|---|---|---|---|---|---|
| `agency` | 107 | ~1,381 | 151 | General | ✓ |
| `authorization` | 107 | ~1,003 | 107 | Security | ✓ |
| `causation` | 107 | ~1,434 | 165 | General | ✓ |
| `certainty` | 107 | ~1,407 | 164 | General | ✓ |
| `credibility` | 107 | ~1,415 | 161 | General | ✓ |
| `deception` | 107 | ~1,415 | 160 | Security | ✓ |
| `exfiltration` | 107 | ~969 | 107 | Security | ✓ |
| `formality` | 107 | ~1,385 | 161 | General | ✓ |
| `moral_valence` | 107 | ~1,410 | 165 | General | ✓ |
| `negation` | 107 | ~1,322 | 150 | General | ✓ |
| `obfuscation` | 20 | ~223 | 20 | Security | CIA only¹ |
| `plurality` | 107 | ~1,421 | 157 | General | ✓ |
| `sarcasm` | 107 | ~1,403 | 154 | General | ✓ |
| `sentiment` | 107 | ~1,442 | 107 | General | ✓ |
| `specificity` | 107 | ~1,389 | 167 | General | ✓ |
| `temporal_order` | 107 | ~1,429 | 161 | General | ✓ |
| `threat_severity` | 107 | ~850 | 107 | Security | ✓ |
| `urgency` | 107 | ~975 | 107 | Security | ✓ |

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

## License

Apache 2.0
