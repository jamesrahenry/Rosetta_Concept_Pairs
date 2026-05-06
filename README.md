# Rosetta Concept Pairs

Contrastive text pairs for mechanistic interpretability research. Each pair consists of two texts on the same topic — one that strongly expresses a target concept and one that doesn't — designed to extract concept directions from transformer residual streams via difference-of-means.

## What's in the dataset

**18 concepts**, **1,490 unique pair topics**, **14 model variants per topic** = **38,854 total records**.

Each pair topic was independently written by 14 diverse language models (Claude, GPT, Gemini, Kimi, Mistral, o4-mini). The models were given identical generation prompts but produced their own text — the "consensus" is in the concept labeling, not the wording.

| Concept | Pairs | Topics | Domain |
|---|---|---|---|
| `agency` | 107 | 151 | General |
| `authorization` | 20 | 20 | Security |
| `causation` | 107 | 165 | General |
| `certainty` | 107 | 164 | General |
| `credibility` | 107 | 161 | General |
| `deception` | 107 | 160 | Security |
| `exfiltration` | 20 | 20 | Security |
| `formality` | 107 | 161 | General |
| `moral_valence` | 107 | 165 | General |
| `negation` | 107 | 150 | General |
| `obfuscation` | 20 | 20 | Security |
| `plurality` | 106 | 156 | General |
| `sarcasm` | 107 | 154 | General |
| `sentiment` | 107 | 107 | General |
| `specificity` | 107 | 167 | General |
| `temporal_order` | 107 | 161 | General |
| `threat_severity` | 20 | 20 | Security |
| `urgency` | 20 | 20 | Security |

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
| `pair_id` | string | Unique pair identifier. Two records share a pair_id (label 0 and label 1). |
| `label` | int | `1` = high-concept (positive class), `0` = low-concept (negative class) |
| `domain` | string | Source domain (currently `"consensus"` for all v1 pairs) |
| `model_name` | string | Which LLM generated this text |
| `text` | string | The contrastive text |
| `topic` | string | The shared topic for the pair (e.g., "restaurant dining experience") |
| `concept` | string | The target concept this pair contrasts on |

Additional fields may appear and are captured as `metadata` by consumers (see rosetta_tools `ConceptPair.metadata`).

## Directory structure

```
Rosetta_Concept_Pairs/
  pairs/
    raw/v1/               # All 14 model variants per pair — generation artifacts
    canonical/v1/         # One text per (pair_id, label) — for downstream consumption
  validation/
    scores/               # Per-pair, per-target-model separation scores
    configs/              # Validation run configurations
  generation/
    prompts/              # Generation prompt templates per concept
    scripts/              # Multi-model generation pipeline
  metadata/
    v1_summary.json       # Concept-level statistics
```

### `pairs/raw/` vs `pairs/canonical/`

**Raw** contains every model variant — 14 texts per (pair_id, label). This is the full generation output.

**Canonical** contains one text per (pair_id, label) — the version downstream consumers should use. In v1, one model variant was selected per (pair_id, label) based on highest cross-model separation score: the text that most consistently activated the target concept across diverse validation models.

### Validation

Cross-model validation scores each pair against multiple target architectures. A pair is "consensus-validated" if the concept separates consistently across diverse models — evidence that the pair captures the concept itself, not a model-specific encoding artifact.

Validation pipeline:
1. For each (pair_id, model_variant), feed both texts through N target models
2. At each target model's CAZ peak layer, compute concept separation
3. Pairs with separation above threshold in >= K/N target models are validated
4. Survival rate = fraction of pairs that pass = dataset quality metric

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

## License

Apache 2.0
