#!/usr/bin/env python3
"""
gen_pairs.py — Generate additional concept pairs for sparse Rosetta concepts.

Uses FuelIX (OpenAI-compatible proxy) to call all 14 frontier models,
matching the format of existing consensus pairs. Appends to existing
JSONL files and updates metadata/split at the end. Checkpoints after
each completed topic — safe to interrupt and resume.

Usage:
    python gen_pairs.py --status
    python gen_pairs.py --concept authorization --target 100
    python gen_pairs.py --all-sparse --target 100
    python gen_pairs.py --concept exfiltration --target 100 --dry-run

Requirements:
    pip install requests tqdm
    FUELIX_API_KEY set in env or ~/Source/fuelix_kilocode_profiles/.env
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "pairs" / "raw" / "v1"
METADATA_DIR = ROOT / "metadata"
CHECKPOINT_DIR = ROOT / ".gen_checkpoints"

# ---------------------------------------------------------------------------
# FuelIX config
# ---------------------------------------------------------------------------

FUELIX_BASE_URL = os.environ.get("FUELIX_BASE_URL", "https://api.fuelix.ai")
FUELIX_API_KEY = os.environ.get("FUELIX_API_KEY", "")

if not FUELIX_API_KEY:
    _env_candidates = [
        Path.home() / "Source" / "fuelix_kilocode_profiles" / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path.cwd() / ".env",
    ]
    for _env_path in _env_candidates:
        if _env_path.exists():
            for _line in _env_path.read_text().splitlines():
                if _line.startswith("FUELIX_API_KEY="):
                    FUELIX_API_KEY = _line.split("=", 1)[1].strip().strip('"')
                elif _line.startswith("FUELIX_BASE_URL="):
                    FUELIX_BASE_URL = _line.split("=", 1)[1].strip().strip('"')
            if FUELIX_API_KEY:
                break

# Rate limit: stay comfortably under 75 req/min
_CALL_INTERVAL = 0.85
_last_call_time: float = 0.0

# ---------------------------------------------------------------------------
# Model roster (matches existing dataset)
# ---------------------------------------------------------------------------

MODELS = [
    "claude-sonnet-4-6",    # Anthropic
    "claude-sonnet-4-5",    # Anthropic
    "claude-3-7-sonnet",    # Anthropic
    "claude-haiku-4-5",     # Anthropic
    "gpt-5.4",              # OpenAI
    "gpt-5-mini",           # OpenAI
    "gpt-5-nano",           # OpenAI
    "gpt-4o",               # OpenAI
    "gemini-3.1-pro",       # Google
    "gemini-3-flash",       # Google
    "gemini-2.5-pro",       # Google
    "kimi-k2.5",            # Moonshot
    "mistral-large",        # Mistral
    "o4-mini",              # OpenAI reasoning
]

TOPIC_GEN_MODEL = "gpt-5-mini"

# After this many consecutive failures, skip a model for the rest of the run
MODEL_FAIL_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Concept definitions
# ---------------------------------------------------------------------------

CONCEPT_DEFS: dict[str, dict] = {
    "authorization": {
        "desc": "authorization and access control",
        "pos_label": "proper authorization: following access-control policies, least-privilege, "
                     "correct credential management, auditing, legitimate permission grants",
        "neg_label": "unauthorized access: bypassing authorization checks, exploiting stolen "
                     "credentials, privilege escalation, deliberately circumventing access policies",
        "domains": [
            "API key and token management", "role-based access control",
            "identity and access management", "cloud IAM policies",
            "database access control", "physical access systems",
            "OAuth and SSO flows", "service account management",
            "zero-trust network access", "privileged access workstations",
            "multi-factor authentication", "certificate-based auth",
        ],
    },
    "exfiltration": {
        "desc": "data exfiltration vs. legitimate data transfer",
        "pos_label": "authorized, controlled data transfer: documented procedures, encryption "
                     "in transit, audit logging, data governance compliance, approved sharing",
        "neg_label": "covert unauthorized exfiltration: concealing transfers, using covert channels, "
                     "evading DLP, exfiltrating via DNS/steganography, staging data without authorization",
        "domains": [
            "cloud storage exfiltration", "DNS tunneling", "email-based data theft",
            "USB and removable media", "C2 channel data staging",
            "insider threat scenarios", "HTTPS exfiltration over legitimate services",
            "database dumping", "memory scraping", "supply chain data theft",
            "printer and fax exfiltration", "steganography in images",
        ],
    },
    "threat_severity": {
        "desc": "threat severity level — critical/high vs. low/informational",
        "pos_label": "high-severity or critical threat requiring immediate response: RCE, "
                     "privilege escalation, zero-day, active breach, CVSS ≥ 9.0",
        "neg_label": "low-severity or informational finding with minimal impact: CVSS < 4, "
                     "requires physical access, no known exploitation, purely theoretical",
        "domains": [
            "CVE vulnerability advisories", "penetration test reports",
            "bug bounty findings", "SIEM alert triage",
            "incident severity classification", "red team reports",
            "security audit findings", "threat intelligence reporting",
            "patch management prioritization", "risk register entries",
            "network scan findings", "web application security reports",
        ],
    },
    "urgency": {
        "desc": "urgency and time pressure",
        "pos_label": "genuine urgency: immediate action required, time-critical deadlines, "
                     "active incident in progress, escalating consequences, explicit time constraints",
        "neg_label": "no urgency: relaxed timelines, routine matters, flexible scheduling, "
                     "informational content with no action deadline, delay carries no consequence",
        "domains": [
            "active security incident response", "medical triage",
            "financial market deadlines", "legal filing deadlines",
            "customer escalations", "production outage response",
            "regulatory compliance deadlines", "disaster recovery",
            "supply chain disruptions", "public safety alerts",
            "contract negotiation deadlines", "political campaign timing",
        ],
    },
}

SPARSE_CONCEPTS = list(CONCEPT_DEFS.keys())

# ---------------------------------------------------------------------------
# FuelIX call
# ---------------------------------------------------------------------------

def call_model(model_id: str, prompt: str, max_retries: int = 4) -> str | None:
    global _last_call_time
    import requests

    elapsed = time.time() - _last_call_time
    if elapsed < _CALL_INTERVAL:
        time.sleep(_CALL_INTERVAL - elapsed)
    _last_call_time = time.time()

    headers = {
        "Authorization": f"Bearer {FUELIX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 2000,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{FUELIX_BASE_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(60, 2 ** attempt * 5)
                log.warning("  Rate limited on %s — sleeping %ds", model_id, wait)
                time.sleep(wait)
            else:
                log.warning("  %s → %d: %s", model_id, resp.status_code, resp.text[:120])
                return None
        except Exception as e:
            log.warning("  %s error (attempt %d): %s", model_id, attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(3)
    return None


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try extracting first {...} block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None

# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def existing_topics(concept: str) -> list[str]:
    records = load_jsonl(DATA_DIR / f"{concept}_consensus_pairs.jsonl")
    seen: dict[str, str] = {}
    for r in records:
        pid = r["pair_id"]
        if pid not in seen:
            seen[pid] = r.get("topic", "")
    return list(seen.values())


def existing_pair_ids(concept: str) -> set[str]:
    return {r["pair_id"] for r in load_jsonl(DATA_DIR / f"{concept}_consensus_pairs.jsonl")}


def next_index(concept: str) -> int:
    pids = existing_pair_ids(concept)
    if not pids:
        return 0
    indices = []
    for pid in pids:
        parts = pid.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return max(indices) + 1 if indices else 0


def append_records(concept: str, records: list[dict]) -> None:
    path = DATA_DIR / f"{concept}_consensus_pairs.jsonl"
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(concept: str) -> dict:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"{concept}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_topics": [], "planned_topics": []}


def save_checkpoint(concept: str, data: dict) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    with open(CHECKPOINT_DIR / f"{concept}.json", "w") as f:
        json.dump(data, f, indent=2)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_topics(concept: str, existing: list[str], n: int) -> list[str]:
    defn = CONCEPT_DEFS[concept]
    existing_sample = "\n".join(f"- {t}" for t in existing[:60])

    prompt = (
        f"Generate {n} diverse, specific topic titles for contrastive text pairs "
        f"about: {defn['desc']}.\n\n"
        f"Each topic needs a concrete scenario where we can write:\n"
        f"  Positive (label=1): {defn['pos_label']}\n"
        f"  Negative (label=0): {defn['neg_label']}\n\n"
        f"Draw from varied domains such as: {', '.join(defn['domains'][:8])}.\n\n"
        f"Existing topics (do not duplicate):\n{existing_sample}\n\n"
        f"Return ONLY a JSON array of {n} short topic title strings. No commentary."
    )

    text = call_model(TOPIC_GEN_MODEL, prompt)
    if not text:
        raise RuntimeError(f"Topic generation failed for {concept}")

    data = parse_json(text)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list from topic gen, got: {type(data)}")

    existing_set = set(existing)
    return [t for t in data if t not in existing_set][:n]


def generate_pair(concept: str, topic: str, model_id: str) -> tuple[str, str] | None:
    defn = CONCEPT_DEFS[concept]

    prompt = (
        f'Write two texts (3–5 paragraphs each) on the topic: "{topic}"\n\n'
        f"Text A (label=1 — positive): {defn['pos_label']}\n\n"
        f"Text B (label=0 — negative): {defn['neg_label']}\n\n"
        f"Be specific, realistic, and detailed. Natural prose — no headers, "
        f"no labels, no meta-commentary about what you're writing.\n\n"
        f'Return ONLY valid JSON: {{"positive": "...", "negative": "..."}}'
    )

    text = call_model(model_id, prompt)
    if not text:
        return None

    data = parse_json(text)
    if not data or "positive" not in data or "negative" not in data:
        log.warning("  Bad JSON from %s for topic '%s'", model_id, topic[:50])
        return None

    return data["positive"], data["negative"]

# ---------------------------------------------------------------------------
# Per-concept orchestration
# ---------------------------------------------------------------------------

def run_concept(concept: str, target: int, models: list[str], dry_run: bool = False) -> None:
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = None

    curr_topics = existing_topics(concept)
    curr_count = len(existing_pair_ids(concept))
    needed = target - curr_count

    print(f"\n{'='*60}")
    print(f"  {concept}  ({curr_count} pair_ids → target {target}, need {needed} new)")
    print(f"  Models: {len(models)}")
    print(f"{'='*60}")

    if needed <= 0:
        print("  Already at or above target.")
        return

    ckpt = load_checkpoint(concept)
    completed = set(ckpt.get("completed_topics", []))
    planned = ckpt.get("planned_topics", [])

    remaining_planned = [t for t in planned if t not in completed]
    still_needed = needed - len(completed)

    if dry_run:
        n_to_brainstorm = max(0, still_needed - len(remaining_planned))
        print(f"  [dry-run] {still_needed} topics × {len(models)} models")
        if n_to_brainstorm:
            print(f"  Would brainstorm {n_to_brainstorm} new topic ideas")
        for t in remaining_planned[:5]:
            print(f"    - {t}")
        if len(remaining_planned) > 5:
            print(f"    ... and {len(remaining_planned)-5} more already planned")
        return

    if still_needed > len(remaining_planned):
        n_gen = still_needed - len(remaining_planned)
        log.info("  Brainstorming %d new topic ideas...", n_gen)
        all_existing = curr_topics + planned
        new_topics = generate_topics(concept, all_existing, n_gen)
        planned = list(completed) + remaining_planned + new_topics
        ckpt["planned_topics"] = planned
        save_checkpoint(concept, ckpt)
        log.info("  Topics ready: %d (%d already done)", len(planned), len(completed))

    todo = [t for t in planned if t not in completed][:still_needed]
    idx = next_index(concept)

    # Per-model stats for this run
    model_ok: dict[str, int] = defaultdict(int)
    model_fail: dict[str, int] = defaultdict(int)
    model_consec_fail: dict[str, int] = defaultdict(int)
    skipped_models: set[str] = set()

    iterator = _tqdm(todo, desc=concept, unit="topic") if _tqdm else todo

    for topic in iterator:
        pid = f"consensus_{concept}_{idx:03d}"
        records = []

        for model_name in models:
            if model_name in skipped_models:
                continue

            result = generate_pair(concept, topic, model_name)
            if result is None:
                model_fail[model_name] += 1
                model_consec_fail[model_name] += 1
                if model_consec_fail[model_name] >= MODEL_FAIL_THRESHOLD:
                    log.warning(
                        "  %s: %d consecutive failures — skipping for remainder of run",
                        model_name, MODEL_FAIL_THRESHOLD,
                    )
                    skipped_models.add(model_name)
                continue

            model_ok[model_name] += 1
            model_consec_fail[model_name] = 0
            pos_text, neg_text = result
            records.append({
                "pair_id": pid, "label": 1, "domain": "consensus",
                "model_name": model_name, "text": pos_text,
                "topic": topic, "concept": concept,
            })
            records.append({
                "pair_id": pid, "label": 0, "domain": "consensus",
                "model_name": model_name, "text": neg_text,
                "topic": topic, "concept": concept,
            })

        if records:
            append_records(concept, records)
            idx += 1

        completed.add(topic)
        ckpt["completed_topics"] = list(completed)
        save_checkpoint(concept, ckpt)

    # Per-model summary
    print(f"\n  --- {concept} model summary ---")
    all_models = set(list(model_ok.keys()) + list(model_fail.keys()))
    for m in sorted(all_models):
        ok = model_ok[m]
        fail = model_fail[m]
        skip_note = "  SKIPPED (too many failures)" if m in skipped_models else ""
        print(f"    {m:<25}  ok={ok:>3}  fail={fail:>3}{skip_note}")
    if skipped_models:
        print(f"\n  To permanently exclude: --skip-models {' '.join(sorted(skipped_models))}")
    print(f"\n  Done. {len(completed)} total topics completed for {concept}.")

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def update_summary() -> None:
    summary = {}
    for path in sorted(DATA_DIR.glob("*_consensus_pairs.jsonl")):
        concept = path.stem.replace("_consensus_pairs", "")
        records = load_jsonl(path)
        pair_ids = {r["pair_id"] for r in records}
        models = {r.get("model_name", "") for r in records if r.get("model_name")}
        topics = {r.get("topic", "") for r in records if r.get("topic")}
        summary[concept] = {
            "records": len(records),
            "pair_ids": len(pair_ids),
            "models": len(models),
            "model_list": sorted(models),
            "topics": len(topics),
            "file_size_kb": round(path.stat().st_size / 1024, 0),
        }
    with open(METADATA_DIR / "v1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Updated v1_summary.json")


def update_split(seed: int = 42) -> None:
    import random
    split_path = METADATA_DIR / "v1_validation_split.json"

    existing_split: dict = {}
    if split_path.exists():
        with open(split_path) as f:
            existing_split = json.load(f)

    rng = random.Random(seed)
    new_split: dict = {}

    for path in sorted(DATA_DIR.glob("*_consensus_pairs.jsonl")):
        concept = path.stem.replace("_consensus_pairs", "")
        records = load_jsonl(path)
        all_pids = sorted({r["pair_id"] for r in records})

        existing = existing_split.get(concept, {})
        already_train = set(existing.get("train", []))
        already_val = set(existing.get("validation", []))
        already_assigned = already_train | already_val

        new_pids = [p for p in all_pids if p not in already_assigned]
        rng.shuffle(new_pids)
        cut = max(1, int(len(new_pids) * 0.8))

        new_split[concept] = {
            "train": sorted(already_train | set(new_pids[:cut])),
            "validation": sorted(already_val | set(new_pids[cut:])),
        }

    with open(split_path, "w") as f:
        json.dump(new_split, f, indent=2)
    log.info("Updated v1_validation_split.json")

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def show_status() -> None:
    summary_path = METADATA_DIR / "v1_summary.json"
    if not summary_path.exists():
        print("No v1_summary.json found.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\n{'Concept':<20} {'pair_ids':>9} {'records':>9} {'models':>7}")
    print("-" * 50)
    for concept, v in sorted(summary.items(), key=lambda x: -x[1]["pair_ids"]):
        flag = "" if v["pair_ids"] >= 100 else "  ← sparse"
        print(f"  {concept:<18} {v['pair_ids']:>9} {v['records']:>9} {v['models']:>7}{flag}")
    print()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate missing Rosetta concept pairs via FuelIX")
    p.add_argument("--concept", nargs="+", metavar="CONCEPT")
    p.add_argument("--all-sparse", action="store_true",
                   help=f"All sparse concepts: {', '.join(SPARSE_CONCEPTS)}")
    p.add_argument("--status", action="store_true")
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--models", nargs="+", default=None,
                   help="Override model list (default: all 14)")
    p.add_argument("--skip-models", nargs="+", default=None, metavar="MODEL",
                   help="Exclude specific models (e.g. if one is known broken)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--update-metadata-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.status:
        show_status()
        return

    if args.update_metadata_only:
        update_summary()
        update_split()
        return

    concepts = []
    if args.all_sparse:
        concepts = SPARSE_CONCEPTS
    elif args.concept:
        for c in args.concept:
            if c not in CONCEPT_DEFS:
                print(f"Unknown concept '{c}'. Available: {', '.join(CONCEPT_DEFS)}")
                sys.exit(1)
        concepts = args.concept
    else:
        print("Specify --concept, --all-sparse, --status, or --update-metadata-only.")
        sys.exit(1)

    if not args.dry_run and not FUELIX_API_KEY:
        print("FUELIX_API_KEY not found. Set in env or ~/Source/fuelix_kilocode_profiles/.env")
        sys.exit(1)

    models = args.models or MODELS
    if args.skip_models:
        skipping = [m for m in args.skip_models if m in models]
        models = [m for m in models if m not in args.skip_models]
        log.info("Skipping models: %s", ", ".join(skipping))
    log.info("Using %d models: %s ...", len(models), ", ".join(models[:4]))

    for concept in concepts:
        run_concept(concept, args.target, models, dry_run=args.dry_run)

    if not args.dry_run:
        update_summary()
        update_split()
        log.info("All done. Commit Rosetta_Concept_Pairs when satisfied.")


if __name__ == "__main__":
    main()
