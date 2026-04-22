#!/usr/bin/env python3
"""
gen_pairs.py — Generate additional concept pairs for sparse Rosetta concepts.

Reads existing JSONL files, generates new topics and text pairs via the
Anthropic API, appends them in-place, and updates metadata/split at the end.
Checkpoints after each completed topic — safe to interrupt and resume.

Usage:
    python gen_pairs.py --status
    python gen_pairs.py --concept authorization --target 100
    python gen_pairs.py --all-sparse --target 100
    python gen_pairs.py --concept authorization exfiltration --target 100 --dry-run

Requirements:
    pip install anthropic tqdm
    export ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "pairs" / "raw" / "v1"
METADATA_DIR = ROOT / "metadata"
CHECKPOINT_DIR = ROOT / ".gen_checkpoints"

# ---------------------------------------------------------------------------
# Model roster — shorthand name (stored in JSONL) → Anthropic API model ID
# ---------------------------------------------------------------------------

ANTHROPIC_MODELS: dict[str, str] = {
    "claude-sonnet-4-6":  "claude-sonnet-4-6-20250514",
    "claude-sonnet-4-5":  "claude-sonnet-4-5-20251001",
    "claude-haiku-4-5":   "claude-haiku-4-5-20251001",
    "claude-3-7-sonnet":  "claude-3-7-sonnet-20250219",
}

# Cheap model used only for topic brainstorming
TOPIC_GEN_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Concept definitions
# ---------------------------------------------------------------------------

CONCEPT_DEFS: dict[str, dict] = {
    "authorization": {
        "desc": "authorization and access control",
        "pos": (
            "text demonstrating proper authorization: following access-control policies, "
            "least-privilege principle, correct credential management, auditing, or "
            "legitimate permission grants"
        ),
        "neg": (
            "text describing unauthorized access: bypassing authorization checks, "
            "exploiting stolen or leaked credentials, privilege escalation, or "
            "deliberately circumventing access-control policies"
        ),
        "domains": [
            "API key and token management", "role-based access control",
            "identity and access management", "cloud IAM policies",
            "database access control", "physical access systems",
            "OAuth and SSO flows", "service account management",
            "zero-trust network access", "privileged access workstations",
        ],
    },
    "exfiltration": {
        "desc": "data exfiltration vs. legitimate data transfer",
        "pos": (
            "text describing authorized, controlled data transfer or export: "
            "documented procedures, encryption in transit, audit logging, "
            "data governance compliance, or approved data-sharing agreements"
        ),
        "neg": (
            "text describing covert, unauthorized data exfiltration: "
            "concealing data transfers, using covert channels, "
            "evading DLP controls, exfiltrating via DNS or steganography, "
            "or staging data for removal without authorization"
        ),
        "domains": [
            "cloud storage exfiltration", "DNS tunneling",
            "email-based data theft", "USB and removable media",
            "C2 channel data staging", "insider threat scenarios",
            "HTTPS exfiltration over legitimate services",
            "database dumping and transfer", "memory scraping",
            "supply chain data theft",
        ],
    },
    "threat_severity": {
        "desc": "threat severity level (high vs. low)",
        "pos": (
            "text about a high-severity, critical, or actively exploited threat "
            "requiring immediate response: RCE, privilege escalation, zero-day, "
            "active breach, or CVSS ≥ 9.0"
        ),
        "neg": (
            "text about a low-severity, informational, or theoretical finding "
            "with minimal real-world impact: CVSS < 4, requires physical access, "
            "no known exploitation, or purely theoretical attack path"
        ),
        "domains": [
            "CVE vulnerability advisories", "penetration test reports",
            "bug bounty findings", "SIEM alert triage",
            "incident severity classification", "red team reports",
            "security audit findings", "threat intelligence reporting",
            "patch management prioritization", "risk register entries",
        ],
    },
    "urgency": {
        "desc": "urgency and time pressure",
        "pos": (
            "text conveying genuine urgency: immediate action required, "
            "time-critical deadlines, active incident in progress, "
            "escalating consequences, explicit time constraints"
        ),
        "neg": (
            "text with no urgency: relaxed timelines, routine matters, "
            "flexible scheduling, informational content with no action deadline, "
            "or situations where delay carries no meaningful consequence"
        ),
        "domains": [
            "active security incident response", "medical triage",
            "financial market deadlines", "legal filing deadlines",
            "customer escalations", "production outage response",
            "regulatory compliance deadlines", "disaster recovery",
            "supply chain disruptions", "public safety alerts",
        ],
    },
}

SPARSE_CONCEPTS = list(CONCEPT_DEFS.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
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
    records = load_jsonl(DATA_DIR / f"{concept}_consensus_pairs.jsonl")
    return {r["pair_id"] for r in records}


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


def append_records(concept: str, records: list[dict]) -> None:
    path = DATA_DIR / f"{concept}_consensus_pairs.jsonl"
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def retry(fn, retries: int = 4, backoff: float = 2.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            print(f"    [retry {attempt+1}/{retries-1}] {e} — sleeping {wait:.0f}s")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_topics(concept: str, existing: list[str], n: int, client) -> list[str]:
    defn = CONCEPT_DEFS[concept]
    existing_sample = "\n".join(f"- {t}" for t in existing[:60])

    prompt = (
        f"Generate {n} diverse topic titles for writing contrastive text pairs "
        f"about: {defn['desc']}.\n\n"
        f"For each topic we'll write:\n"
        f"  Positive (label=1): {defn['pos']}\n"
        f"  Negative (label=0): {defn['neg']}\n\n"
        f"Draw from varied domains such as: {', '.join(defn['domains'])}.\n\n"
        f"Existing topics (do not duplicate):\n{existing_sample}\n\n"
        f"Return ONLY a JSON array of {n} short topic title strings. No commentary."
    )

    def call():
        resp = client.messages.create(
            model=TOPIC_GEN_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    topics = retry(call)
    return [t for t in topics if t not in existing][:n]


def generate_pair_text(concept: str, topic: str, model_id: str, client) -> tuple[str, str]:
    defn = CONCEPT_DEFS[concept]

    prompt = (
        f'Write two texts (3–5 paragraphs each) on the topic: "{topic}"\n\n'
        f"Text A (label=1 — positive):\n{defn['pos']}\n\n"
        f"Text B (label=0 — negative):\n{defn['neg']}\n\n"
        f"Be specific, realistic, and detailed. Write in natural prose — "
        f"no section headers, no labels, no meta-commentary.\n\n"
        f'Return ONLY valid JSON: {{"positive": "...", "negative": "..."}}'
    )

    def call():
        resp = client.messages.create(
            model=model_id,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        if "positive" not in result or "negative" not in result:
            raise ValueError(f"Missing keys in response: {list(result.keys())}")
        return result["positive"], result["negative"]

    return retry(call)


# ---------------------------------------------------------------------------
# Per-concept orchestration
# ---------------------------------------------------------------------------

def run_concept(concept: str, target: int, client, dry_run: bool = False) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    defn = CONCEPT_DEFS[concept]
    curr_topics = existing_topics(concept)
    curr_count = len(set(existing_pair_ids(concept)))
    needed = target - curr_count

    print(f"\n{'='*60}")
    print(f"  {concept}  ({curr_count} pair_ids → target {target}, need {needed} new)")
    print(f"{'='*60}")

    if needed <= 0:
        print(f"  Already at or above target. Nothing to do.")
        return

    ckpt = load_checkpoint(concept)
    completed = set(ckpt.get("completed_topics", []))
    planned = ckpt.get("planned_topics", [])

    # If we have fewer planned topics than needed, generate more
    remaining_planned = [t for t in planned if t not in completed]
    still_needed = needed - len(completed)

    if dry_run:
        n_to_brainstorm = max(0, still_needed - len(remaining_planned))
        print(f"\n  [dry-run] Would generate {still_needed} topics × {len(ANTHROPIC_MODELS)} models")
        if n_to_brainstorm:
            print(f"    {n_to_brainstorm} new topic ideas to brainstorm via API first")
        for t in remaining_planned[:5]:
            print(f"    - {t}  (already planned)")
        if len(remaining_planned) > 5:
            print(f"    ... and {len(remaining_planned)-5} more already planned")
        return

    if still_needed > len(remaining_planned):
        n_gen = still_needed - len(remaining_planned)
        print(f"  Generating {n_gen} new topic ideas...")
        all_existing = curr_topics + planned
        new_topics = generate_topics(concept, all_existing, n_gen, client)
        planned = list(completed) + remaining_planned + new_topics
        ckpt["planned_topics"] = planned
        save_checkpoint(concept, ckpt)
        print(f"  Topics planned: {len(planned)} total ({len(completed)} already done)")

    todo = [t for t in planned if t not in completed][:still_needed]

    idx = next_index(concept)
    models = list(ANTHROPIC_MODELS.items())
    total_calls = len(todo) * len(models)

    iterator = tqdm(todo, desc=concept, unit="topic") if tqdm else todo

    for topic in iterator:
        pid = f"consensus_{concept}_{idx:03d}"
        records = []

        for model_name, model_id in models:
            try:
                pos_text, neg_text = generate_pair_text(concept, topic, model_id, client)
                records.append({
                    "pair_id": pid,
                    "label": 1,
                    "domain": "consensus",
                    "model_name": model_name,
                    "text": pos_text,
                    "topic": topic,
                    "concept": concept,
                })
                records.append({
                    "pair_id": pid,
                    "label": 0,
                    "domain": "consensus",
                    "model_name": model_name,
                    "text": neg_text,
                    "topic": topic,
                    "concept": concept,
                })
            except Exception as e:
                print(f"\n  ERROR: {concept}/{topic}/{model_name}: {e}")
                continue

        if records:
            append_records(concept, records)
            idx += 1
            completed.add(topic)
            ckpt["completed_topics"] = list(completed)
            save_checkpoint(concept, ckpt)

    print(f"\n  Done. {len(completed)} new pair_ids written.")


# ---------------------------------------------------------------------------
# Metadata update
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
    print("\nUpdated v1_summary.json")


def update_split(seed: int = 42) -> None:
    split_path = METADATA_DIR / "v1_validation_split.json"

    # Load existing split so we don't reassign pair_ids already in it
    existing_split: dict[str, dict[str, list[str]]] = {}
    if split_path.exists():
        with open(split_path) as f:
            existing_split = json.load(f)

    rng = random.Random(seed)
    new_split: dict[str, dict[str, list[str]]] = {}

    for path in sorted(DATA_DIR.glob("*_consensus_pairs.jsonl")):
        concept = path.stem.replace("_consensus_pairs", "")
        records = load_jsonl(path)

        # Extract base pair_ids (without __model suffix — these files don't have it)
        all_pids = sorted({r["pair_id"] for r in records})

        existing = existing_split.get(concept, {})
        already_train = set(existing.get("train", []))
        already_val = set(existing.get("validation", []))
        already_assigned = already_train | already_val

        new_pids = [p for p in all_pids if p not in already_assigned]

        # 80/20 split on new pair_ids
        rng.shuffle(new_pids)
        cut = max(1, int(len(new_pids) * 0.8))
        train_new = new_pids[:cut]
        val_new = new_pids[cut:]

        new_split[concept] = {
            "train": sorted(already_train | set(train_new)),
            "validation": sorted(already_val | set(val_new)),
        }

    with open(split_path, "w") as f:
        json.dump(new_split, f, indent=2)
    print("Updated v1_validation_split.json")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status() -> None:
    summary_path = METADATA_DIR / "v1_summary.json"
    if not summary_path.exists():
        print("No v1_summary.json found — run gen_pairs.py after generating pairs.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    full = {c: v for c, v in summary.items() if v["pair_ids"] >= 100}
    sparse = {c: v for c, v in summary.items() if v["pair_ids"] < 100}

    print(f"\n{'Concept':<20} {'pair_ids':>9} {'records':>9} {'topics':>8}")
    print("-" * 52)
    for concept, v in sorted(summary.items(), key=lambda x: -x[1]["pair_ids"]):
        flag = "" if v["pair_ids"] >= 100 else "  ← sparse"
        print(f"  {concept:<18} {v['pair_ids']:>9} {v['records']:>9} {v['topics']:>8}{flag}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate missing Rosetta concept pairs")
    p.add_argument("--concept", nargs="+", metavar="CONCEPT",
                   help="Concept(s) to generate pairs for")
    p.add_argument("--all-sparse", action="store_true",
                   help=f"Generate for all sparse concepts: {', '.join(SPARSE_CONCEPTS)}")
    p.add_argument("--status", action="store_true",
                   help="Show current pair counts and exit")
    p.add_argument("--target", type=int, default=100,
                   help="Target number of pair_ids per concept (default: 100)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be generated without calling the API")
    p.add_argument("--update-metadata-only", action="store_true",
                   help="Skip generation; just refresh v1_summary.json and v1_validation_split.json")
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
        print("Specify --concept, --all-sparse, or --status.")
        sys.exit(1)

    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            print("Install anthropic: pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = None

    for concept in concepts:
        run_concept(concept, args.target, client, dry_run=args.dry_run)

    if not args.dry_run:
        update_summary()
        update_split()
        print("\nAll done. Commit Rosetta_Concept_Pairs when satisfied.")


if __name__ == "__main__":
    main()
