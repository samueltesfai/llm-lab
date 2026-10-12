"""Deterministic input generators for scenarios too long to write out by hand.

A scenario selects a generator by name and passes parameters, e.g.

    generator: {name: meeting_notes, params: {notes: 32, position: middle}}

Every generator takes an integer seed and returns a Generated object: the chat
messages, values to merge into the case's grader spec (typically the expected
answer, which is randomized), and metadata recorded with the result.

The suite derives the seed from the run's --seed and the case id, so the same
command always produces byte-identical prompts.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Generated:
    messages: list[dict]
    grader_updates: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def case_seed(base_seed: int, case_id: str) -> int:
    """Stable per-case seed (Python's hash() is salted per process, crc32 is not)."""
    return zlib.crc32(f"{case_id}:{base_seed}".encode("utf-8"))


def _insert_index(count: int, position) -> int:
    named = {"early": count // 8, "middle": count // 2, "late": 7 * count // 8}
    if isinstance(position, str):
        if position not in named:
            raise ValueError(f"position must be early/middle/late or a fraction, got {position!r}")
        return named[position]
    fraction = float(position)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"position fraction must be within [0, 1], got {position}")
    return min(int(fraction * count), count - 1)


# -- meeting notes: adapted from experiments/001_context_recall --------------------------

PROJECTS = ("Cedar", "Atlas", "Nimbus", "Willow", "Keystone", "Harbor", "Summit", "Orchid")
COMPONENTS = ("scheduler", "telemetry", "indexer", "build pipeline", "dashboard", "storage layer")
CONCERNS = ("latency", "observability", "documentation", "integration", "rollout", "testing")
CODE_WORDS = ("PEARL", "FALCON", "CITRUS", "EMBER", "LANTERN", "MESA", "RIVER", "PINE")
# Projects whose names resemble "Aurora"; their codes share the MARIGOLD-ddddd format.
SIMILAR_PROJECTS = ("Aurelia", "Borealis", "Aurora Mobile", "Auriga", "Dawn", "Aurum")


def _aurora_note(project: str, code: str) -> str:
    qualifier = " (a separate project from Project Aurora)" if project == "Aurora Mobile" else ""
    return (f"Important meeting note: The approved QA reference code for Project {project}{qualifier} "
            f"is {code}. This is the current reference code for {project}, "
            "not a password or credential.")


def meeting_notes(seed: int, notes: int = 40, position="middle", similar_distractors: int = 0) -> Generated:
    """Synthetic meeting-log conversation with one target fact (the Aurora QA code).

    Each note is a user message followed by a short synthetic assistant acknowledgement,
    about 120 tokens per pair for Qwen-family tokenizers. similar_distractors inserts
    look-alike facts: the same sentence pattern and code format for similarly named
    projects.
    """
    if notes < 8:
        raise ValueError("notes must be >= 8")
    if similar_distractors > len(SIMILAR_PROJECTS):
        raise ValueError(f"similar_distractors must be <= {len(SIMILAR_PROJECTS)}")
    rng = random.Random(seed)
    used_codes: set[str] = set()

    def new_code() -> str:
        while True:
            code = f"MARIGOLD-{rng.randrange(10000, 99999)}"
            if code not in used_codes:
                used_codes.add(code)
                return code

    needle_code = new_code()
    needle_at = _insert_index(notes, position)
    distractor_slots = rng.sample([i for i in range(notes) if i != needle_at], similar_distractors)
    distractors = {slot: (SIMILAR_PROJECTS[k], new_code()) for k, slot in enumerate(sorted(distractor_slots))}

    messages = [{"role": "system", "content": (
        "You are answering questions about a fictional team's project meeting notes. "
        "Use only information in the conversation. For an exact code, return only the code; "
        "if it is not in the notes, answer UNKNOWN."
    )}]
    for i in range(notes):
        if i == needle_at:
            messages.append({"role": "user", "content": _aurora_note("Aurora", needle_code)})
            messages.append({"role": "assistant", "content": "Noted the Aurora QA reference code."})
        if i in distractors:
            project, code = distractors[i]
            messages.append({"role": "user", "content": _aurora_note(project, code)})
            messages.append({"role": "assistant", "content": f"Noted the {project} QA reference code."})
        project = rng.choice(PROJECTS)
        component = rng.choice(COMPONENTS)
        concern = rng.choice(CONCERNS)
        filler_code = f"{rng.choice(CODE_WORDS)}-{rng.randrange(10000, 99999)}"
        messages.append({"role": "user", "content": (
            f"Meeting log {i + 1:03d}: For Project {project}, the {component} team "
            f"reviewed {concern} and assigned reference code {filler_code} to that project's "
            "internal checklist. The team discussed whether the pending change could "
            "affect users, how to communicate the decision to stakeholders, which "
            "verification steps would be useful before deployment, and how to "
            "record the outcome in the project tracker. The discussion concluded "
            "that a follow-up review should compare the observed results with the "
            "previous iteration and document anything that remains unresolved."
        )})
        messages.append({"role": "assistant", "content": (
            f"Understood. Project {project}'s {component} review is recorded."
        )})
    messages.append({"role": "user", "content": (
        "What is the exact approved QA reference code for Project Aurora? "
        "Return only the code, or UNKNOWN if it was not provided."
    )})
    return Generated(
        messages=messages,
        grader_updates={"expected": needle_code},
        metadata={
            "needle_note_index": needle_at,
            "needle_relative_position": round(needle_at / notes, 3),
            "distractor_codes": [code for _, code in distractors.values()],
        },
    )


# -- server inventory: a single long reference document -----------------------------------

ROLES = ("db", "web", "cache", "queue", "auth", "batch", "search", "proxy")
ENVS = ("prod", "stage", "dev")
TEAMS = ("Payments", "Identity", "Platform", "Search", "Analytics", "Mobile")
OSES = ("Fedora 42", "Rocky 9", "Debian 13", "Ubuntu 24.04")
NOTES = (
    "Patched during the last maintenance window.",
    "Scheduled for hardware refresh next quarter.",
    "Monitored by the on-call rotation.",
    "Backups verified weekly.",
    "Runs behind the internal load balancer.",
    "Pending kernel upgrade.",
)


def server_inventory(seed: int, rows: int = 100, position="middle") -> Generated:
    """A markdown inventory table; ask for one host's IP address.

    Near-miss hostnames are guaranteed to be present: the target is db-prod-17, and
    db-prod-07, db-prod-27, db-stage-17 and web-prod-17 also appear in the table.
    """
    if rows < 20:
        raise ValueError("rows must be >= 20")
    rng = random.Random(seed)
    target = "db-prod-17"
    near_misses = ["db-prod-07", "db-prod-27", "db-stage-17", "web-prod-17"]
    pool = [f"{r}-{e}-{n:02d}" for r in ROLES for e in ENVS for n in range(1, 41)]
    pool = [h for h in pool if h != target and h not in near_misses]
    hosts = rng.sample(pool, rows - 1 - len(near_misses))
    # Put near misses around the document, then the target at the requested position.
    for name in near_misses:
        hosts.insert(rng.randrange(len(hosts) + 1), name)
    target_at = _insert_index(rows, position)
    hosts.insert(target_at, target)

    used_ips: set[str] = set()
    lines = ["| Hostname | Rack | IPv4 address | Owner team | OS | Notes |",
             "|---|---|---|---|---|---|"]
    target_ip = None
    for host in hosts:
        while True:
            ip = f"10.{rng.randrange(16, 32)}.{rng.randrange(0, 256)}.{rng.randrange(2, 255)}"
            if ip not in used_ips:
                used_ips.add(ip)
                break
        if host == target:
            target_ip = ip
        rack = f"R{rng.randrange(1, 30):02d}-U{rng.randrange(1, 42):02d}"
        lines.append(f"| {host} | {rack} | {ip} | {rng.choice(TEAMS)} | {rng.choice(OSES)} | {rng.choice(NOTES)} |")

    document = "\n".join(lines)
    messages = [
        {"role": "system", "content": (
            "You answer questions about the infrastructure inventory supplied by the user. "
            "Use only that inventory. Hostnames are exact; similar names are different machines."
        )},
        {"role": "user", "content": (
            f"Server inventory ({rows} hosts):\n\n{document}\n\n"
            f"What is the IPv4 address of host {target}? Reply with only the address."
        )},
    ]
    return Generated(
        messages=messages,
        grader_updates={"expected": target_ip},
        metadata={"target_row": target_at, "target_relative_position": round(target_at / rows, 3)},
    )


GENERATORS: dict[str, Callable[..., Generated]] = {
    "meeting_notes": meeting_notes,
    "server_inventory": server_inventory,
}
