"""
generate_prompts.py

Generates the text prompts we will later synthesize into audio (via TTS).

Design goal: NOT hyper-specific to a single guessed use case (e.g. "operator
on a factory floor"). Instead we build a general-but-relevant dataset for a
"domain-adaptable, noise-robust speech-to-text" system, with manufacturing /
data-logging vocabulary as one strong example domain (numbers, measurements,
technical terms) alongside general dictation-style sentences.

Run:
    python generate_prompts.py --output prompts.json --n_general 150 --n_domain 200
"""

import argparse
import json
import random

# ---------------------------------------------------------------------------
# 1. General dictation-style sentences (accent / phonetic diversity, no
#    domain assumptions baked in). Keeps the model generalizable.
# ---------------------------------------------------------------------------
GENERAL_SENTENCES = [
    "please schedule the meeting for tomorrow morning",
    "send me the updated report by end of day",
    "the quarterly numbers look better than expected",
    "can you confirm the delivery address for this order",
    "the system will restart automatically after the update",
    "let us review the checklist before we proceed",
    "the new employee starts on monday next week",
    "please share the invoice with the accounts team",
    "the temperature outside is quite high today",
    "we need to finalize the budget before friday",
    "the client requested a follow up call next week",
    "make sure the backup is completed before shutting down",
    "the training session has been moved to the afternoon",
    "please update the spreadsheet with the latest figures",
    "the shipment is expected to arrive within three days",
    "our team completed the project ahead of schedule",
    "the server logs show no errors this morning",
    "kindly forward this email to the concerned department",
    "the audit report has been submitted for review",
    "we should increase the frequency of quality checks",
]

# ---------------------------------------------------------------------------
# 2. Manufacturing / industrial data-logging vocabulary.
#    Grounded in real sand-casting / foundry terminology (moisture,
#    compactability, mold hardness, permeability, GCS, rejection rate) since
#    that maps to MPM Infosoft's actual product domain — but written as
#    generic "spoken measurement/observation" templates so the SAME model
#    architecture and pipeline would work for ANY numeric/technical
#    voice-logging scenario, not just this one guessed deployment.
# ---------------------------------------------------------------------------
# Each term paired with physically plausible units (avoids nonsense
# combinations like "grain fineness number in degrees celsius").
DOMAIN_TERMS_UNITS = {
    "sand moisture": ["percent"],
    "green compactability": ["percent", "points"],
    "mold hardness": ["points"],
    "green compressive strength": ["kg per square centimeter"],
    "permeability number": ["points", ""],
    "sand rejection rate": ["percent"],
    "core hardness": ["points"],
    "wet tensile strength": ["kg per square centimeter"],
    "loss on ignition": ["percent"],
    "active clay content": ["percent"],
    "dead clay content": ["percent"],
    "flowability index": ["percent", "points"],
    "bentonite percentage": ["percent"],
    "shatter index": ["percent"],
    "sand temperature": ["degrees celsius"],
    "muller efficiency": ["percent"],
    "grain fineness number": ["points", ""],
}
DOMAIN_TERMS = list(DOMAIN_TERMS_UNITS.keys())

STATIONS = ["station one", "station two", "station three", "line four", "mixer two", "muller one"]

NUMBERS = [str(n) for n in range(1, 100)] + [f"{n} point {d}" for n in range(1, 20) for d in range(1, 9)]

DOMAIN_TEMPLATES = [
    "{term} reading is {number} {unit}",
    "{term} at {station} is {number} {unit}",
    "recording {term} as {number} {unit} on {station}",
    "{term} for this batch is {number} {unit}",
    "please note {term} value {number} {unit}",
    "{station} shows {term} of {number} {unit}",
    "the {term} has dropped to {number} {unit}",
    "increase {term} to {number} {unit} on {station}",
]

# Generic voice-logging phrases usable in ANY inspection/observation context
# (retail stock counts, lab readings, warehouse checks, etc.) — proof the
# pipeline generalizes beyond one guessed factory scenario.
GENERIC_OBSERVATION_TEMPLATES = [
    "batch number {batch_num} passed inspection",
    "batch number {batch_num} flagged for rework",
    "reading {number} recorded at {station}",
    "shift report for {station} completed",
    "no anomalies detected at {station}",
    "operator confirms reading of {number} at {station}",
]

BATCH_NUMBERS = [str(n) for n in range(100, 999)]  # batch numbers are whole numbers


def build_domain_prompt():
    template = random.choice(DOMAIN_TEMPLATES)
    term = random.choice(DOMAIN_TERMS)
    unit = random.choice(DOMAIN_TERMS_UNITS[term])
    return template.format(
        term=term,
        number=random.choice(NUMBERS),
        unit=unit,
        station=random.choice(STATIONS),
    ).replace("  ", " ").strip()


def build_generic_observation_prompt():
    template = random.choice(GENERIC_OBSERVATION_TEMPLATES)
    return template.format(
        number=random.choice(NUMBERS),
        batch_num=random.choice(BATCH_NUMBERS),
        station=random.choice(STATIONS),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="prompts.json")
    parser.add_argument("--n_general", type=int, default=150)
    parser.add_argument("--n_domain", type=int, default=200)
    parser.add_argument("--n_generic_obs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    prompts = []

    # general sentences (repeat/sample if we need more than the fixed list)
    for i in range(args.n_general):
        prompts.append({
            "id": f"general_{i:04d}",
            "text": random.choice(GENERAL_SENTENCES),
            "category": "general",
        })

    for i in range(args.n_domain):
        prompts.append({
            "id": f"domain_{i:04d}",
            "text": build_domain_prompt(),
            "category": "manufacturing_domain",
        })

    for i in range(args.n_generic_obs):
        prompts.append({
            "id": f"obs_{i:04d}",
            "text": build_generic_observation_prompt(),
            "category": "generic_observation",
        })

    random.shuffle(prompts)

    with open(args.output, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Generated {len(prompts)} prompts -> {args.output}")
    print(f"  general: {args.n_general}, domain: {args.n_domain}, generic_obs: {args.n_generic_obs}")


if __name__ == "__main__":
    main()
