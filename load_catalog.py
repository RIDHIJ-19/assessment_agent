import json


with open("dataset.json", encoding="utf-8") as f:
    catalog = json.load(f)


# -------------------------------------------------
# SHL test-type taxonomy
# -------------------------------------------------
# The scraped "keys" field on each catalog entry is actually the broad
# SHL test-type category (e.g. "Personality & Behavior"), not a specific
# skill like "Java". We map those categories to the standard SHL
# single-letter test_type codes here, once, at load time.

# Lenient substring matching instead of exact-string lookup: as long as
# a key CONTAINS one of these hint words, it counts. Catches spelling
# variants ("Judgment" vs "Judgement"), extra wording, different
# punctuation, plural/singular, etc. without needing every variant listed.

CODE_KEYWORDS = {
    "A": ["ability", "aptitude"],
    "B": ["biodata", "situational"],
    "C": ["competenc"],
    "D": ["development", "360"],
    "E": ["exercise"],
    "K": ["knowledge", "skill"],
    "P": ["personality", "behavior", "behaviour"],
    "S": ["simulation"],
}


def map_test_types(keys):
    codes = []

    for key in keys:
        key_lower = key.lower()

        for code, hints in CODE_KEYWORDS.items():
            if code in codes:
                continue
            if any(hint in key_lower for hint in hints):
                codes.append(code)

    return codes


assessments = []

for item in catalog:

    keys = item.get("keys", [])

    assessment = {
        "id": item.get("entity_id"),
        "name": item.get("name", ""),
        "url": item.get("link", ""),
        "job_levels": item.get("job_levels", []),
        "languages": item.get("languages", []),
        "duration": item.get("duration", ""),
        "status": item.get("status", ""),
        "remote": item.get("remote", ""),
        "adaptive": item.get("adaptive", ""),
        "description": item.get("description", ""),
        "keys": keys,
        "test_types": map_test_types(keys)
    }

    assessments.append(assessment)


print(f"Loaded {len(assessments)} assessments")


def build_catalog_features(assessments):

    features = {
        "job_levels": set(),
        "languages": set(),
        "keys": set(),
        "keywords": set(),
        "durations": set(),
        "remote": set(),
        "adaptive": set()
    }


    for assessment in assessments:

        for level in assessment.get("job_levels", []):
            features["job_levels"].add(level)


        for lang in assessment.get("languages", []):
            features["languages"].add(lang)


        for key in assessment.get("keys", []):
            features["keys"].add(key)


        for keyword in assessment.get("keywords", []):
            features["keywords"].add(keyword)


        if assessment.get("duration"):
            features["durations"].add(
                assessment["duration"]
            )


        if assessment.get("remote"):
            features["remote"].add(
                assessment["remote"]
            )


        if assessment.get("adaptive"):
            features["adaptive"].add(
                assessment["adaptive"]
            )


    return {
        key: list(value)
        for key, value in features.items()
    }

import re
from collections import Counter


def extract_behavioral_taxonomy():

    behavioral_types = {
        "P",  # Personality & Behavior
        "C",  # Competency
        "B",  # Biodata / Situational Judgment
        "D"   # Development / 360
    }


    unique_keys = set()
    term_counter = Counter()


    for a in assessments:

        test_types = set(
            a.get("test_types", [])
        )

        if test_types & behavioral_types:

            # Collect SHL category labels
            for key in a.get("keys", []):
                unique_keys.add(key)


            # Extract words from name + description
            text = (
                a.get("name", "")
                + " "
                + a.get("description", "")
            ).lower()


            words = re.findall(
                r"\b[a-z]{5,}\b",
                text
            )


            for word in words:
                term_counter[word] += 1


    with open(
        "behavioral_taxonomy.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("===== UNIQUE BEHAVIORAL TEST CATEGORIES =====\n\n")

        for key in sorted(unique_keys):
            f.write(f"- {key}\n")


        f.write(
            "\n\n===== MOST COMMON BEHAVIORAL TERMS =====\n\n"
        )


        for term, count in term_counter.most_common(100):
            f.write(
                f"{term}: {count}\n"
            )


    print(
        "Created behavioral_taxonomy.txt"
    )


def print_behavioral_catalog():
    extract_behavioral_taxonomy()


if __name__ == "__main__":
    print_behavioral_catalog()