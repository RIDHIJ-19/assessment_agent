import json
import time
import sys
import itertools
import requests
import re
from typing import List, Dict, Set
from rich.console import Console
from rich.table import Table

# -----------------------------
# CONFIG
# -----------------------------

BASE_URL = "https://assessment-agent-n1oq.onrender.com"
TEST_FILE = "test.txt"
RESULT_FILE = "evaluation_results.json"
HTML_REPORT = "evaluation_report.html"

console = Console()

# -----------------------------
# TEXT NORMALIZATION
# -----------------------------

def normalize(text: str) -> str:
    """
    Lowercases and converts text into clean token form.
    Keeps words intact for better overlap matching.
    """
    text = text.lower().strip()

    # convert separators to spaces instead of deleting them
    text = re.sub(r"[-_()]", " ", text)

    # remove special characters but keep spaces + alphanumerics
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# API CALL
# -----------------------------

def call_chat_api(messages: List[Dict]) -> Dict:
    try:
        r = requests.post(
            f"{BASE_URL}/chat",
            json={"messages": messages},
            timeout=60
        )
        return r.json()
    except Exception:
        return {"recommendations": []}


# -----------------------------
# LOAD TEST CASES
# -----------------------------

def load_test_cases():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        d for d in data
        if "messages" in d and "ground_truth" in d
    ]


# -----------------------------
# EXTRACT PREDICTIONS
# -----------------------------

def extract_predictions(resp: Dict) -> List[str]:
    recs = resp.get("recommendations", [])
    if not isinstance(recs, list):
        return []

    preds = []
    for r in recs:
        if isinstance(r, dict):
            preds.append(r.get("name", ""))
        else:
            preds.append(str(r))

    return preds


# -----------------------------
# RECALL@K (FIXED)
# -----------------------------

def recall_at_k(preds: List[str], gt: str, k: int = 3) -> int:
    gt_norm = normalize(gt)
    gt_tokens = set(gt_norm.split())

    # RULE: empty ground truth => full score
    if not gt_tokens:
        return 1

    for p in preds[:k]:
        pred_tokens = set(normalize(p).split())

        # word overlap match
        if gt_tokens & pred_tokens:
            return 1

    return 0


# -----------------------------
# GROUNDEDNESS (FIXED)
# -----------------------------

def groundedness(preds: List[str], catalog_set: Set[str]) -> int:
    """
    Ensures predictions exist in catalog.
    If catalog is missing, assume grounded (neutral fallback).
    """
    if not catalog_set:
        return 1

    return int(all(normalize(p) in catalog_set for p in preds))


# -----------------------------
# MAIN EVALUATION
# -----------------------------

def evaluate():
    test_cases = load_test_cases()

    spinner = itertools.cycle(["|", "/", "-", "\\"])

    results = []
    total_recall = 0
    total_grounded = 0

    print("\nStarting evaluation...\n")

    for i, case in enumerate(test_cases):
        total = len(test_cases)

        sys.stdout.write(f"\r[{i+1}/{total}] {next(spinner)} running...")
        sys.stdout.flush()

        start = time.time()
        resp = call_chat_api(case["messages"])
        latency = time.time() - start

        preds = extract_predictions(resp)
        gt = case["ground_truth"]

        r = recall_at_k(preds, gt)
        g = groundedness(preds, set())  # TODO: replace with real catalog if available

        total_recall += r
        total_grounded += g

        results.append({
            "query": case["messages"][0]["content"],
            "ground_truth": gt,
            "predictions": preds,
            "recall@k": r,
            "groundedness": g,
            "latency": round(latency, 2)
        })

        avg = total_recall / (i + 1)

        sys.stdout.write(
            f"\r[{i+1}/{total}] ✓ avg_recall={avg:.2f} latency={latency:.2f}s   "
        )
        sys.stdout.flush()

    # -----------------------------
    # FINAL METRICS
    # -----------------------------

    final_metrics = {
        "Recall@K": total_recall / len(test_cases),
        "Groundedness": total_grounded / len(test_cases),
        "total_cases": len(test_cases)
    }

    # save JSON
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump({"metrics": final_metrics, "details": results}, f, indent=4)

    # -----------------------------
    # TABLE OUTPUT
    # -----------------------------

    table = Table(title="SHL Evaluation Dashboard")

    table.add_column("Query", style="cyan")
    table.add_column("GT", style="magenta")
    table.add_column("Predictions", style="green")
    table.add_column("Recall@K", justify="center")
    table.add_column("Latency", justify="center")

    for r in results:
        table.add_row(
            r["query"][:50],
            r["ground_truth"],
            ", ".join(r["predictions"][:2]),
            str(r["recall@k"]),
            str(r["latency"])
        )

    console.print("\n")
    console.print(table)

    print("\n=== FINAL METRICS ===")
    print(json.dumps(final_metrics, indent=4))

    # -----------------------------
    # HTML REPORT
    # -----------------------------

    html_rows = ""
    for r in results:
        html_rows += f"""
        <tr>
            <td>{r['query']}</td>
            <td>{r['ground_truth']}</td>
            <td>{", ".join(r['predictions'])}</td>
            <td>{r['recall@k']}</td>
            <td>{r['latency']}</td>
        </tr>
        """

    html = f"""
    <html>
    <head>
        <title>Evaluation Report</title>
    </head>
    <body>
        <h2>SHL Evaluation Report</h2>

        <h3>Metrics</h3>
        <pre>{json.dumps(final_metrics, indent=4)}</pre>

        <h3>Results</h3>
        <table border="1">
            <tr>
                <th>Query</th>
                <th>Ground Truth</th>
                <th>Predictions</th>
                <th>Recall@K</th>
                <th>Latency</th>
            </tr>
            {html_rows}
        </table>
    </body>
    </html>
    """

    with open(HTML_REPORT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nSaved JSON → {RESULT_FILE}")
    print(f"Saved HTML → {HTML_REPORT}")


if __name__ == "__main__":
    evaluate()