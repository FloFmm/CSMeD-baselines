import json
import os
import csv
import numpy as np
from ranx import Qrels, Run, compare


def n_precision_at_recall(
    run: dict[str, dict[str, float]],
    qrels_dict: dict[str, dict[str, int]],
    total_docs: int,
    recall_level: float = 0.95,
) -> float:
    """calculates normalised Precision achieved at a given Recall level.
    Recall level is achieved when a greater or equal number of docs obtains that recall.
        "normalisedPrecision": r"nPrecision@r\% &= \frac{TP \cdot TN}{\mathcal{E} \cdot (TP + FP)} ",

    """
    n_precision_values = []

    for query_id, rankings in run.items():
        tp, fp, tn, fn = 0, 0, 0, 0

        relevant_docs = sum(qrels_dict[query_id].values())
        non_relevant_docs = total_docs - relevant_docs

        for rank, (doc_id, _) in enumerate(rankings.items()):
            if qrels_dict[query_id].get(doc_id, 0) == 1:  # Document is relevant
                tp += 1
            else:
                fp += 1

            recall = tp / (tp + fn + relevant_docs - tp)

            if recall >= recall_level:
                tn = non_relevant_docs - fp
                fn = relevant_docs - tp
                # E = FP + TN
                # n_precision = (TP * TN) / (E * (TP + FP)) if (E * (TP + FP)) != 0 else 0
                n_precision = (tp * tn) / ((fp + tn) * (tp + fp))
                n_precision_values.append(n_precision)
                break

    return (
        sum(n_precision_values) / len(n_precision_values) if n_precision_values else 0
    )


def tnr_at_recall(
    run: dict[str, dict[str, float]],
    qrels_dict: dict[str, dict[str, int]],
    total_docs: int,
    recall_level: float = 0.95,
) -> float:
    """calculates the True Negative Rate achieved at a given Recall level.
    Recall level is achieved when a greater or equal number of docs obtains that recall.
    """
    tnr_values = []

    for query_id, rankings in run.items():
        tp, fp, tn, fn = 0, 0, 0, 0

        relevant_docs = sum(qrels_dict[query_id].values())
        non_relevant_docs = total_docs - relevant_docs

        for rank, (doc_id, _) in enumerate(rankings.items()):
            if qrels_dict[query_id].get(doc_id, 0) == 1:  # Document is relevant
                tp += 1
            else:
                fp += 1

            recall = tp / (tp + fn + relevant_docs - tp)

            if recall >= recall_level:
                tn = non_relevant_docs - fp
                fn = relevant_docs - tp
                tnr = tn / (tn + fp)
                tnr_values.append(tnr)
                break

    return sum(tnr_values) / len(tnr_values) if tnr_values else 0


def find_last_relevant(
    run: dict[str, dict[str, float]], qrels_dict: dict[str, dict[str, int]]
) -> float:
    """Find percentage of the run where the last relevant item was found."""
    percentages = []

    for query, docs in run.items():
        # Check if the query exists in the qrels_dict
        if query in qrels_dict:
            # List of relevant document ids for the given query
            relevant_docs = [
                doc_id for doc_id, rel in qrels_dict[query].items() if rel > 0
            ]

            # Find the position of the last relevant document
            last_relevant_position = None
            for doc_id, _ in reversed(docs.items()):
                if doc_id in relevant_docs:
                    last_relevant_position = (
                        list(docs.keys()).index(doc_id) + 1
                    )  # Adding 1 as indexing starts from 0
                    break

            # If a relevant document is found in the run
            if last_relevant_position is not None:
                percentages.append(last_relevant_position / len(docs) * 100)

    # Return the average of the percentages
    return sum(percentages) / len(percentages) if percentages else 0.0


def load_run_dict_for_model(
    model_name: str,
    query_type: str,
    total_docs: int,
    base_path: str
):
    """
    Loads all NPZ ranking files for a given model + query_type + total_docs.

    Returns:
        run_dict: dict[review_name -> dict[doc_id -> score]]
    """

    # Build: base_path/model_name/query_type/docs={total_docs}/
    model_dir = os.path.join(
        base_path, model_name, query_type, f"docs={total_docs}"
    )

    if not os.path.isdir(model_dir):
        raise ValueError(f"Directory not found: {model_dir}")

    run_dict = {}

    # Iterate files inside docs={total_docs}
    for fname in os.listdir(model_dir):
        if not fname.endswith(".npz"):
            continue

        full_path = os.path.join(model_dir, fname)

        # Remove extension → review_name
        review_name = os.path.splitext(fname)[0]

        # Load stored arrays
        data = np.load(full_path)
        ids = data["ids"]
        scores = data["scores"]

        # Convert to dict: {doc_id: score}
        run_dict[review_name] = {
            doc_id: float(score)
            for doc_id, score in zip(ids, scores)
        }

    return run_dict

def evaluate_model(
    model_name: str,
    query_type: str,
    total_docs: int,
    qrels_dict: dict[str, dict[str, int]],
    output_dir: str = "reports/title_and_abstract",
    rankings_base_path: str = "../systematic-review-datasets/data/rankings",
):
    """
    Evaluate a single model using only its NPZ ranking files.
    Produces one JSON output file with metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load qrels only once
    qrels = Qrels(qrels_dict)

    # Load ALL queries for this model
    run_dict = load_run_dict_for_model(model_name, query_type, total_docs, rankings_base_path)

    # Convert to ranx Run
    run = Run(run_dict, name=model_name)

    # Standard ranx metrics
    report = compare(
        qrels=qrels,
        runs=[run],
        metrics=[
            "ndcg", "ndcg@5", "ndcg@10", "ndcg@100",
            "map", "map@10", "map@100",
            "recall", "recall@10", "recall@50", "recall@100", "recall@1000",
            "precision", "precision@10", "precision@50", "precision@100", "precision@1000",
            "f1", "f1@10", "f1@50", "f1@100",
            "r-precision",
            "mrr@100",
        ],
    )

    report_dict = report.to_dict()
    metrics = report_dict[model_name]["scores"]

    # --- Custom metrics (adapted versions still work on run_dict) ---
    metrics["dataset_size"] = total_docs
    metrics["last_rel"] = find_last_relevant(run_dict, qrels_dict)
    metrics["tnr@95"] = tnr_at_recall(run_dict, qrels_dict, total_docs, recall_level=0.95)
    metrics["tnr@90"] = tnr_at_recall(run_dict, qrels_dict, total_docs, recall_level=0.90)
    metrics["n_precision@95"] = n_precision_at_recall(run_dict, qrels_dict, total_docs, recall_level=0.95)
    metrics["n_precision@80"] = n_precision_at_recall(run_dict, qrels_dict, total_docs, recall_level=0.80)

    # --- Write JSON ---
    out_json = os.path.join(output_dir, f"{model_name}_{query_type}_docs={total_docs}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"✓ Finished {model_name}, wrote: {out_json}")
