import os
CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH # has to be up here

import os.path
import pickle
import random
from typing import Any

import numpy as np
import torch
from numba import njit
from retriv import SparseRetriever, DenseRetriever, Encoder
from retriv.dense_retriever.dense_retriever import compute_scores, compute_scores_multi
from typing import List, Dict, Any
import sys

# Add path to the parent repo (systematic-review-datasets)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "../systematic-review-datasets")))
from csmed.csmed.csmed_cochrane import CSMeDCochrane
from measures import evaluate_runs

# Constants
os.environ["RETRIV_BASE_PATH"] = "../systematic-review-datasets/data/indexes" # has to be down here
SEED = 42
USE_GPU = True
QUERY_TYPES =  ["title"]#, "abstract", "query", "criteria"]:

# Initialize seed for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

class DenseRetrieverWithQueryEncoder(DenseRetriever):
    def __init__(self, query_model=None, **kwargs):
        """
        Args:
            query_model: str or Encoder object for encoding queries.
            kwargs: passed to original DenseRetriever (document encoder)
        """
        super().__init__(**kwargs)

        if isinstance(query_model, str):
            # Wrap string path in Encoder like the doc encoder
            self.query_encoder = Encoder(
                index_name=self.index_name + "_query",
                model=query_model,
                normalize=self.normalize,
                max_length=self.max_length,
            )
        else:
            self.query_encoder = query_model

    def search(
        self,
        query: str,
        return_docs: bool = True,
        cutoff: int = 100,
    ) -> List:
        """Override search to use query_encoder instead of doc encoder for queries."""

        encoded_query = self.query_encoder(query)

        if self.use_ann:
            doc_ids, scores = self.ann_searcher.search(encoded_query, cutoff)
        else:
            if self.embeddings is None:
                self.load_embeddings()
            doc_ids, scores = compute_scores(encoded_query, self.embeddings, cutoff)

        doc_ids = self.map_internal_ids_to_original_ids(doc_ids)

        return (
            self.prepare_results(doc_ids, scores)
            if return_docs
            else dict(zip(doc_ids, scores))
        )


    def msearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
        batch_size: int = 32,
    ) -> Dict:
        """Override multi-search to use query_encoder."""

        q_ids = [x["id"] for x in queries]
        q_texts = [x["text"] for x in queries]
        encoded_queries = self.query_encoder(q_texts, batch_size, show_progress=False)

        if self.use_ann:
            doc_ids, scores = self.ann_searcher.msearch(encoded_queries, cutoff)
        else:
            if self.embeddings is None:
                self.load_embeddings()
            doc_ids, scores = compute_scores_multi(
                encoded_queries, self.embeddings, cutoff
            )

        doc_ids = [
            self.map_internal_ids_to_original_ids(_doc_ids) for _doc_ids in doc_ids
        ]

        results = {q: dict(zip(doc_ids[i], scores[i])) for i, q in enumerate(q_ids)}

        return {q_id: results[q_id] for q_id in q_ids}

@njit
def set_seed(value):
    np.random.seed(value)


def load_dataset():
    return CSMeDCochrane().load_dataset(
        # base_path="../../csmed/datasets/datasets/csmed_cochrane/"
        base_path="../systematic-review-datasets/csmed/csmed_cochrane"
    )


def create_retrievers(configs, collection):
    retrievers = {}
    for name, conf in configs.items():
        index_name = f"{name}_{conf['type']}_index_docs={len(collection)}"
        if conf["type"] == "sparse":
            if os.path.exists(f"{os.environ['RETRIV_BASE_PATH']}/collections/{index_name}"):
                print(f"Loading existing index: {index_name}")
                retrievers[name] = SparseRetriever.load(index_name)

            else:
                retrievers[name] = SparseRetriever(
                    index_name=index_name,
                    model=conf["model"],
                    min_df=1,
                    tokenizer="whitespace",
                    stemmer="english",
                    stopwords="english",
                    do_lowercasing=True,
                    do_ampersand_normalization=True,
                    do_special_chars_normalization=True,
                    do_acronyms_normalization=True,
                    do_punctuation_removal=True,
                )
                retrievers[name].index(collection)
        elif conf["type"] == "dense":
            if os.path.exists(f"{os.environ['RETRIV_BASE_PATH']}/collections/{index_name}"):
                print(f"Loading existing index: {index_name}")
                retrievers[name] = DenseRetriever.load(index_name)
            else:
                if "query_model" in conf:
                    retrievers[name] = DenseRetrieverWithQueryEncoder(
                        index_name=index_name,
                        model=conf["model"],
                        query_model=conf["query_model"],
                        normalize=True,
                        max_length=conf["max_length"],
                        use_ann=True,
                    )
                else:
                    retrievers[name] = DenseRetriever(
                        index_name=index_name,
                        model=conf["model"],
                        normalize=True,
                        max_length=conf["max_length"],
                        use_ann=True,
                    )
                retrievers[name].index(collection, use_gpu=USE_GPU, batch_size=128)
            
    return retrievers


def extract_review_details(review_data):
    review_details = {
        "title": review_data["dataset_details"]["title"],
        "abstract": review_data["dataset_details"]["abstract"],
        "criteria": " ".join(
            [f"{k}: {v}" for k, v in review_data["dataset_details"]["criteria"].items()]
        ),
    }

    # Handling the case where the search strategy might not be present
    try:
        review_details["query"] = list(
            review_data["dataset_details"]["search_strategy"].values()
        )[0]
    except IndexError:
        review_details["query"] = "no search query"

    return review_details


def prepare_data(review_data):
    # Extracting documents for indexing
    collection = [
        {"id": doc["document_id"], "text": doc["text"]}
        for doc in review_data["data"]["train"]
    ]

    # Extracting relevance judgments (qrels)
    qrels = {
        doc["document_id"]: int(doc["labels"][0])
        for doc in review_data["data"]["train"]
    }

    return collection, qrels


def build_global_corpus(dataset):
    """
    Build a global corpus containing all documents from all splits and all reviews,
    removing duplicates.
    """
    doc_dict = {}  # key: document_id, value: text

    for split, reviews in dataset.items():
        for review_name, review_data in reviews.items():
            for split_name in review_data["data"].keys():  # 'train', 'val', 'test' etc.
                for doc in review_data["data"][split_name]:
                    doc_id = doc["document_id"]
                    if doc_id not in doc_dict:
                        doc_dict[doc_id] = doc["text"]

    # Convert to list of dicts for retriever
    collection = [{"id": doc_id, "text": text} for doc_id, text in doc_dict.items()]
    # collection = collection[:1000]
    return collection


def initialise_runs(retriever_configs: dict[str, dict[str, str]]) -> dict[str, Any]:
    runs = {}
    for retriever_name in retriever_configs.keys():
        for query_type in QUERY_TYPES:
            run_key = f"{retriever_name}_{query_type}"
            runs[run_key] = {}
    return runs


def process_review(
    retrievers,
    review_data,
    runs: dict[str, dict[str, dict[str, float]]],
    cutoff: int,
    review_name: str,
):
    review_details = extract_review_details(review_data)

    for model_name, retriever in retrievers.items():
        for slr_protocol_key, slr_protocol_value in review_details.items():
            if slr_protocol_key not in QUERY_TYPES:
                continue
            run_key = f"{model_name}_{slr_protocol_key}"
            runs[run_key][review_name] = retriever.search(
                query=slr_protocol_value, cutoff=cutoff, return_docs=False
            )


if __name__ == "__main__":
    set_seed(SEED)

    dataset = load_dataset()
    
    eval_reviews = dataset["EVAL"]

    outfile_path = "reports/title_and_abstract/"
    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path)

    retriever_configs = {
        # "bm25": {
        #     "type": "sparse",
        #     "model": "bm25",
        # },
        # "tf-idf": {
        #     "type": "sparse",
        #     "model": "tf-idf",
        # },
        "MedCPT": {
            "type": "dense",
            "model": "ncbi/MedCPT-Article-Encoder",
            "query_model": "ncbi/MedCPT-Query-Encoder",
            "max_length": 256,
        },
        # "MiniLM-128": {
        #     "type": "dense",
        #     "model": "sentence-transformers/all-MiniLM-L6-v2",
        #     "max_length": 128,
        # },
        # "MiniLM-256": {
        #     "type": "dense",
        #     "model": "sentence-transformers/all-MiniLM-L6-v2",
        #     "max_length": 256,
        # },
        # "qa-MiniLM-512": {
        #     "type": "dense",
        #     "model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        #     "max_length": 512,
        # },
        # "mpnet": {
        #     "type": "dense",
        #     "model": "sentence-transformers/all-mpnet-base-v2",
        #     "max_length": 512,
        # },
        # "nli-mpnet": {
        #     "type": "dense",
        #     "model": "sentence-transformers/nli-mpnet-base-v2",
        #     "max_length": 512,
        # },
        # "biobert-nli": {
        #     "type": "dense",
        #     "model": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        #     "max_length": 512,
        # },
        # "S-BioBert": {
        #     "type": "dense",
        #     "model": "pritamdeka/S-BioBert-snli-multinli-stsb",
        #     "max_length": 512,
        # },
        # "pubmedbert": {
        #     "type": "dense",
        #     "model": "pritamdeka/S-PubMedBert-MS-MARCO",
        #     "max_length": 512,
        # },
        # "roberta": {
        #     "type": "dense",
        #     "model": "sentence-transformers/stsb-roberta-base-v2",
        #     "max_length": 512,
        # },
    }

    dataset = load_dataset()
    global_corpus = build_global_corpus(dataset)
    total_docs = len(global_corpus)
    retrievers = create_retrievers(retriever_configs, collection=global_corpus)
    
    for split, reviews in dataset.items():
        print(f"\n=== Split: {split} ===")
        print(f"Number of reviews: {len(reviews)}")

    eval_reviews = dataset["EVAL"]
    qrels_dict = {}
    
    runs = initialise_runs(retriever_configs)
    print(f"Processing {len(eval_reviews)} reviews with {len(runs)} runs")
    for index, (review_name, review_data) in enumerate(eval_reviews.items(), start=1):
        qrels = {
            doc["document_id"]: int(doc["labels"][0])
            for doc in review_data["data"]["train"]
        }

        qrels_dict[review_name] = qrels

        process_review(
            retrievers,
            review_data,
            runs,
            cutoff=5000,
            review_name=review_name,
        )
        if index % 10 == 0:
            evaluate_runs(runs=runs, qrels_dict=qrels_dict, total_docs=total_docs)

            with open(f"{outfile_path}/runs.pkl", "wb") as f:
                pickle.dump(runs, f)

    evaluate_runs(runs=runs, qrels_dict=qrels_dict, total_docs=total_docs)
    with open(f"{outfile_path}/runs.pkl", "wb") as f:
        pickle.dump(runs, f)

