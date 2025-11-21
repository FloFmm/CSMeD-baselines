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
from typing import List, Dict, Any
import sys

# Add path to the parent repo (systematic-review-datasets)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "../systematic-review-datasets")))
from csmed.csmed.csmed_cochrane import CSMeDCochrane
from measures import evaluate_model
from modified_dense_retriever import ModifiedDenseRetriever
from retriv import SparseRetriever

# Constants
os.environ["RETRIV_BASE_PATH"] = "../systematic-review-datasets/data/indexes" # has to be down here
SEED = 42
USE_GPU = True
QUERY_TYPES =  ["title"]#, "abstract", "query", "criteria"]:

# Initialize seed for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)



@njit
def set_seed(value):
    np.random.seed(value)


def load_dataset():
    return CSMeDCochrane().load_dataset(
        # base_path="../../csmed/datasets/datasets/csmed_cochrane/"
        base_path="../systematic-review-datasets/csmed/csmed_cochrane"
    )


def create_retriever(name, conf, collection):
    index_name = f"{name}_{conf['type']}_index_docs={len(collection)}"
    if conf["type"] == "sparse":
        if os.path.exists(f"{os.environ['RETRIV_BASE_PATH']}/collections/{index_name}/dr_state.npz"):
            print(f"Loading existing index: {index_name}")
            retriever = SparseRetriever.load(index_name)

        else:
            retriever = SparseRetriever(
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
            retriever.index(collection)
    elif conf["type"] == "dense":
        if os.path.exists(f"{os.environ['RETRIV_BASE_PATH']}/collections/{index_name}/dr_state.npz"):
            print(f"Loading existing index: {index_name}")
            retriever = ModifiedDenseRetriever.load(index_name)
        else:
            retriever = ModifiedDenseRetriever(
                index_name=index_name,
                model=conf["model"],
                query_model=conf["query_model"] if "query_model" in conf and conf["query_model"] != conf["model"] else None,
                normalize=True,
                max_length=conf["max_length"],
                use_ann=False,
            )
            retriever.index(collection, use_gpu=USE_GPU, batch_size=128)
            
    return retriever


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
    # collection = collection[:1001]
    return collection


def process_review(
    model_name: str,
    retriever,
    review_data,
    cutoff: int,
    review_name: str,
):
    review_details = extract_review_details(review_data)

    for query_type, query in review_details.items():
        if query_type not in QUERY_TYPES:
            continue
        
        ranking = retriever.search(
            query=query, cutoff=cutoff, return_docs=False
        ) # returns (doc_ids, score) pairs
        base_dir = f"../systematic-review-datasets/data/rankings/{model_name}/{query_type}/docs={total_docs}"
        os.makedirs(base_dir, exist_ok=True)
        doc_ids = np.array(list(ranking.keys()), dtype="U64")
        scores = np.array(list(ranking.values()), dtype=np.float32)

        outpath = f"{base_dir}/{review_name}.npz"
        np.savez_compressed(outpath, ids=doc_ids, scores=scores)


if __name__ == "__main__":
    set_seed(SEED)
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
        "MedCPT-Doc-Enc-Only": {
            "type": "dense",
            "model": "ncbi/MedCPT-Article-Encoder",
            "max_length": 256,
        },
        "MiniLM-128": {
            "type": "dense",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 128,
        },
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
        "pubmedbert": {
            "type": "dense",
            "model": "pritamdeka/S-PubMedBert-MS-MARCO",
            "max_length": 512,
        },
        # "roberta": {
        #     "type": "dense",
        #     "model": "sentence-transformers/stsb-roberta-base-v2",
        #     "max_length": 512,
        # },
    }

    dataset = load_dataset()
    # mini dataset
    # max_items = 20
    # mini_dataset = {"EVAL":{}}
    # count = 0
    # for review_name, review_data in dataset["EVAL"].items():
    #     mini_dataset["EVAL"][review_name] = review_data
    #     count += 1
    #     if count >= max_items:
    #         break

    # dataset = mini_dataset
    # mini dataset
    eval_reviews = dataset["EVAL"]
    global_corpus = build_global_corpus(dataset)
    
    total_docs = len(global_corpus)
    for split, reviews in dataset.items():
        print(f"\n=== Split: {split} ===")
        print(f"Number of reviews: {len(reviews)}")
    print("total_docs:", total_docs)
    for name, conf in retriever_configs.items():
        print("Processing model:", name)
        retriever = create_retriever(name, conf, collection=global_corpus)

        qrels_dict = {}
        
        for index, (review_name, review_data) in enumerate(eval_reviews.items(), start=1):
            qrels = {
                doc["document_id"]: int(doc["labels"][0])
                for doc in review_data["data"]["train"]
            }

            qrels_dict[review_name] = qrels

            process_review(
                name,
                retriever,
                review_data,
                cutoff=10_000,
                review_name=review_name,
            )

        for query_type in QUERY_TYPES:
            evaluate_model(
                model_name=name,
                query_type=query_type,
                total_docs=total_docs,
                qrels_dict=qrels_dict,
                output_dir = "reports/title_and_abstract",
                rankings_base_path = "../systematic-review-datasets/data/rankings"
            )


