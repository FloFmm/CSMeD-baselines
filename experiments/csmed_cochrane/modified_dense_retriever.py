import numpy as np
import os 
from retriv import DenseRetriever, Encoder
from retriv.dense_retriever.dense_retriever import compute_scores, compute_scores_multi, ANN_Searcher
from retriv.paths import embeddings_folder_path, dr_state_path
from typing import List, Dict, Any

class ModifiedDenseRetriever(DenseRetriever):
    def __init__(self, query_model=None, **kwargs):
        """
        Args:
            query_model: str or Encoder object for encoding queries.
            kwargs: passed to original DenseRetriever (document encoder)
        """
        super().__init__(**kwargs)

        self.use_ann = False # TODO hacky
        self.query_model = query_model
        if query_model is not None:
            self.query_encoder = Encoder(
                index_name=self.index_name + "_query",
                model=query_model,
                normalize=self.normalize,
                max_length=self.max_length,
            )

    def search(
        self,
        query: str,
        return_docs: bool = True,
        cutoff: int = 100,
    ) -> List:
        """Override search to use query_encoder instead of doc encoder for queries."""

        if self.query_model is not None:
            encoded_query = self.query_encoder(query)
        else:
            encoded_query = self.encoder(query)

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
        if self.query_model is not None:
            encoded_queries = self.query_encoder(q_texts, batch_size, show_progress=False)
        else:
            encoded_queries = self.encoder(q_texts, batch_size, show_progress=False)

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
    
    @staticmethod
    def load(index_name: str = "new-index"):
        """Load a retriever and its index.

        Args:
            index_name (str, optional): Name of the index. Defaults to "new-index".

        Returns:
            DenseRetriever: Dense Retriever.
        """

        state = np.load(dr_state_path(index_name), allow_pickle=True)["state"][()]
        dr = ModifiedDenseRetriever(**state["init_args"])
        dr.initialize_doc_index() #docs.jsonl
        dr.id_mapping = state["id_mapping"]
        dr.doc_count = state["doc_count"]
        if state["embeddings"]:
            dr.load_embeddings()
        if dr.use_ann:
            dr.ann_searcher = ANN_Searcher.load(index_name)
        return dr

    def load_embeddings(self):
        """Internal usage."""
        print("start efficient loading")
        path = embeddings_folder_path(self.index_name)
        npy_file_paths = sorted(f for f in os.listdir(path) if f.endswith(".npy"))

        # Memory-mapped loading
        mapped = [np.load(path / f, mmap_mode="r") for f in npy_file_paths]

        # Compute full shape without loading into RAM
        total_rows = sum(m.shape[0] for m in mapped)
        dim = mapped[0].shape[1]

        # Create a big memmap file for efficient random access (optional)
        memmap_path = path / "all_embeddings.dat"
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(total_rows, dim))

        pos = 0
        for m in mapped:
            mm[pos : pos + m.shape[0]] = m[:]  # paged in small chunks
            pos += m.shape[0]
        print("finished efficient loading")
        
        self.embeddings = mm


    def save(self):
        """Save the state of the retriever to be able to restore it later."""

        state = dict(
            init_args=dict(
                index_name=self.index_name,
                model=self.model,
                query_model=self.query_model,
                normalize=self.normalize,
                max_length=self.max_length,
                use_ann=self.use_ann,
            ),
            id_mapping=self.id_mapping,
            doc_count=self.doc_count,
            embeddings=True if self.embeddings is not None else None,
        )
        np.savez_compressed(dr_state_path(self.index_name), state=state)


