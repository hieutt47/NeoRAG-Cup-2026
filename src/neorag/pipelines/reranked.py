from __future__ import annotations

from pathlib import Path

import numpy as np

from neorag.data import Document
from neorag.pipelines.hybrid import HybridPipeline


class LocalReranker:
    def __init__(self, batch_size: int = 2):
        self.batch_size = batch_size
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from flashrank import Ranker

            self._model = Ranker(
                model_name="ms-marco-MultiBERT-L-12",
                cache_dir=".cache/flashrank",
                max_length=512,
            )

    def rank(self, query: str, docs: list[Document]) -> list[Document]:
        from flashrank import RerankRequest

        self.load()
        results = []
        for start in range(0, len(docs), self.batch_size):
            passages = [
                {"id": doc.id, "text": doc.text}
                for doc in docs[start : start + self.batch_size]
            ]
            results.extend(self._model.rerank(RerankRequest(query=query, passages=passages)))
        by_id = {doc.id: doc for doc in docs}
        return [by_id[item["id"]] for item in sorted(results, key=lambda item: -float(item["score"]))]


class BGEReranker:
    def __init__(self, model_dir: str = ".cache/bge-reranker-v2-m3", batch_size: int = 1):
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self._session = None

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self._session = ort.InferenceSession(
            str(self.model_dir / "model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=512, strategy="only_second")
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def score(self, query: str, docs: list[Document]) -> np.ndarray:
        self.load()
        scores = []
        inputs = {item.name for item in self._session.get_inputs()}
        for start in range(0, len(docs), self.batch_size):
            encoded = self._tokenizer.encode_batch(
                [(query, doc.text) for doc in docs[start : start + self.batch_size]]
            )
            values = {
                "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
            }
            if "token_type_ids" in inputs:
                values["token_type_ids"] = np.asarray([item.type_ids for item in encoded], dtype=np.int64)
            logits = self._session.run(None, values)[0]
            scores.extend(np.asarray(logits).reshape(-1).tolist())
        return np.asarray(scores)

    def rank(self, query: str, docs: list[Document]) -> list[Document]:
        scores = self.score(query, docs)
        return [docs[index] for index in np.argsort(-scores, kind="stable")]


class RerankedHybridPipeline(HybridPipeline):
    def __init__(self, candidate_count: int = 30, reranker=None, **kwargs):
        super().__init__(**kwargs)
        self.candidate_count = candidate_count
        self.reranker = reranker or LocalReranker()

    def prepare(self) -> None:
        self.reranker.load()

    def retrieve(self, query: str) -> list[Document]:
        candidates = self.retrieve_candidates(query, max(self.candidate_count, self.n_retrieved))
        return self.reranker.rank(query, candidates)[: self.n_retrieved]


class BGEHybridPipeline(RerankedHybridPipeline):
    def __init__(self, reranker=None, **kwargs):
        super().__init__(reranker=reranker or BGEReranker(), **kwargs)


def blend_scores(hybrid: np.ndarray, bge: np.ndarray, bge_weight: float) -> np.ndarray:
    if not 0 <= bge_weight <= 1:
        raise ValueError("bge_weight phải nằm trong [0, 1]")
    hybrid = (hybrid - hybrid.min()) / (np.ptp(hybrid) or 1.0)
    bge = (bge - bge.min()) / (np.ptp(bge) or 1.0)
    return (1 - bge_weight) * hybrid + bge_weight * bge


def diverse_ranking(
    scores: np.ndarray, vectors: np.ndarray, limit: int, diversity: float
) -> list[int]:
    if not 0 <= diversity <= 1:
        raise ValueError("diversity phải nằm trong [0, 1]")
    remaining = list(range(len(scores)))
    selected = []
    similarities = np.clip(vectors @ vectors.T, 0, 1)
    while remaining and len(selected) < limit:
        index = max(
            remaining,
            key=lambda i: (1 - diversity) * scores[i]
            - diversity * (max(similarities[i, selected]) if selected else 0),
        )
        selected.append(index)
        remaining.remove(index)
    return selected


class BlendedBGEHybridPipeline(BGEHybridPipeline):
    def __init__(self, bge_weight: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= bge_weight <= 1:
            raise ValueError("bge_weight phải nằm trong [0, 1]")
        self.bge_weight = bge_weight

    def retrieve(self, query: str) -> list[Document]:
        candidates, hybrid = self.scored_candidates(
            query, max(self.candidate_count, self.n_retrieved)
        )
        bge = self.reranker.score(query, candidates)
        scores = blend_scores(hybrid, bge, self.bge_weight)
        ranking = np.argsort(-scores, kind="stable")[: self.n_retrieved]
        return [candidates[index] for index in ranking]


class DiverseBGEHybridPipeline(BlendedBGEHybridPipeline):
    def __init__(self, diversity: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= diversity <= 1:
            raise ValueError("diversity phải nằm trong [0, 1]")
        self.diversity = diversity

    def index(self, corpus: list[Document]) -> None:
        super().index(corpus)
        self._doc_positions = {doc.id: index for index, doc in enumerate(corpus)}

    def retrieve(self, query: str) -> list[Document]:
        candidates, hybrid = self.scored_candidates(
            query, max(self.candidate_count, self.n_retrieved)
        )
        bge = self.reranker.score(query, candidates)
        scores = blend_scores(hybrid, bge, self.bge_weight)
        vectors = self.doc_vectors[[self._doc_positions[doc.id] for doc in candidates]]
        ranking = diverse_ranking(scores, vectors, self.n_retrieved, self.diversity)
        return [candidates[index] for index in ranking]
