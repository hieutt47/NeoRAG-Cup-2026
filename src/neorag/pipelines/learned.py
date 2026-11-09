from __future__ import annotations

from pathlib import Path

import numpy as np

from neorag.data import Document, Query
from neorag.pipelines.evidence import EvidenceFeedbackPipeline


def grouped_folds(queries: list[Query], n_folds: int = 3) -> list[list[str]]:
    labelled = [q for q in queries if q.relevant_ids]
    parent = list(range(len(labelled)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, query in enumerate(labelled):
        for doc_id in query.relevant_ids:
            if doc_id in seen:
                parent[root(index)] = root(seen[doc_id])
            seen[doc_id] = index
    groups = {}
    for index, query in enumerate(labelled):
        groups.setdefault(root(index), []).append(query.id)
    folds = [[] for _ in range(n_folds)]
    for group in sorted(groups.values(), key=lambda g: (-len(g), sorted(g))):
        target = min(range(n_folds), key=lambda i: (len(folds[i]), i))
        folds[target].extend(sorted(group))
    return folds


def hard_negative_pairs(query: Query, ranking: list[int], count: int = 4) -> list[tuple[int, int]]:
    positives = set(query.relevant_ids)
    negatives = list(dict.fromkeys(i for i in ranking if i not in positives))[:count]
    return [(positive, negative) for positive in sorted(positives) for negative in negatives]


class TrainedCrossEncoder:
    def __init__(self, model_dir: str = '.cache/trained-reranker-v1/final', batch_size: int = 8):
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (self.model_dir / 'config.json').exists():
            raise FileNotFoundError(f'Chưa có reranker đã học tại {self.model_dir}')
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).to(self._device).eval()

    def score(self, query: str, docs: list[Document]) -> np.ndarray:
        self.load()
        import torch

        scores = []
        with torch.inference_mode():
            for start in range(0, len(docs), self.batch_size):
                batch = docs[start:start + self.batch_size]
                encoded = self._tokenizer([query] * len(batch), [d.text for d in batch],
                                          padding=True, truncation='only_second', max_length=512,
                                          return_tensors='pt').to(self._device)
                scores.extend(self._model(**encoded).logits.flatten().float().cpu().tolist())
        return np.asarray(scores)

    def rank(self, query: str, docs: list[Document]) -> list[Document]:
        return [docs[i] for i in np.argsort(-self.score(query, docs), kind='stable')]


class TrainedEvidencePipeline(EvidenceFeedbackPipeline):
    def __init__(self, model_dir: str = '.cache/trained-reranker-v1/final', **kwargs):
        super().__init__(reranker=TrainedCrossEncoder(model_dir), **kwargs)
