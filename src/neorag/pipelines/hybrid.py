from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np

from neorag.data import Document
from neorag.pipelines.baseline import BaselinePipeline

_ALIASES = (
    (r"\b(?:proptit|clb lt ptit|clb lt)\b", "clb lap trinh ptit"),
    (r"\b(?:mpc|clb mm)\b", "clb multimedia ptit"),
    (r"\b(?:gcc|clb lam game)\b", "clb nha sang tao game ptit"),
    (r"\bnk\b", "nhiem ky"),
    (r"\btv\b", "thanh vien"),
)
_STOP_WORDS = frozenset(
    [
        "a", "ak", "ah", "e", "em", "minh", "toi", "ban", "anh", "chi", "oi",
        "thi", "la", "cua", "va", "o", "cho", "voi", "co", "khong", "ko", "k",
        "dc", "duoc", "nhung", "nhieu", "nao", "gi", "sao", "the", "vay", "nay",
        "do", "da", "se", "mot", "cac", "trong", "tren", "khi", "de", "tu", "ve",
        "hay", "nhu", "con", "het", "roi",
    ]
)


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    for pattern, replacement in _ALIASES:
        text = re.sub(pattern, replacement, text)
    return [word for word in re.findall(r"[a-z0-9]+", text) if word not in _STOP_WORDS]


class HybridPipeline(BaselinePipeline):
    def _lexical_text(self, doc: Document) -> str:
        return doc.text

    def index(self, corpus: list[Document]) -> None:
        super().index(corpus)
        self.term_counts = [Counter(_tokens(self._lexical_text(doc))) for doc in corpus]
        self.doc_lengths = np.asarray([sum(counts.values()) for counts in self.term_counts])
        self.average_length = max(float(self.doc_lengths.mean()), 1.0)
        self.doc_frequency = Counter(term for counts in self.term_counts for term in counts)

    def retrieve(self, query: str) -> list[Document]:
        return self.retrieve_candidates(query, self.n_retrieved)

    def retrieve_candidates(self, query: str, limit: int) -> list[Document]:
        docs, _ = self.scored_candidates(query, limit)
        return docs

    def scored_candidates(self, query: str, limit: int) -> tuple[list[Document], np.ndarray]:
        q = self.embedder.embed([query])[0]
        dense = self.doc_vectors @ q
        lexical = np.zeros(len(self.docs))
        for term in set(_tokens(query)):
            frequency = self.doc_frequency.get(term, 0)
            if not frequency:
                continue
            tf = np.asarray([counts.get(term, 0) for counts in self.term_counts])
            idf = np.log(1 + (len(self.docs) - frequency + 0.5) / (frequency + 0.5))
            lexical += idf * tf * 2.5 / (
                tf + 1.5 * (0.25 + 0.75 * self.doc_lengths / self.average_length)
            )

        if not lexical.any():
            scores = dense
            ranking = np.argsort(-scores)[:limit]
        else:
            dense = (dense - dense.min()) / (np.ptp(dense) or 1.0)
            lexical /= lexical.max()
            scores = 0.5 * dense + 0.5 * lexical
            ranking = np.argsort(-scores, kind="stable")[:limit]
        return [self.docs[i] for i in ranking], scores[ranking]


class MetadataHybridPipeline(HybridPipeline):
    def _lexical_text(self, doc: Document) -> str:
        return f"{doc.club}\n{doc.topic}\n{doc.text}"
