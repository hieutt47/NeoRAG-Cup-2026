import numpy as np
import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.data import DATA_DIR, Document, Query, load_corpus, load_queries
from neorag.pipelines import RAGPipeline, load_pipeline
from neorag.pipelines.baseline import PROMPT, BaselinePipeline
from neorag.pipelines.hybrid import HybridPipeline, MetadataHybridPipeline, _tokens
from neorag.pipelines.reranked import (
    BGEHybridPipeline,
    BGEReranker,
    BlendedBGEHybridPipeline,
    DiverseBGEHybridPipeline,
    LocalReranker,
    RerankedHybridPipeline,
    blend_scores,
    diverse_ranking,
)
from neorag.predict import predict


def test_baseline_runs_end_to_end(corpus):
    pipeline = BaselinePipeline(
        top_k=2, llm=FakeLLM(lambda m: "Ngày 09/10/2011."), embedder=FakeEmbedder()
    )
    pipeline.prepare()
    pipeline.index(corpus)
    preds = predict(pipeline, [Query("q1", "ProPTIT thành lập ngày nào?")], max_retrieved=10)
    assert preds[0].answer == "Ngày 09/10/2011."
    assert len(preds[0].retrieved_ids) == 3  # trả về tối đa n_retrieved (corpus chỉ có 3 doc)
    assert preds[0].retrieved_ids[0] == 1  # doc ProPTIT gần câu hỏi nhất
    assert preds[0].latency_ms is not None


@pytest.mark.parametrize("top_k", [1, 2])
def test_baseline_prompt_preserves_retrieval_and_context(corpus, top_k):
    query = "ProPTIT thành lập ngày nào?"
    embedder = FakeEmbedder()
    vectors = embedder.embed([doc.text for doc in corpus])
    ranking = np.argsort(-(vectors @ embedder.embed([query])[0]))
    expected_docs = [corpus[i] for i in ranking]
    llm = FakeLLM(lambda messages: "Ngày 09/10/2011.")
    pipeline = BaselinePipeline(top_k=top_k, llm=llm, embedder=embedder)
    pipeline.index(corpus)

    output = pipeline.run(query)

    assert output.answer == "Ngày 09/10/2011."
    assert output.retrieved_ids == [doc.id for doc in expected_docs]
    context = "\n\n".join(doc.text for doc in expected_docs[:top_k])
    assert llm.calls == [
        [{"role": "user", "content": PROMPT.format(context=context, query=query)}]
    ]


@pytest.mark.parametrize(
    ("short", "full"),
    [
        ("proptit nk 2025 tv", "CLB Lập Trình PTIT nhiệm kỳ 2025 thành viên"),
        ("clb mm", "CLB Multimedia PTIT"),
        ("gcc", "CLB Nhà Sáng Tạo Game PTIT"),
    ],
)
def test_hybrid_normalizes_accents_and_aliases(short, full):
    assert _tokens(short) == _tokens(full)
    assert "gccabc" in _tokens("gccabc")


@pytest.mark.parametrize("query", ["clb mm studio rieng", "CLB Multimedia studio riêng"])
def test_hybrid_retrieves_unaccented_keywords_and_keeps_prompt(query):
    class TiedEmbedder:
        def embed(self, texts):
            return np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2)

    docs = [
        Document(10, "ProPTIT", "Đào tạo", "CLB Lập Trình PTIT dạy thuật toán."),
        Document(20, "Multimedia", "Cơ sở", "CLB Multimedia PTIT có studio riêng."),
        Document(30, "GCC", "Đào tạo", "GCC đào tạo làm game."),
    ]
    llm = FakeLLM(lambda messages: "Có studio riêng.")
    pipeline = HybridPipeline(top_k=1, n_retrieved=2, llm=llm, embedder=TiedEmbedder())
    pipeline.index(docs)

    output = pipeline.run(query)

    assert output.retrieved_ids[0] == 20
    assert len(output.retrieved_ids) == len(set(output.retrieved_ids)) == 2
    assert llm.calls == [[{"role": "user", "content": PROMPT.format(context=docs[1].text, query=query)}]]


def test_hybrid_without_keyword_matches_preserves_dense_ranking(corpus):
    embedder = FakeEmbedder()
    llm = FakeLLM()
    baseline = BaselinePipeline(llm=llm, embedder=embedder)
    hybrid = HybridPipeline(llm=llm, embedder=embedder)
    baseline.index(corpus)
    hybrid.index(corpus)

    assert hybrid.retrieve("zzzzzz") == baseline.retrieve("zzzzzz")
    assert hybrid.retrieve("") == baseline.retrieve("")
    assert isinstance(load_pipeline("neorag.pipelines.hybrid:HybridPipeline", llm=llm), HybridPipeline)


@pytest.mark.parametrize("candidate_count", [1, 3, 30])
def test_reranked_hybrid_keeps_ids_and_prompt(corpus, candidate_count):
    class FakeReranker:
        loaded = False
        calls = []

        def load(self):
            self.loaded = True

        def rank(self, query, docs):
            self.calls.append((query, docs))
            return list(reversed(docs))

    query = "ProPTIT thành lập ngày nào?"
    llm = FakeLLM()
    reranker = FakeReranker()
    pipeline = RerankedHybridPipeline(
        candidate_count=candidate_count, n_retrieved=2, top_k=1,
        reranker=reranker, llm=llm, embedder=FakeEmbedder(),
    )
    pipeline.prepare()
    pipeline.index(corpus)
    candidates = pipeline.retrieve_candidates(query, max(candidate_count, 2))

    output = pipeline.run(query)

    assert reranker.loaded
    assert reranker.calls == [(query, candidates)]
    expected = list(reversed(candidates))[:2]
    assert output.retrieved_ids == [doc.id for doc in expected]
    assert llm.calls == [[{"role": "user", "content": PROMPT.format(context=expected[0].text, query=query)}]]
    assert isinstance(load_pipeline("neorag.pipelines.reranked:RerankedHybridPipeline", llm=llm), RerankedHybridPipeline)


def test_local_reranker_batches_and_sorts_globally(monkeypatch, corpus):
    import sys
    from types import SimpleNamespace

    calls = []
    scores = {doc.id: score for doc, score in zip(corpus, [0.6, 0.2, 0.9], strict=True)}

    class FakeRanker:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def rerank(self, request):
            calls.append(request)
            return sorted(
                [dict(item, score=scores[item["id"]]) for item in request.passages],
                key=lambda item: -item["score"],
            )

    monkeypatch.setitem(sys.modules, "flashrank", SimpleNamespace(Ranker=FakeRanker, RerankRequest=SimpleNamespace))
    reranker = LocalReranker(batch_size=2)
    assert not calls
    reranker.load()
    reranker.load()
    ranked = reranker.rank("query", corpus)

    assert len(calls) == 3
    assert calls[0]["model_name"] == "ms-marco-MultiBERT-L-12"
    assert [len(call.passages) for call in calls[1:]] == [2, 1]
    assert all(call.query == "query" for call in calls[1:])
    assert ranked == [corpus[2], corpus[0], corpus[1]]


@pytest.mark.parametrize("with_types", [False, True])
def test_bge_reranker_preserves_global_raw_logit_order(corpus, with_types):
    from types import SimpleNamespace

    class FakeTokenizer:
        def encode_batch(self, pairs):
            assert all(query == "query" for query, _ in pairs)
            return [SimpleNamespace(ids=[1, 2], attention_mask=[1, 1], type_ids=[0, 0]) for _ in pairs]

    class FakeSession:
        calls = []

        def get_inputs(self):
            names = ["input_ids", "attention_mask"] + (["token_type_ids"] if with_types else [])
            return [SimpleNamespace(name=name) for name in names]

        def run(self, outputs, values):
            self.calls.append(values)
            assert all(array.dtype == np.int64 for array in values.values())
            return [np.array([[1000.1], [1000.0]]) if len(self.calls) == 1 else np.array([[1000.2]])]

    reranker = BGEReranker(batch_size=2)
    reranker._session = FakeSession()
    reranker._tokenizer = FakeTokenizer()
    assert reranker.rank("query", corpus) == [corpus[2], corpus[0], corpus[1]]
    assert len(reranker._session.calls) == 2
    assert ("token_type_ids" in reranker._session.calls[0]) == with_types


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_blended_pipeline_preserves_prompt_and_candidate_alignment(corpus, weight):
    class FakeScorer:
        def load(self):
            pass

        def score(self, query, docs):
            assert query == "ProPTIT"
            return np.array([float(doc.id) for doc in docs])

    llm = FakeLLM()
    pipeline = BlendedBGEHybridPipeline(
        bge_weight=weight, reranker=FakeScorer(), llm=llm,
        embedder=FakeEmbedder(), n_retrieved=2, top_k=1,
    )
    pipeline.prepare()
    pipeline.index(corpus)
    docs, scores = pipeline.scored_candidates("ProPTIT", 30)
    expected_scores = blend_scores(scores, np.array([float(d.id) for d in docs]), weight)
    expected = [docs[i] for i in np.argsort(-expected_scores, kind="stable")[:2]]
    output = pipeline.run("ProPTIT")
    assert output.retrieved_ids == [doc.id for doc in expected]
    assert llm.calls == [[{"role": "user", "content": PROMPT.format(context=expected[0].text, query="ProPTIT")}]]
    assert docs == pipeline.retrieve_candidates("ProPTIT", 30)
    if weight == 0:
        assert expected == docs[:2]


def test_blend_handles_constant_scores_and_rejects_invalid_weights():
    hybrid = np.array([2.0, 1.0, 0.0])
    assert np.all(np.isfinite(blend_scores(hybrid, np.ones(3), 0.5)))
    assert np.array_equal(np.argsort(-blend_scores(hybrid, np.ones(3), 0.5)), [0, 1, 2])
    for weight in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            blend_scores(hybrid, hybrid, weight)
        with pytest.raises(ValueError):
            BlendedBGEHybridPipeline(bge_weight=weight, llm=FakeLLM())


def test_diverse_ranking_balances_similarity_and_relevance():
    scores = np.array([1.0, 0.9, 0.8])
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert diverse_ranking(scores, vectors, 3, 0) == [0, 1, 2]
    assert diverse_ranking(scores, vectors, 2, 0.3) == [0, 2]
    assert diverse_ranking(scores, vectors, 9, 0.3) == [0, 2, 1]
    assert diverse_ranking(np.array([]), np.empty((0, 2)), 5, 0.3) == []
    with pytest.raises(ValueError):
        diverse_ranking(scores, vectors, 2, -0.1)


def test_diverse_pipeline_zero_penalty_matches_blended(corpus):
    class Scorer:
        def score(self, query, docs):
            return np.arange(len(docs), dtype=float)

    kwargs = dict(llm=FakeLLM(), embedder=FakeEmbedder(), reranker=Scorer())
    blend = BlendedBGEHybridPipeline(**kwargs)
    diverse = DiverseBGEHybridPipeline(diversity=0, **kwargs)
    blend.index(corpus)
    diverse.index(corpus)
    assert diverse.retrieve("ProPTIT") == blend.retrieve("ProPTIT")


def test_metadata_hybrid_uses_metadata_without_changing_source_or_embeddings():
    class RecordingEmbedder:
        calls = []

        def embed(self, texts):
            self.calls.append(texts)
            return np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2)

    docs = [
        Document(10, "GCC", "Đào tạo vẽ", "Có bài tập hình khối."),
        Document(20, "Multimedia", "Chụp ảnh", "Có bài tập ánh sáng."),
    ]
    embedder = RecordingEmbedder()
    llm = FakeLLM()
    pipeline = MetadataHybridPipeline(embedder=embedder, llm=llm, top_k=1)
    pipeline.index(docs)
    output = pipeline.run("gcc ve")
    assert output.retrieved_ids[0] == 10
    assert embedder.calls[0] == [doc.text for doc in docs]
    assert pipeline.docs == docs
    assert llm.calls == [[{"role": "user", "content": PROMPT.format(context=docs[0].text, query="gcc ve")}]]
    assert isinstance(load_pipeline("neorag.pipelines.hybrid:MetadataHybridPipeline", llm=llm), MetadataHybridPipeline)


def test_load_pipeline_validates_spec():
    with pytest.raises(ValueError):
        load_pipeline("neorag.pipelines.baseline")
    with pytest.raises(TypeError):
        load_pipeline("neorag.data:Document")
    assert issubclass(BaselinePipeline, RAGPipeline)
    bge = load_pipeline("neorag.pipelines.reranked:BGEHybridPipeline", llm=FakeLLM())
    assert isinstance(bge, BGEHybridPipeline)
    assert isinstance(bge.reranker, BGEReranker)
    assert bge.reranker._session is None


def test_shipped_data_is_consistent():
    corpus = load_corpus()
    ids = {d.id for d in corpus}
    assert len(ids) == len(corpus)
    for q in load_queries(DATA_DIR / "train" / "questions.jsonl"):
        assert set(q.relevant_ids) <= ids
        if q.answerable:
            assert q.relevant_ids, q.id
