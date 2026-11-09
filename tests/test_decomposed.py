import json

import numpy as np
import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.pipelines.baseline import PROMPT
from neorag.pipelines.decomposed import (
    DECOMPOSE_PROMPT,
    DecomposedBGEHybridPipeline,
    coverage_ranking,
    parse_aspects,
)
from neorag.pipelines.reranked import BlendedBGEHybridPipeline


@pytest.mark.parametrize('value', ['bad', '{}', 'null', '[1, 2]', '["", "ok"]', '["one"]'])
def test_invalid_aspects_preserve_original_query(value):
    assert parse_aspects(value, 'query') == []


def test_aspects_deduplicate_and_bound_external_response():
    assert parse_aspects('[" first ", "second", "first"]', 'query') == ['first', 'second']
    assert parse_aspects(json.dumps(['a', 'b', 'c', 'd']), 'query') == []
    assert parse_aspects(json.dumps(['x' * 401, 'b']), 'query') == []
    assert parse_aspects('["query", "other"]', 'query') == []


@pytest.mark.parametrize('fence', ['```json', '```'])
def test_aspects_accept_fenced_json(fence):
    assert parse_aspects(f'{fence}\n["first", "second"]\n```', 'query') == ['first', 'second']
    assert parse_aspects('prefix\n["first", "second"]', 'query') == []


def test_coverage_selects_complementary_evidence():
    original = np.array([1.0, 0.99, 0.8])
    aspects = np.array([[1.0, 0.5, 0], [0, 0, 1.0]])
    assert coverage_ranking(original, aspects, 2) == [0, 2]
    assert coverage_ranking(original, aspects, 10) == [0, 2, 1]
    assert coverage_ranking(original, np.empty((0, 3)), 2) == [0, 1]
    assert coverage_ranking(np.array([]), np.empty((0, 0)), 10) == []


def test_simple_query_matches_original_blend(corpus):
    class Scorer:
        def score(self, query, docs):
            return np.array([float(doc.id) for doc in docs])

    llm = FakeLLM(lambda messages: '[]')
    kwargs = dict(llm=llm, embedder=FakeEmbedder(), reranker=Scorer())
    original = BlendedBGEHybridPipeline(**kwargs)
    decomposed = DecomposedBGEHybridPipeline(**kwargs)
    original.index(corpus)
    decomposed.index(corpus)
    assert decomposed.retrieve('ProPTIT') == original.retrieve('ProPTIT')
    assert llm.calls == [[{'role': 'user', 'content': DECOMPOSE_PROMPT.format(query='ProPTIT')}]]


def test_decomposed_pipeline_preserves_ids_source_and_answer_prompt(corpus):
    class Scorer:
        calls = []

        def score(self, query, docs):
            self.calls.append((query, [doc.id for doc in docs]))
            return np.arange(len(docs), dtype=float)

    plans = ['ProPTIT thành lập ngày nào?', 'Multimedia thành lập ngày nào?']
    llm = FakeLLM(lambda messages: json.dumps(plans) if 'lập kế hoạch' in messages[0]['content'] else 'answer')
    scorer = Scorer()
    pipeline = DecomposedBGEHybridPipeline(
        llm=llm, embedder=FakeEmbedder(), reranker=scorer, candidate_count=1,
        n_retrieved=2, aspect_candidates=2,
    )
    pipeline.index(corpus)
    output = pipeline.run('So sánh ProPTIT và Multimedia')
    assert output.answer == 'answer'
    assert len(output.retrieved_ids) == len(set(output.retrieved_ids)) == 2
    assert set(output.retrieved_ids) <= {doc.id for doc in corpus}
    by_id = {doc.id: doc for doc in corpus}
    context = '\n\n'.join(by_id[i].text for i in output.retrieved_ids)
    assert llm.calls[-1] == [{'role': 'user', 'content': PROMPT.format(context=context, query='So sánh ProPTIT và Multimedia')}]
    assert [query for query, _ in scorer.calls] == ['So sánh ProPTIT và Multimedia', *plans]
