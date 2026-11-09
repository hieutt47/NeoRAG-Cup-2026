import json

import numpy as np
import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.pipelines.baseline import PROMPT
from neorag.pipelines.feedback import (
    FEEDBACK_PROMPT,
    FeedbackBGEHybridPipeline,
    ScopedFeedbackBGEHybridPipeline,
    club_scope,
    parse_feedback,
    protected_coverage_ranking,
)
from neorag.pipelines.reranked import BlendedBGEHybridPipeline


@pytest.mark.parametrize('text', [
    'bad', '{}', 'null', '[1]', '[""]', '["query"]',
    '["a", "b", "c", "d"]', json.dumps(['x' * 401]),
    'prefix\n["a", "b"]',
])
def test_invalid_feedback_preserves_original(text):
    assert parse_feedback(text, 'query') == []


@pytest.mark.parametrize('fence', ['', '```json\n', '```\n'])
def test_feedback_accepts_single_and_multiple_branches(fence):
    suffix = '\n```' if fence else ''
    assert parse_feedback(f'{fence}[" next "]{suffix}', 'query') == ['next']
    assert parse_feedback(f'{fence}["a", "b", "a"]{suffix}', 'query') == ['a', 'b']


def test_protected_coverage_keeps_branch_winners_in_top_five():
    original = np.array([1., .99, .98, .97, .96, .01, .01])
    aspects = np.array([[.99, 0, 0, 0, 0, 1., 0], [.99, 0, 0, 0, 0, 0, 1.]])
    ranking = protected_coverage_ranking(original, aspects, 10)
    assert {0, 5, 6} <= set(ranking[:5])
    assert len(ranking) == len(set(ranking)) == 7
    assert protected_coverage_ranking(original, np.zeros((2, 7)), 3) == [0, 1, 2]
    assert protected_coverage_ranking(original, np.empty((0, 7)), 3) == [0, 1, 2]
    assert protected_coverage_ranking(np.array([]), np.empty((0, 0)), 10) == []
    assert protected_coverage_ranking(original, aspects, 0) == []


@pytest.mark.parametrize(('text', 'clubs'), [
    ('CLB Lập Trình PTIT', {'proptit'}),
    ('clb lt ptit va GCC', {'proptit', 'gcc'}),
    ('MPC, MM, Multimedia', {'multimedia'}),
    ('CLB Nhà Sáng Tạo Game PTIT', {'gcc'}),
    ('clb game ben ve', {'gcc'}),
    ('S2B', {'proptit', 'multimedia', 'gcc'}),
    ('CLB IT PTIT', set()),
])
def test_club_scope_aliases(text, clubs):
    assert club_scope(text) == clubs


def test_branch_credit_rejects_wrong_club_without_removing_candidates(corpus):
    pipeline = ScopedFeedbackBGEHybridPipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    assert pipeline.branch_credit('S2B', 'GCC học vẽ', corpus[0], 1) == 0
    assert pipeline.branch_credit('S2B', 'GCC học vẽ', corpus[2], 2) == .5
    assert pipeline.branch_credit('ProPTIT', 'Multimedia học vẽ', corpus[1], 1) == 0
    assert pipeline.branch_credit('CLB nào?', 'học vẽ', corpus[0], 1) == 1
    from neorag.data import Document
    shared = Document(4, 'Cộng đồng S2B', '', 'Sự kiện hàng năm.')
    explicit = Document(5, 'Cộng đồng S2B', '', 'GCC cùng tổ chức sự kiện.')
    assert pipeline.branch_credit('S2B', 'GCC sự kiện', shared, 1) == 0
    assert pipeline.branch_credit('S2B', 'GCC sự kiện', explicit, 1) == 1


class Scorer:
    def __init__(self):
        self.calls = []

    def score(self, query, docs):
        self.calls.append((query, [d.id for d in docs]))
        return np.array([float(d.id) for d in docs])


def test_empty_feedback_matches_original_and_uses_initial_evidence(corpus):
    scorer = Scorer()
    llm = FakeLLM(lambda messages: '[]')
    kwargs = dict(llm=llm, embedder=FakeEmbedder(), reranker=scorer)
    original = BlendedBGEHybridPipeline(**kwargs)
    pipeline = FeedbackBGEHybridPipeline(**kwargs)
    original.index(corpus)
    pipeline.index(corpus)
    expected = original.retrieve('ProPTIT')
    scorer.calls.clear()
    assert pipeline.retrieve('ProPTIT') == expected
    assert len(scorer.calls) == 1
    context = '\n\n'.join(f'[{d.id}] {d.text[:1600]}' for d in expected[:5])
    assert llm.calls == [
        [{'role': 'user', 'content': FEEDBACK_PROMPT.format(query='ProPTIT', context=context)}]
    ]


@pytest.mark.parametrize('top_k', [1, 10])
def test_feedback_expands_pool_preserves_sources_and_answer_prompt(corpus, top_k):
    class RoutedPipeline(FeedbackBGEHybridPipeline):
        def scored_candidates(self, query, limit):
            docs = self.docs[::-1] if query == 'followup' else self.docs
            return docs[:limit], np.array([1., .5, 0.])[:limit]

    class StrictLLM:
        def __init__(self):
            self.calls = []

        def chat(self, messages, model):
            assert model == 'claude-haiku-5-5'
            self.calls.append(messages)
            return '["followup"]' if len(self.calls) == 1 else 'answer'

    llm = StrictLLM()
    scorer = Scorer()
    pipeline = RoutedPipeline(
        llm=llm, embedder=FakeEmbedder(), reranker=scorer,
        candidate_count=1, n_retrieved=2, aspect_candidates=1,
        llm_model='claude-haiku-5-5', top_k=top_k,
    )
    pipeline.index(corpus)
    output = pipeline.run('query')
    assert output.answer == 'answer'
    assert len(output.retrieved_ids) == len(set(output.retrieved_ids)) == 2
    assert set(output.retrieved_ids) <= {d.id for d in corpus}
    assert scorer.calls == [('query', [1, 2]), ('query', [3]), ('followup', [3])]
    by_id = {d.id: d for d in corpus}
    context = '\n\n'.join(by_id[i].text for i in output.retrieved_ids[:top_k])
    assert llm.calls[-1] == [{'role': 'user', 'content': PROMPT.format(context=context, query='query')}]
    assert all(by_id[d.id] is d for d in pipeline.docs)
