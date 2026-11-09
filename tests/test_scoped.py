import json

import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.data import Document
from neorag.pipelines.evidence import EvidenceFeedbackPipeline
from neorag.pipelines.scoped import (
    FILL_PROMPT,
    SCOPE_PROMPT,
    ClaimPlanPipeline,
    DeletionEvidencePipeline,
    RelationalScopedPipeline,
    RequirementSearchPipeline,
    RoutedScopedPipeline,
    ScopedEvidencePipeline,
    ScopedFullSourcePipeline,
    SelectedSourcePipeline,
    StructuredEvidencePipeline,
    WindowedScopedPipeline,
    answer_spans,
    apply_sentence_deletions,
    evidence_windows,
    parse_cards,
    parse_claim_plan,
    parse_requirements,
    parse_structured_answer,
)

REQS = [{'need': 'ngày thành lập', 'entity': 'ProPTIT', 'time': ''}]


@pytest.mark.parametrize('raw', ['null', '{}', 'bad', '{"requirements":[]}', '{"requirements":[1]}'])
def test_invalid_requirements(raw):
    assert parse_requirements(raw) == []


@pytest.mark.parametrize('status', ['supported', 'partial', 'missing'])
def test_requirement_search_expands_only_incomplete_cards_and_preserves_top_five(monkeypatch, status):
    docs = [Document(id=i, club='club', topic='topic', text=f'Exact source evidence number {i}.') for i in range(1, 16)]
    pipeline = RequirementSearchPipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    pipeline.docs = docs
    original = {'retrieved_ids': list(range(1, 11)), 'evidence': []}
    events, pools = [], []
    monkeypatch.setattr(pipeline, 'plan_requirements', lambda query: events.append('plan') or REQS)
    monkeypatch.setattr(pipeline, 'initial_details', lambda query: events.append('initial') or original)

    def search(requirement, excluded, limit):
        events.append(('search', set(excluded)))
        return [docs[11]] if not excluded else [docs[12]]

    def cards(query, requirements, sources):
        pools.append([d.id for d in sources])
        source = docs[11] if len(pools) == 1 else docs[12]
        return [dict(REQS[0], status=status, evidence=[] if status == 'missing' else [
            {'id': source.id, 'club': source.club, 'quote': source.text},
        ])]

    monkeypatch.setattr(pipeline, 'search_requirement', search)
    monkeypatch.setattr(pipeline, 'select_cards', cards)
    details = pipeline.retrieve_details('query')
    assert events[:2] == ['plan', 'initial']
    assert details['retrieved_ids'][:5] == list(range(1, 6))
    assert len(details['retrieved_ids']) == len(set(details['retrieved_ids'])) == 10
    assert 12 in pools[0]
    assert len(pools) == (1 if status == 'supported' else 2)
    assert details['supplemental_ids'] == ([] if status == 'supported' else [13])
    assert original == {'retrieved_ids': list(range(1, 11)), 'evidence': []}
    if status != 'missing':
        assert (12 if status == 'supported' else 13) in details['retrieved_ids']


@pytest.mark.parametrize('invalid_scope', [True, False])
def test_requirement_search_invalid_plan_or_cards_keeps_original(monkeypatch, invalid_scope):
    pipeline = RequirementSearchPipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    pipeline.docs = [Document(id=1, club='', topic='', text='source')]
    original = {'retrieved_ids': [1], 'evidence': []}
    monkeypatch.setattr(pipeline, 'plan_requirements', lambda query: [] if invalid_scope else REQS)
    monkeypatch.setattr(pipeline, 'initial_details', lambda query: original)
    monkeypatch.setattr(pipeline, 'search_requirement', lambda *args: [])
    monkeypatch.setattr(pipeline, 'select_cards', lambda *args: [])
    assert pipeline.retrieve_details('query') is original


def test_requirement_search_excludes_seen_sources_before_reranking(monkeypatch):
    import numpy as np

    docs = [Document(id=i, club='', topic='', text=str(i)) for i in range(1, 5)]
    pipeline = RequirementSearchPipeline(llm=FakeLLM(), embedder=FakeEmbedder())
    pipeline.docs = docs
    monkeypatch.setattr(pipeline, 'scored_candidates', lambda q, n: (docs, np.array([1., .8, .6, .4])))
    seen = []

    def score(query, candidates):
        seen.append((query, [d.id for d in candidates]))
        return np.arange(len(candidates), dtype=float)

    monkeypatch.setattr(pipeline.reranker, 'score', score)
    result = pipeline.search_requirement(REQS[0], {1, 2}, 1)
    assert seen[0][1] == [3, 4]
    assert 'ProPTIT' in seen[0][0]
    assert [d.id for d in result] == [4]
    assert pipeline.search_requirement(REQS[0], {1, 2, 3, 4}, 4) == []


def test_requirement_search_writer_uses_full_sources_and_original_prompt(monkeypatch, corpus):
    llm = FakeLLM(lambda messages: 'answer')
    pipeline = RequirementSearchPipeline(llm=llm, embedder=FakeEmbedder())
    pipeline.docs = corpus
    monkeypatch.setattr(pipeline, 'retrieve_details', lambda query: {'retrieved_ids': [2, 1], 'evidence': []})
    result = pipeline.run('query')
    assert result.retrieved_ids == [2, 1]
    assert corpus[0].text in llm.calls[0][0]['content']
    assert corpus[1].text in llm.calls[0][0]['content']
    assert 'BẰNG CHỨNG THEO TỪNG Ý ĐƯỢC HỎI' not in llm.calls[0][0]['content']


def claim_payload(corpus):
    return {'requirements': [{'index': 1, 'status': 'supported', 'claims': [
        {'text': 'ProPTIT được thành lập theo mốc trong nguồn.', 'role': 'answer',
         'evidence': [{'id': 1, 'quote': corpus[0].text}]},
        {'text': 'Điều kiện trong nguồn.', 'role': 'qualification',
         'evidence': [{'id': 2, 'quote': corpus[1].text}]},
    ]}]}


def test_claim_plan_keeps_frozen_scope_and_qualifications(corpus):
    payload = claim_payload(corpus)
    payload['requirements'][0]['need'] = 'ignored'
    parsed = parse_claim_plan('```json\n' + json.dumps(payload) + '\n```', REQS, corpus)
    assert parsed[0]['need'] == REQS[0]['need']
    assert [c['role'] for c in parsed[0]['claims']] == ['answer', 'qualification']


@pytest.mark.parametrize('case', ['bool_index', 'wrong_index', 'unknown_role', 'fake_quote',
                                 'bool_id', 'wrong_id', 'empty_claim', 'no_evidence',
                                 'only_qualification', 'missing_with_claim', 'empty_supported'])
def test_claim_plan_rejects_invalid_plan_atomically(corpus, case):
    payload = claim_payload(corpus)
    item = payload['requirements'][0]
    claim = item['claims'][0]
    if case == 'bool_index':
        item['index'] = True
    elif case == 'wrong_index':
        item['index'] = 2
    elif case == 'unknown_role':
        claim['role'] = 'extra'
    elif case == 'fake_quote':
        claim['evidence'][0]['quote'] = 'fabricated source quote'
    elif case == 'bool_id':
        claim['evidence'][0]['id'] = True
    elif case == 'wrong_id':
        claim['evidence'][0]['id'] = 999
    elif case == 'empty_claim':
        claim['text'] = ' '
    elif case == 'no_evidence':
        claim['evidence'] = []
    elif case == 'only_qualification':
        claim['role'] = 'qualification'
    elif case == 'missing_with_claim':
        item['status'] = 'missing'
    elif case == 'empty_supported':
        item['claims'] = []
    assert parse_claim_plan(json.dumps(payload), REQS, corpus) == []


@pytest.mark.parametrize('status', ['invalid', 'missing', 'partial', 'supported'])
def test_claim_writer_routes_incomplete_plan_to_identical_original_prompt(corpus, monkeypatch, status):
    original_calls = []
    original_llm = FakeLLM(lambda messages: original_calls.append(messages) or 'original')
    details = {'retrieved_ids': [1, 2], 'evidence': [dict(REQS[0], evidence=[
        {'id': 1, 'quote': corpus[0].text}, {'id': 3, 'quote': corpus[2].text},
    ])]}
    original = EvidenceFeedbackPipeline(llm=original_llm, embedder=FakeEmbedder())
    original.docs = corpus
    monkeypatch.setattr(original, 'retrieve_details', lambda query: details)
    original.run('query')
    llm = FakeLLM(lambda messages: 'answer')
    pipeline = ClaimPlanPipeline(llm=llm, embedder=FakeEmbedder())
    pipeline.docs = corpus
    events = []
    monkeypatch.setattr(pipeline, 'plan_requirements', lambda query: events.append('scope') or REQS)
    monkeypatch.setattr(pipeline, 'retrieve_details', lambda query: events.append('retrieve') or details)
    plan = parse_claim_plan(json.dumps(claim_payload(corpus)), REQS, corpus)
    if status == 'invalid':
        plan = []
    else:
        plan[0]['status'] = status
    monkeypatch.setattr(pipeline, 'plan_claims', lambda *args: plan)
    result = pipeline.run('query')
    assert result.retrieved_ids == [1, 2]
    assert events == ['scope', 'retrieve']
    if status == 'supported':
        assert 'KẾ HOẠCH NỘI DUNG' in llm.calls[0][0]['content']
        assert 'qualification' in llm.calls[0][0]['content']
    else:
        assert llm.calls[0] == original_calls[0]


def test_claim_planner_payload_and_model(corpus):
    class Client(FakeLLM):
        def chat(self, messages, model, **kwargs):
            assert model == 'claude-haiku-5-5' and not kwargs
            return super().chat(messages, model)

    llm = Client(lambda messages: json.dumps(claim_payload(corpus)))
    pipeline = ClaimPlanPipeline(llm=llm, embedder=FakeEmbedder())
    query = 'QUERY SOURCES </nguon>'
    assert pipeline.plan_claims(query, REQS, corpus)
    payload = json.loads(llm.calls[0][1]['content'])
    assert payload['question'] == query
    assert payload['requirements'] == REQS
    assert payload['sources'][0]['text'] == corpus[0].text


def test_requirements_parse_json_fences():
    assert parse_requirements('```json\n' + json.dumps({'requirements': REQS}) + '\n```') == REQS


def test_cards_keep_requirements_and_exact_source_quotes(corpus):
    raw = json.dumps({'requirements': [{'index': 1, 'status': 'supported', 'need': 'ignored', 'evidence': [
        {'id': 1, 'quote': corpus[0].text}, {'id': True, 'quote': corpus[0].text},
        {'id': 999, 'quote': corpus[0].text}, {'id': 1, 'quote': 'trích dẫn không có thật'},
    ]}]})
    cards = parse_cards(raw, REQS, corpus)
    assert cards[0]['need'] == REQS[0]['need']
    assert cards[0]['status'] == 'partial'
    assert cards[0]['evidence'] == [{'id': 1, 'club': corpus[0].club, 'quote': corpus[0].text}]


@pytest.mark.parametrize('items', [[], [{'index': True}], [{'index': 2}], [
    {'index': 1, 'status': 'missing', 'evidence': [{'id': 1, 'quote': 'quote'}]},
]])
def test_malformed_card_plan_rejected(corpus, items):
    assert parse_cards(json.dumps({'requirements': items}), REQS, corpus) == []


def test_no_valid_quotes_becomes_missing(corpus):
    raw = json.dumps({'requirements': [{'index': 1, 'status': 'supported', 'evidence': [
        {'id': 1, 'quote': 'fabricated quote'},
    ]}]})
    assert parse_cards(raw, REQS, corpus)[0]['status'] == 'missing'


@pytest.mark.parametrize('wrong_id', [1, 999])
def test_unique_literal_quote_recovers_original_source_id(corpus, wrong_id):
    raw = json.dumps({'requirements': [{'index': 1, 'status': 'supported', 'evidence': [
        {'id': wrong_id, 'quote': corpus[1].text},
    ]}]})
    card = parse_cards(raw, REQS, corpus)[0]
    assert card['status'] == 'supported'
    assert card['evidence'] == [{'id': 2, 'club': corpus[1].club, 'quote': corpus[1].text}]


def test_ambiguous_quote_is_not_remapped(corpus):
    duplicate = Document(id=999, club=corpus[1].club, topic='', text=corpus[1].text)
    raw = json.dumps({'requirements': [{'index': 1, 'status': 'supported', 'evidence': [
        {'id': 1, 'quote': corpus[1].text},
    ]}]})
    assert parse_cards(raw, REQS, corpus + [duplicate])[0]['status'] == 'missing'


def test_duplicate_valid_quotes_do_not_reduce_support(corpus):
    evidence = {'id': 1, 'quote': corpus[0].text}
    raw = json.dumps({'requirements': [{'index': 1, 'status': 'supported',
                                       'evidence': [evidence, evidence]}]})
    card = parse_cards(raw, REQS, corpus)[0]
    assert card['status'] == 'supported'
    assert len(card['evidence']) == 1


def test_evidence_windows_preserve_neighbors_merge_and_exclude_unselected():
    text = 'Far away. Before fact. Target supported fact. After fact. Unrelated tail.'
    doc = Document(id=42, club='club', topic='topic', text=text)
    card = dict(REQS[0], status='supported', evidence=[
        {'id': 42, 'quote': 'Target supported fact.'},
        {'id': 42, 'quote': 'Target supported fact.'},
    ])
    result = evidence_windows([card], [doc, Document(id=43, club='', topic='', text='excluded')])
    assert 'Before fact. Target supported fact. After fact.' in result
    assert 'Far away.' not in result
    assert 'Unrelated tail.' not in result
    assert 'excluded' not in result
    assert result.count('[Nguồn 42;') == 1


def test_evidence_windows_include_repeated_quote_and_multisentence_span():
    doc = Document(id=42, club='', topic='', text='First. Repeated target. Third. Fourth. Repeated target. Last.')
    card = dict(REQS[0], status='supported', evidence=[{'id': 42, 'quote': 'Repeated target.'}])
    assert evidence_windows([card], [doc]).count('Repeated target.') == 2
    card['evidence'] = [{'id': 42, 'quote': 'Third. Fourth.'}]
    assert 'Repeated target. Third. Fourth. Repeated target.' in evidence_windows([card], [doc])


@pytest.mark.parametrize('status,answer,quote,expected', [
    ('supported', 'Valid answer.', True, 'Valid answer.'),
    ('partial', 'Partial answer.', True, ''),
    ('missing', 'Missing answer.', False, ''),
    ('supported', '', True, ''),
    ('supported', 'a' * 2001, True, ''),
    ('supported', 'Unsupported answer.', False, ''),
])
def test_structured_answer_requires_complete_validated_evidence(corpus, status, answer, quote, expected):
    raw = json.dumps({'requirements': [{'index': 1, 'status': status, 'answer': answer,
                                       'evidence': [{'id': 1, 'quote': corpus[0].text}] if quote else []}]})
    assert parse_structured_answer(raw, REQS, corpus) == expected


@pytest.mark.parametrize('valid', [True, False])
def test_structured_runtime_preserves_retrieval_or_falls_back(corpus, monkeypatch, valid):
    monkeypatch.setattr(EvidenceFeedbackPipeline, 'retrieve_details', lambda self, query: {
        'retrieved_ids': [1], 'evidence': [],
    })
    monkeypatch.setattr(EvidenceFeedbackPipeline, 'run', lambda self, query: 'fallback')
    raw = json.dumps({'requirements': [{'index': 1, 'status': 'supported', 'answer': 'Answer.',
                                       'evidence': [{'id': 1, 'quote': corpus[0].text}]}]})
    llm = FakeLLM(lambda messages: raw if valid else 'bad')
    pipeline = StructuredEvidencePipeline(llm=llm, embedder=FakeEmbedder())
    pipeline.docs = corpus
    monkeypatch.setattr(pipeline, 'plan_requirements', lambda query: REQS)
    result = pipeline.run('query')
    if valid:
        assert result.answer == 'Answer.' and result.retrieved_ids == [1]
    else:
        assert result == 'fallback'


def test_deletions_preserve_original_sentence_text():
    draft = 'Chủ thể A.\n\n- Thông tin cần.\n- Ngoài phạm vi.\n'
    assert ''.join(answer_spans(draft)) == draft
    index = next(i for i, span in enumerate(answer_spans(draft), 1) if 'Ngoài' in span)
    result = apply_sentence_deletions(json.dumps({'remove': [{'index': index, 'reason': 'irrelevant'}]}), draft)
    assert 'Chủ thể A.' in result and '- Thông tin cần.' in result
    assert 'Ngoài' not in result
    assert apply_sentence_deletions('{"remove":[]}', draft) == draft


@pytest.mark.parametrize('raw', ['bad', '{}', '{"remove":[true]}',
                                '{"remove":[{"index":true,"reason":"irrelevant"}]}',
                                '{"remove":[{"index":99,"reason":"irrelevant"}]}',
                                '{"remove":[{"index":1,"reason":"rewrite"}]}',
                                '{"remove":[{"index":1,"reason":"irrelevant"},{"index":1,"reason":"duplicate"}]}'])
def test_invalid_deletion_preserves_draft(raw):
    assert apply_sentence_deletions(raw, 'Original answer.') == 'Original answer.'


def test_deletion_pipeline_uses_only_numbered_existing_sentences(corpus):
    draft = 'Required fact. Extra fact.'
    llm = FakeLLM(lambda messages: '{"remove":[{"index":2,"reason":"irrelevant"}]}')
    pipeline = DeletionEvidencePipeline(llm=llm, embedder=FakeEmbedder())
    assert pipeline.finalize_answer('question', 'context', draft, corpus) == 'Required fact.'
    payload = json.loads(llm.calls[0][1]['content'])
    assert payload['question'] == 'question' and payload['sources'] == 'context'
    assert ''.join(s['text'] for s in payload['sentences']) == draft


def test_deleting_everything_preserves_draft():
    assert apply_sentence_deletions('{"remove":[{"index":1,"reason":"irrelevant"}]}', 'Only answer.') == 'Only answer.'


def test_selected_sources_keep_whole_text_and_original_order(corpus):
    pipeline = SelectedSourcePipeline(llm=FakeLLM(lambda messages: ''), embedder=FakeEmbedder())
    cards = [dict(REQS[0], status='supported', evidence=[
        {'id': 2, 'quote': corpus[1].text}, {'id': 1, 'quote': corpus[0].text},
    ])]
    result = pipeline.render_cards(cards, corpus)
    assert corpus[0].text in result and corpus[1].text in result
    assert corpus[2].text not in result
    assert result.index('[Nguồn 1;') < result.index('[Nguồn 2;')
    assert result.count('[Nguồn 1;') == 1


def test_relational_prompt_does_not_change_evidence_fallback():
    assert RelationalScopedPipeline.answer_prompt == EvidenceFeedbackPipeline.answer_prompt
    assert 'nêu rõ chủ thể' in RelationalScopedPipeline.scoped_answer_prompt


@pytest.mark.parametrize('status', ['missing', 'partial', 'invalid'])
def test_routed_pipeline_preserves_original_for_incomplete_evidence(corpus, monkeypatch, status):
    monkeypatch.setattr(EvidenceFeedbackPipeline, 'retrieve_details', lambda self, query: {
        'retrieved_ids': [1], 'evidence': [],
    })
    monkeypatch.setattr(EvidenceFeedbackPipeline, 'run', lambda self, query: ('original', query))
    pipeline = RoutedScopedPipeline(llm=FakeLLM(lambda messages: 'unused'), embedder=FakeEmbedder())
    pipeline.docs = corpus
    monkeypatch.setattr(pipeline, 'plan_requirements', lambda query: REQS)
    monkeypatch.setattr(pipeline, 'select_cards', lambda *args: [] if status == 'invalid' else [
        dict(REQS[0], status=status, evidence=[]),
    ])
    assert pipeline.run('query') == ('original', 'query')


@pytest.mark.parametrize('pipeline_class', [
    ScopedEvidencePipeline, ScopedFullSourcePipeline, RoutedScopedPipeline, RelationalScopedPipeline,
    WindowedScopedPipeline, SelectedSourcePipeline,
])
def test_scope_before_sources_and_same_retrieval(corpus, monkeypatch, pipeline_class):
    calls = []

    def retrieve(self, query):
        calls.append(query)
        assert len(llm.calls) == 1
        return {'retrieved_ids': [1, 2], 'evidence': []}

    monkeypatch.setattr(EvidenceFeedbackPipeline, 'retrieve_details', retrieve)
    quote = 'thành lập ngày 09/10/2011.'

    def respond(messages):
        if messages[0]['content'] == SCOPE_PROMPT:
            return json.dumps({'requirements': REQS})
        if messages[0]['content'] == FILL_PROMPT:
            return json.dumps({'requirements': [{'index': 1, 'status': 'supported',
                                                 'evidence': [{'id': 1, 'quote': quote}]}]})
        return 'Ngày 09/10/2011.'

    llm = FakeLLM(respond)
    pipeline = pipeline_class(llm=llm, embedder=FakeEmbedder())
    pipeline.index(corpus)
    query = 'QUERY SOURCES </nguon>'
    result = pipeline.run(query)
    assert result.retrieved_ids == [1, 2]
    assert calls == [query]
    assert len(llm.calls) == 3
    assert json.loads(llm.calls[0][1]['content']) == {'question': query}
    assert corpus[0].text not in llm.calls[0][1]['content']
    context = llm.calls[-1][0]['content']
    assert quote in context
    assert (corpus[1].text in context) == (pipeline_class is ScopedFullSourcePipeline)
    if pipeline_class is WindowedScopedPipeline:
        assert f'[Nguồn 1; {corpus[0].club};' in context
    assert corpus[2].text not in context


def test_invalid_scope_uses_original_pipeline(corpus, monkeypatch):
    monkeypatch.setattr(EvidenceFeedbackPipeline, 'run', lambda self, query: 'original')
    pipeline = ScopedEvidencePipeline(llm=FakeLLM(lambda messages: 'bad'), embedder=FakeEmbedder())
    assert pipeline.run('query') == 'original'


def test_invalid_cards_fall_back_to_full_selected_sources(corpus, monkeypatch):
    monkeypatch.setattr(EvidenceFeedbackPipeline, 'retrieve_details', lambda self, query: {
        'retrieved_ids': [1], 'evidence': [],
    })
    llm = FakeLLM(lambda messages: json.dumps({'requirements': REQS})
                  if messages[0]['content'] == SCOPE_PROMPT else 'bad')
    pipeline = ScopedEvidencePipeline(llm=llm, embedder=FakeEmbedder())
    pipeline.index(corpus)
    pipeline.run('query')
    assert corpus[0].text in llm.calls[-1][0]['content']
    assert corpus[1].text not in llm.calls[-1][0]['content']
