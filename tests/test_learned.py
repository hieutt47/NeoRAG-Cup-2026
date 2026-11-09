import numpy as np
import pytest

from neorag.data import Document, Query
from neorag.pipelines.learned import (
    TrainedCrossEncoder,
    TrainedEvidencePipeline,
    grouped_folds,
    hard_negative_pairs,
)


def test_grouped_folds_keep_transitive_source_overlap_together():
    queries = [Query(id='a', query='a', relevant_ids=[1]),
               Query(id='b', query='b', relevant_ids=[1, 2]),
               Query(id='c', query='c', relevant_ids=[2]),
               Query(id='d', query='d', relevant_ids=[3]),
               Query(id='e', query='e', relevant_ids=[4]),
               Query(id='unlabelled', query='u')]
    folds = grouped_folds(queries)
    assert folds == grouped_folds(queries)
    assert sorted(i for fold in folds for i in fold) == ['a', 'b', 'c', 'd', 'e']
    assert any(set(fold) >= {'a', 'b', 'c'} for fold in folds)
    sources = [{d for q in queries if q.id in fold for d in q.relevant_ids} for fold in folds]
    for i in range(len(sources)):
        for j in range(i):
            assert not sources[i] & sources[j]


def test_hard_negatives_never_include_any_positive_or_duplicates():
    query = Query(id='a', query='q', relevant_ids=[2, 3])
    assert hard_negative_pairs(query, [2, 1, 1, 3, 4, 5], 2) == [(2, 1), (2, 4), (3, 1), (3, 4)]
    assert hard_negative_pairs(Query(id='b', query='q'), [1, 2]) == []


def test_trained_reranker_missing_model_is_explicit(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    with pytest.raises(FileNotFoundError, match='Chưa có reranker'):
        TrainedCrossEncoder(str(tmp_path)).load()


def test_trained_reranker_uses_stable_scores_without_model_load(monkeypatch):
    reranker = TrainedCrossEncoder()
    docs = [Document(id=i, club='', topic='', text=str(i)) for i in range(3)]
    monkeypatch.setattr(reranker, 'score', lambda q, ds: np.array([.1, .8, .8]))
    assert [d.id for d in reranker.rank('q', docs)] == [1, 2, 0]


def test_trained_pipeline_does_not_load_training_labels():
    pipeline = TrainedEvidencePipeline(model_dir='not-yet-trained')
    assert isinstance(pipeline.reranker, TrainedCrossEncoder)
    assert str(pipeline.reranker.model_dir) == 'not-yet-trained'
