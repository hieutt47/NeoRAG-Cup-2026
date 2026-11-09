from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np

from neorag.data import load_corpus, load_predictions, load_queries
from neorag.metrics.retrieval import score_query
from neorag.pipelines.learned import TrainedCrossEncoder, grouped_folds, hard_negative_pairs
from neorag.pipelines.reranked import blend_scores

MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
REVISION = '1427fd652930e4ba29e8149678df786c240d8825'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def train(base, target, queries, docs, rankings, epochs, seed):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, local_files_only=True, trust_remote_code=False, use_safetensors=True,
    ).to('cuda')
    # Small data: update only the last two encoder blocks and the scoring head.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(('roberta.encoder.layer.10.', 'roberta.encoder.layer.11.', 'classifier.'))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=.01)
    pairs = [(q.query, positive, negative) for q in queries
             for positive, negative in hard_negative_pairs(q, rankings[q.id], 4)]
    model.train()
    history = []
    for epoch in range(epochs):
        order = list(pairs)
        random.Random(seed + epoch).shuffle(order)
        losses = []
        optimizer.zero_grad()
        batches = (len(order) + 3) // 4
        for batch_index, start in enumerate(range(0, len(order), 4)):
            batch = order[start:start + 4]
            questions = [q for q, _, _ in batch] * 2
            texts = [docs[p].text for _, p, _ in batch] + [docs[n].text for _, _, n in batch]
            encoded = tokenizer(questions, texts, padding=True, truncation='only_second',
                                max_length=512, return_tensors='pt').to('cuda')
            logits = model(**encoded).logits.flatten()
            positive, negative = logits.chunk(2)
            loss = torch.nn.functional.softplus(negative - positive).mean()
            window_size = min(4, batches - (batch_index // 4) * 4)
            (loss / window_size).backward()
            losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % 4 == 0 or batch_index + 1 == batches:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
                optimizer.zero_grad()
        history.append(mean(losses))
        print('EPOCH', target.name, epoch + 1, 'pairs', len(pairs), 'loss', history[-1], flush=True)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return dict(pairs=len(pairs), losses=history)


def aggregate(rows):
    keys = ('recall@5', 'mrr@5', 'ndcg@5')
    result = {key: mean(r[key] for r in rows) for key in keys}
    result['retrieval_score'] = mean(result.values())
    return result


def main():
    parser = argparse.ArgumentParser(description='Fine-tune reranker từ nhãn nguồn train, không gọi gateway.')
    parser.add_argument('--work-dir', default='.cache/trained-reranker-v1')
    parser.add_argument('--report-dir', default='outputs/trained_reranker_v1')
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError('Cần PyTorch CUDA cho lượt huấn luyện này.')
    root, report_dir = Path(args.work_dir), Path(args.report_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    base = root / 'base'
    snapshot_download(MODEL, revision=REVISION, local_dir=base, allow_patterns=[
        'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'sentencepiece.bpe.model',
    ])
    queries = [q for q in load_queries('data/train/questions.jsonl') if q.relevant_ids]
    corpus = load_corpus()
    docs = {d.id: d for d in corpus}
    folds = grouped_folds(queries)
    manifest = dict(model=MODEL, revision=REVISION, seed=20261107, epochs=args.epochs,
                    folds=folds, negative_count=4, learning_rate=1e-5,
                    trainable='last two encoder blocks and classifier',
                    caveat='Train only, no Public/Private labels. Folds have disjoint positive source IDs; corpus text is shared. All train was inspected previously; not untouched validation.')
    manifest_path = root / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Cấu hình khác checkpoint; dùng work-dir mới.')
    write(manifest_path, manifest)
    write(report_dir / 'manifest.json', manifest)
    cache_path = root / 'base_scores.json'
    base_scores = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    scorer = TrainedCrossEncoder(str(base))
    for q in queries:
        if q.id not in base_scores:
            base_scores[q.id] = scorer.score(q.query, corpus).tolist()
            write(cache_path, base_scores)
            print('BASE', len(base_scores), '/', len(queries), flush=True)
    del scorer
    gc.collect()
    torch.cuda.empty_cache()
    rankings = {q.id: [corpus[i].id for i in np.argsort(-np.asarray(base_scores[q.id]), kind='stable')] for q in queries}
    selected = {p.id: p for p in load_predictions('outputs/train_evidence_v1_pred.jsonl')}
    bge_path = Path('outputs/evidence_v1_details.json')
    candidate_details = json.loads(bge_path.read_text(encoding='utf-8'))
    results = []
    for fold, ids in enumerate(folds):
        heldout = set(ids)
        training = [q for q in queries if q.id not in heldout]
        validation = [q for q in queries if q.id in heldout]
        checkpoint = root / f'fold_{fold}'
        if not (checkpoint / 'config.json').exists():
            history = train(base, checkpoint, training, docs, rankings, args.epochs, 20261107 + fold)
            write(report_dir / f'fold_{fold}_training.json', history)
        score_file = root / f'fold_{fold}_scores.json'
        scores = json.loads(score_file.read_text(encoding='utf-8')) if score_file.exists() else {}
        scorer = TrainedCrossEncoder(str(checkpoint))
        for q in validation:
            if q.id not in scores:
                scores[q.id] = scorer.score(q.query, corpus).tolist()
                write(score_file, scores)
                print('VALIDATION', fold, len(scores), '/', len(validation), flush=True)
            learned = np.asarray(scores[q.id])
            base_values = np.asarray(base_scores[q.id])
            ranked = [corpus[i].id for i in np.argsort(-learned, kind='stable')]
            pool = set(candidate_details[q.query]['candidate_ids'])
            pool_positions = [i for i, d in enumerate(corpus) if d.id in pool]
            pool_order = sorted(pool_positions, key=lambda i: -learned[i])
            blended = blend_scores(base_values, learned, .25)
            blend_order = [corpus[i].id for i in np.argsort(-blended, kind='stable')]
            relevant = set(q.relevant_ids)
            results.append(dict(id=q.id, fold=fold, answerable=q.answerable,
                                baseline=score_query(rankings[q.id], relevant, [5]),
                                trained=score_query(ranked, relevant, [5]),
                                blended=score_query(blend_order, relevant, [5]),
                                trained_pool=score_query([corpus[i].id for i in pool_order], relevant, [5]),
                                evidence=score_query(selected[q.id].retrieved_ids, relevant, [5]),
                                retrieved_ids=ranked[:10]))
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
        write(report_dir / 'per_query.json', results)
    metrics = {name: aggregate([r[name] for r in results]) for name in ('baseline', 'trained', 'blended', 'trained_pool', 'evidence')}
    passed = (metrics['trained']['retrieval_score'] >= metrics['baseline']['retrieval_score'] + .01
              and metrics['trained_pool']['retrieval_score'] >= metrics['evidence']['retrieval_score'] - .005)
    report = dict(n_queries=len(results), metrics=metrics, passed=passed,
                  gate=dict(min_gain_over_base=.01, max_drop_vs_evidence_in_candidate_pool=.005),
                  final_model=None, caveat=manifest['caveat'])
    write(report_dir / 'report.json', report)
    if passed:
        history = train(base, root / 'final', queries, docs, rankings, args.epochs, 20261107)
        write(report_dir / 'final_training.json', history)
        report['final_model'] = str(root / 'final')
        write(report_dir / 'report.json', report)
    print('RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
