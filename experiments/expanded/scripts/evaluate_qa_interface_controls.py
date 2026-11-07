"""Four actual QA controls on the frozen 64-question selection.

Runs after the core/neural QA. Rank input reuses only a byte-identical saved
request. Canonical input keeps Top-10 IDs but changes deterministic source order
before the same token-budget materialization; displayed support can change.
Direct evidence-only and permissive parametric-knowledge policies are separate.
No source or reference is supplied to either direct model before generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error

import pandas as pd
from tokenizers import Tokenizer

import evaluate_expanded_qa as qa


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE_METHOD = 'IntentRAG balanced fair'
METHODS = ('direct_model_only', 'direct_parametric_knowledge',
           'rank_source_first', 'canonical_state_first')
PARAMETRIC_SYSTEM = (
    'Answer the aviation question using your existing model knowledge. '
    'No source documents are supplied. If you cannot determine a document-specific '
    'fact, aircraft condition, action or numeric limit, state that uncertainty. '
    'Do not invent sources, quotes, citations or maintenance authorization. '
    'Return JSON only: {"answer":"brief answer in the question language",'
    '"citations":[],"quotes":[]}. '
    'Use at most 35 English words or 65 Chinese characters in answer.'
)


def canonical_key(eid: str, evmap: dict):
    ev = evmap[eid]
    page = str(ev.get('page_number', ''))
    try:
        page_key = (0, float(page))
    except ValueError:
        page_key = (1, page)
    return (str(ev.get('document_id') or ev.get('doc_id') or ev.get('file_name') or ''),
            page_key, str(ev.get('detected_section_id') or ''),
            str(ev.get('role') or ''), eid)


def payload_for(question: dict, context: str, system: str, model: str):
    prompt = 'SOURCE EXCERPTS\n'+context+'\nAIRCRAFT: '+question['aircraft_model']+'\nQUESTION: '+question['question']+'\nReturn the requested concise JSON.'
    return {'model': model, 'system': system, 'prompt': prompt,
            'stream': False, 'format': 'json', 'keep_alive': '30m',
            'options': {'temperature': 0, 'seed': qa.SEED, 'num_predict': 128, 'num_ctx': 4096}}


def request_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, default=EXPERIMENT)
    parser.add_argument('--qa-main', type=Path, default=None)
    parser.add_argument('--retrieval', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--model', default='qwen2.5:3b')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--token-budget', type=int, default=3000)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--max-calls', type=int, default=0)
    args = parser.parse_args()
    exp = args.experiment.resolve()
    main_dir = args.qa_main.resolve() if args.qa_main else exp/'results/qa_expanded_main'
    output = args.output.resolve() if args.output else exp/'results/qa_interface_controls'
    output.mkdir(parents=True, exist_ok=True)
    for name in ('qa_selection_manifest.json', 'qa_selected_questions.json'):
        source, destination = main_dir/name, output/name
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise RuntimeError('Interface controls must use the same frozen question selection.')
        else:
            shutil.copy2(source, destination)
    questions = qa.json_read(output/'qa_selected_questions.json')
    if len(questions) != 64:
        raise RuntimeError(f'Expected fixed 64 questions, found {len(questions)}.')
    retrieval_path = args.retrieval.resolve() if args.retrieval else exp/'results/all_retrieval_runs.jsonl'
    retrieval = {(row['question_id'], row['method']): row for row in qa.load_jsonl(retrieval_path)}
    main_answers = qa.load_jsonl(main_dir/'qa_responses.jsonl')
    response_path = output/'qa_responses.jsonl'
    history = qa.load_jsonl(response_path)
    completed = {(row['question_id'], row['method'], row['request_sha256']) for row in history}
    cache = {row['request_sha256']: row for row in main_answers+history
             if row.get('raw_response') is not None and row['status'] not in {'http_error', 'exception'}}
    ev = pd.read_parquet(qa.PROJECT/'01_processed_corpus/v5/evidence_objects.parquet').fillna('')
    evmap = ev.set_index('evidence_id').to_dict('index')
    tokenizer_path = exp/'models/qwen2_5_3b_tokenizer/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    qa.write_json(output/'qa_protocol.json', {
        'controls': list(METHODS), 'retrieval_source_method': SOURCE_METHOD,
        'n_frozen_questions': len(questions), 'num_predict': 128, 'num_ctx': 4096,
        'temperature': 0, 'seed': qa.SEED, 'model': args.model,
        'source_context_token_budget_including_headers': args.token_budget,
        'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        'canonical_sort': ['document_id_or_filename', 'numeric_page', 'section_id', 'role', 'evidence_id'],
        'top10_ids_unchanged_for_rank_and_canonical': True,
        'materialized_ids_may_change_due_to_order_and_budget': True,
        'evidence_only_system': qa.SYSTEM, 'parametric_system': PARAMETRIC_SYSTEM,
        'qrels_only_after_generation': True, 'citation_repair': False,
        'nominal_rows': len(questions)*len(METHODS),
        'direct_control_boundary': 'Evidence-only no-source refusal measures compliance under no supplied support. Parametric control uses a different permissive policy; results concern this local 3B model, not larger models or absence of parametric knowledge.',
        'prefix_boundary': 'Stored token-prefix overlap is eligibility only. This run does not control cold/warm cache request histories.'})
    calls = 0
    for question in questions:
        ranked = retrieval[(question['question_id'], SOURCE_METHOD)]
        original_ids = qa.evidence_ids(ranked)
        canonical_ids = sorted(original_ids, key=lambda eid: canonical_key(eid, evmap))
        for method in METHODS:
            if method.startswith('direct_'):
                sources, context, omitted, ordered_ids = [], '', [], []
            else:
                ordered_ids = original_ids if method == 'rank_source_first' else canonical_ids
                row = {'top_evidence_ids': ordered_ids}
                sources, context, omitted = qa.materialize(row, evmap, 2400, tokenizer, args.token_budget)
            system = PARAMETRIC_SYSTEM if method == 'direct_parametric_knowledge' else qa.SYSTEM
            payload = payload_for(question, context, system, args.model)
            sha = request_hash(payload)
            identity = (question['question_id'], method, sha)
            if identity in completed:
                continue
            cached = cache.get(sha)
            started = time.perf_counter()
            if cached:
                raw, status = cached['raw_response'], cached['status']
                execution_kind, elapsed = 'duplicate_alias', None
                call_origin_id = cached.get('call_origin_id') or cached['answer_id']
                alias_origin_method = cached['method']
            else:
                if method == 'rank_source_first':
                    raise RuntimeError('Rank-source request differs from stage1/2 saved payload; refusing to silently replace the promised alias with a new run.')
                if args.max_calls and calls >= args.max_calls:
                    qa.summarize(output, history)
                    print(f'STOP actual_calls={calls} rows={len(history)}', flush=True)
                    return
                qa.write_json(output/'raw_requests'/f'{sha}.json', payload)
                try:
                    raw = qa.request_json(args.base_url, '/api/generate', payload, args.timeout)
                    status = 'ok' if raw.get('done') else 'incomplete'
                    if raw.get('error'):
                        status = 'model_error'
                except urllib.error.HTTPError as error:
                    raw = {'error': str(error), 'response_body': error.read().decode('utf-8', errors='replace')}
                    status = 'http_error'
                except Exception as error:
                    raw, status = {'error': repr(error)}, 'exception'
                elapsed = time.perf_counter()-started
                execution_kind = 'actual_call'
                call_origin_id = f'{question["question_id"]}|{method}'
                alias_origin_method = None
                calls += 1
                qa.write_json(output/'raw_responses'/f'{sha}.json', raw)
            raw_text = str(raw.get('response', ''))
            # References and qrels enter only after the raw response exists.
            scores = qa.score_output(raw_text, sources, question)
            result = {
                'answer_id': f'{question["question_id"]}|{method}', 'question_id': question['question_id'],
                'question': question['question'], 'aircraft_model': question['aircraft_model'],
                'origin': qa.origin(question), 'split': question.get('split', ''),
                'cluster_id': question.get('cluster_id', question['question_id']),
                'source_group': question.get('source_group', ''), 'method': method,
                'control_kind': 'generation_interface_policy_control', 'retrieval_source_method': SOURCE_METHOD,
                'generation_policy': 'parametric_knowledge_permitted' if method == 'direct_parametric_knowledge' else 'evidence_only',
                'original_top10_ids': original_ids, 'ordered_top10_ids': ordered_ids,
                'materialized_evidence_ids': [source['evidence_id'] for source in sources],
                'materialized_context_characters': len(context),
                'materialized_context_tokens': len(tokenizer.encode(context, add_special_tokens=False).ids),
                'source_context_token_ids': tokenizer.encode(context, add_special_tokens=False).ids,
                'prompt_token_ids': tokenizer.encode(payload['prompt'], add_special_tokens=False).ids,
                'sources': sources, 'omitted_evidence_ids': omitted, 'model': args.model,
                'status': status, 'execution_kind': execution_kind, 'call_origin_id': call_origin_id,
                'alias_origin_method': alias_origin_method, 'request_sha256': sha,
                'request_payload': payload, 'raw_response': raw, 'raw_answer_text': raw_text,
                'wall_time_s': elapsed, 'prompt_eval_count': raw.get('prompt_eval_count'),
                'prompt_eval_cached_count': raw.get('prompt_eval_cached_count'),
                'prompt_eval_duration_ns': raw.get('prompt_eval_duration'),
                'eval_count': raw.get('eval_count'), 'eval_duration_ns': raw.get('eval_duration'),
                'total_duration_ns': raw.get('total_duration'), 'done_reason': raw.get('done_reason'),
                **scores}
            with response_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
                handle.flush()
            history.append(result)
            completed.add(identity)
            if status not in {'http_error', 'exception'}:
                cache[sha] = result
            print(f'CONTROL {len(history)} {question["question_id"]} {method} {execution_kind} parse={scores["structured_parse_ok"]}', flush=True)
        qa.summarize(output, history)
    qa.summarize(output, history)
    qa.write_json(output/'interface_execution_summary.json', {
        'rows': len(history), 'actual_calls': sum(row['execution_kind']=='actual_call' for row in history),
        'duplicate_aliases': sum(row['execution_kind']=='duplicate_alias' for row in history),
        'method_counts': dict(Counter(row['method'] for row in history)),
        'policy_difference_explicit': True, 'independent_semantic_judge': False})
    print(f'COMPLETE controls={len(history)} actual_calls_this_run={calls}', flush=True)


if __name__ == '__main__':
    main()
