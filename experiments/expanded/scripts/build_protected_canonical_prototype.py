"""Lossless offline canonicalization of already-retained QA source fragments.

No retrieval, model calls, answer scoring or qrel-guided selection. This is an
input prototype, not an additional completed QA experiment.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

import evaluate_expanded_qa as qa
from evaluate_qa_interface_controls import canonical_key, payload_for
from analyze_qa_interface_controls import lcp

EXP = Path(__file__).resolve().parents[1]
INPUT = EXP/'results/qa_interface_controls'
OUT = EXP/'results/qa_protected_canonical_prototype'


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def serialize(sources):
    return ''.join(f"[{s['label']}] Document: {s['document']}; page: {s['page']}\n{s['text']}\n\n" for s in sources)


def signatures(sources):
    return Counter((s['evidence_id'], sha(s['text']), s['document'], str(s['page']),
                    s['page_id'], s['section_id']) for s in sources)


def support_after_construction(sources, question):
    hits = []
    for item in question.get('answer_quotes', []):
        text, page = qa.normalized(item['quote']), item.get('page_id', '')
        hits.append(any(text in qa.normalized(s['text']) and
                        (not page or s['page_id'] == page) for s in sources))
    return {'actual_materialized_accepted_id_hit_proxy':
            float(bool({s['evidence_id'] for s in sources}&set(question['accepted_ids']))),
            'actual_materialized_required_quote_fraction': float(np.mean(hits)) if hits else None,
            'all_required_quotes_materialized': float(all(hits)) if hits else None}


def main():
    source_file = INPUT/'qa_responses.jsonl'
    source_before = hashlib.sha256(source_file.read_bytes()).hexdigest()
    rows = qa.load_jsonl(source_file)
    table = {(r['question_id'], r['method']): r for r in rows}
    questions = qa.json_read(INPUT/'qa_selected_questions.json')
    tokenizer_file = EXP/'models/qwen2_5_3b_tokenizer/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    ev = pd.read_parquet(qa.PROJECT/'01_processed_corpus/v5/evidence_objects.parquet').fillna('')
    evmap = ev.set_index('evidence_id').to_dict('index')
    OUT.mkdir(parents=True, exist_ok=True)
    prototypes, diagnostics, previous = [], [], {}
    fallback_count = 0
    for q in questions:
        rank = table[(q['question_id'], 'rank_source_first')]
        raw_sources = rank['sources']
        assert tokenizer.encode(serialize(raw_sources), add_special_tokens=False).ids == rank['source_context_token_ids']
        candidate = sorted(deepcopy(raw_sources), key=lambda s: canonical_key(s['evidence_id'], evmap))
        label_map = []
        for index, source in enumerate(candidate, 1):
            original_label, new_label = source['label'], f'S{index}'
            source['label'] = new_label
            label_map.append({'evidence_id': source['evidence_id'], 'original_label': original_label, 'new_label': new_label})
        context = serialize(candidate)
        source_tokens = tokenizer.encode(context, add_special_tokens=False).ids
        fallback = len(source_tokens) > 3000
        if fallback:
            fallback_count += 1
            candidate = deepcopy(raw_sources)
            context = serialize(candidate)
            source_tokens = tokenizer.encode(context, add_special_tokens=False).ids
            label_map = [{'evidence_id': s['evidence_id'], 'original_label': s['label'], 'new_label': s['label']} for s in candidate]
        assert len(source_tokens) <= 3000
        assert signatures(candidate) == signatures(raw_sources)
        # All annotations enter after the package has been frozen. They cannot
        # determine ordering, text boundaries, token-budget fallback or labels.
        new_support = support_after_construction(candidate, q)
        original_support = support_after_construction(raw_sources, q)
        assert new_support == original_support
        payload = payload_for(q, context, qa.SYSTEM, 'qwen2.5:3b')
        prompt_tokens = tokenizer.encode(payload['prompt'], add_special_tokens=False).ids
        prototype = {'question_id': q['question_id'], 'question': q['question'],
            'aircraft_model': q['aircraft_model'], 'origin': qa.origin(q), 'source_group': q['source_group'],
            'method': 'protected_retained_fragment_canonicalization',
            'control_kind': 'offline_input_prototype_not_generated_QA',
            'generation_status': 'pending_not_run', 'actual_model_calls': 0,
            'source_interface': 'rank_source_first', 'sources': candidate,
            'original_retained_ids': rank['materialized_evidence_ids'],
            'materialized_evidence_ids': [s['evidence_id'] for s in candidate],
            'source_fragment_sha256': [{'evidence_id': s['evidence_id'], 'text_sha256': sha(s['text'])} for s in candidate],
            'source_label_map': label_map, 'fallback_original_rank': fallback,
            'fallback_reason': 'reordered_headers_exceed_3000_real_tokens' if fallback else None,
            'source_context_tokens': len(source_tokens), 'source_context_token_ids': source_tokens,
            'prompt_token_ids': prompt_tokens, 'request_payload_for_future_validation_only': payload,
            'lossless_id_text_provenance_multiset_verified': True, **new_support}
        prototypes.append(prototype)
        for method, row in [('rank_source_first', rank), ('canonical_state_first', table[(q['question_id'], 'canonical_state_first')]),
                            ('protected_retained_fragment_canonicalization', prototype)]:
            source_ids, prompt_ids = row['source_context_token_ids'], row['prompt_token_ids']
            prior = previous.get(method)
            prefix = lcp(prompt_ids, prior['prompt_token_ids']) if prior else 0
            metrics = support_after_construction(row['sources'], q)
            diagnostics.append({'question_id': q['question_id'], 'origin': qa.origin(q),
                'aircraft_model': q['aircraft_model'], 'source_group': q['source_group'], 'method': method,
                'source_context_tokens': len(source_ids), 'n_materialized_sources': len(row['sources']),
                'one_previous_prompt_prefix_tokens_proxy': prefix,
                'one_previous_prompt_prefix_fraction_proxy': prefix/max(1, len(prompt_ids)),
                'one_previous_source_prefix_tokens_proxy': lcp(source_ids, prior['source_context_token_ids']) if prior else 0,
                'fallback_original_rank': bool(method.startswith('protected_') and fallback), **metrics})
            previous[method] = row
    with (OUT/'protected_inputs.jsonl').open('w', encoding='utf-8') as handle:
        for row in prototypes:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    frame = pd.DataFrame(diagnostics)
    frame.to_csv(OUT/'protected_input_diagnostics_by_query.csv', index=False, encoding='utf-8-sig')
    columns = ['source_context_tokens', 'n_materialized_sources', 'one_previous_prompt_prefix_tokens_proxy',
        'one_previous_prompt_prefix_fraction_proxy', 'one_previous_source_prefix_tokens_proxy',
        'actual_materialized_accepted_id_hit_proxy', 'actual_materialized_required_quote_fraction',
        'all_required_quotes_materialized']
    summary = frame.groupby(['origin', 'method'])[columns].mean().reset_index()
    summary.to_csv(OUT/'protected_input_diagnostics_summary.csv', index=False, encoding='utf-8-sig')
    source_after = hashlib.sha256(source_file.read_bytes()).hexdigest()
    assert source_after == source_before
    qa.write_json(OUT/'protected_integrity_audit.json', {'status': 'PASS', 'questions': len(questions),
        'all_64_id_text_provenance_multisets_preserved': True, 'all_contexts_at_most_3000_tokens': True,
        'all_input_support_proxies_equal_rank_by_construction': True,
        'fallback_original_rank_count': fallback_count, 'existing_QA_jsonl_SHA256': source_before,
        'existing_QA_jsonl_unchanged': True, 'actual_model_calls': 0, 'semantic_QA_validation': 'pending_not_run'})
    qa.write_json(OUT/'protected_protocol.json', {
        'question_selection': 'same frozen 64 questions and order as completed QA controls',
        'canonicalization_unit': 'already-retained original substring, never untruncated full source object',
        'canonical_sort': ['document_id_or_filename', 'numeric_page', 'section_id', 'role', 'evidence_id'],
        'labels': 'renumbered S1..Sn with explicit old-to-new map; identity map on whole-package fallback',
        'budget': '3000 real Qwen2.5 tokenizer tokens including source headers; whole original rank fallback if exceeded',
        'tokenizer_SHA256': hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        'selection_uses_qrels': False, 'reference_annotations_only_after_input_frozen': True,
        'support_boundary': 'Retention equality is a lossless construction invariant, not new QA accuracy or certified aircraft/revision/authority admissibility.',
        'prefix_boundary': 'Tokenizer longest-common-prefix in hypothetical one-last-request per-interface stream, no chat template; cache eligibility only, not measured GPU KV hits or latency savings.',
        'model_calls': 0, 'prototype_status': 'exploratory offline input mechanism; semantic answer-quality verification pending',
        'why_protected': 'Naive prebudget canonicalization changes which source fragments fit the ceiling; retained-fragment canonicalization preserves query-local support before testing reorder-dependent reuse.'})
    print(summary.to_string(index=False), flush=True)
    print(f'PASS {len(questions)} lossless packages; fallback={fallback_count}; actual_model_calls=0', flush=True)


if __name__ == '__main__':
    main()
