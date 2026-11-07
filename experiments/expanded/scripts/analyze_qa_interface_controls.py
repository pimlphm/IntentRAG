"""Offline input-prefix and materialized-support diagnostics; no model calls."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd

import evaluate_expanded_qa as qa


EXP = Path(__file__).resolve().parents[1]
OUT = EXP/'results/qa_interface_controls'


def lcp(left, right):
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def main():
    rows = qa.load_jsonl(OUT/'qa_responses.jsonl')
    questions = qa.json_read(OUT/'qa_selected_questions.json')
    qmap = {q['question_id']: q for q in questions}
    table = {(r['question_id'], r['method']): r for r in rows}
    core = {(r['question_id'], r['method']): r for r in qa.load_jsonl(EXP/'results/qa_expanded_main/qa_responses.jsonl')}
    expected = ('direct_model_only', 'direct_parametric_knowledge', 'rank_source_first', 'canonical_state_first')
    assert len(rows) == len(questions)*4 == 256
    assert len(table) == 256
    assert all((q['question_id'], method) in table for q in questions for method in expected)
    diagnostics, previous = [], {}
    for q in questions:
        rank = table[(q['question_id'], 'rank_source_first')]
        canonical = table[(q['question_id'], 'canonical_state_first')]
        original = core[(q['question_id'], 'IntentRAG balanced fair')]
        assert rank['execution_kind'] == 'duplicate_alias'
        assert rank['request_sha256'] == original['request_sha256']
        assert rank['raw_response'] == original['raw_response']
        assert set(rank['original_top10_ids']) == set(canonical['ordered_top10_ids'])
        for method in expected:
            row = table[(q['question_id'], method)]
            assert row['materialized_context_tokens'] <= 3000
            assert row['request_payload']['options']['num_predict'] == 128
            assert row['request_payload']['options']['num_ctx'] == 4096
            ids = row['materialized_evidence_ids']
            accepted_hit = float(bool(set(ids)&set(q['accepted_ids'])))
            quote_support = []
            for quote in q.get('answer_quotes', []):
                text, page = qa.normalized(quote['quote']), quote.get('page_id', '')
                quote_support.append(any(text in qa.normalized(source['text']) and
                    (not page or source['page_id'] == page) for source in row['sources']))
            prompt_ids = row['prompt_token_ids']
            source_ids = row['source_context_token_ids']
            previous_row = previous.get(method)
            # Hypothetical one-last-request stream for each interface, in the
            # frozen selection order. The actual calls interleave policies.
            prefix = lcp(prompt_ids, previous_row['prompt_token_ids']) if previous_row else 0
            source_prefix = lcp(source_ids, previous_row['source_context_token_ids']) if previous_row else 0
            diagnostics.append({
                'question_id': q['question_id'], 'origin': qa.origin(q),
                'aircraft_model': q['aircraft_model'], 'source_group': q.get('source_group', ''),
                'method': method, 'source_context_tokens': len(source_ids),
                'prompt_text_tokens_without_chat_template': len(prompt_ids),
                'one_previous_prompt_prefix_tokens_proxy': prefix,
                'one_previous_prompt_prefix_fraction_proxy': prefix/max(1, len(prompt_ids)),
                'one_previous_source_prefix_tokens_proxy': source_prefix,
                'previous_aircraft_same': bool(previous_row and previous_row['aircraft_model']==row['aircraft_model']),
                'n_original_top10': len(row['original_top10_ids']), 'n_materialized_sources': len(ids),
                'actual_materialized_accepted_id_hit_proxy': accepted_hit,
                'actual_materialized_required_quote_fraction': float(np.mean(quote_support)) if quote_support else None,
                'all_required_quotes_materialized': float(all(quote_support)) if quote_support else None,
                'generation_policy': row['generation_policy'], 'execution_kind': row['execution_kind']})
            previous[method] = row
    frame = pd.DataFrame(diagnostics)
    frame.to_csv(OUT/'interface_input_diagnostics_by_query.csv', index=False, encoding='utf-8-sig')
    metrics = ['source_context_tokens', 'prompt_text_tokens_without_chat_template',
        'one_previous_prompt_prefix_tokens_proxy', 'one_previous_prompt_prefix_fraction_proxy',
        'one_previous_source_prefix_tokens_proxy', 'n_materialized_sources',
        'actual_materialized_accepted_id_hit_proxy', 'actual_materialized_required_quote_fraction',
        'all_required_quotes_materialized']
    summary = frame.groupby(['origin', 'method'])[metrics].mean().reset_index()
    summary.to_csv(OUT/'interface_input_diagnostics_summary.csv', index=False, encoding='utf-8-sig')
    paired = []
    for q in questions:
        rank = frame.loc[(frame.question_id==q['question_id'])&(frame.method=='rank_source_first')].iloc[0]
        canon = frame.loc[(frame.question_id==q['question_id'])&(frame.method=='canonical_state_first')].iloc[0]
        rr, cr = table[(q['question_id'], 'rank_source_first')], table[(q['question_id'], 'canonical_state_first')]
        rank_ids, canon_ids = set(rr['materialized_evidence_ids']), set(cr['materialized_evidence_ids'])
        paired.append({'question_id': q['question_id'], 'origin': qa.origin(q), 'source_group': q['source_group'],
            'aircraft_model': q['aircraft_model'], 'materialized_id_jaccard': len(rank_ids&canon_ids)/max(1,len(rank_ids|canon_ids)),
            'canonical_minus_rank_context_tokens': canon.source_context_tokens-rank.source_context_tokens,
            'canonical_minus_rank_prefix_tokens_proxy': canon.one_previous_prompt_prefix_tokens_proxy-rank.one_previous_prompt_prefix_tokens_proxy,
            'canonical_minus_rank_required_quote_fraction': canon.actual_materialized_required_quote_fraction-rank.actual_materialized_required_quote_fraction,
            'canonical_minus_rank_char_f1_proxy': cr['char_bigram_f1_proxy']-rr['char_bigram_f1_proxy'],
            'canonical_minus_rank_valid_json': cr['structured_parse_ok']-rr['structured_parse_ok']})
    pd.DataFrame(paired).to_csv(OUT/'interface_rank_canonical_paired_diagnostics.csv', index=False, encoding='utf-8-sig')
    qa.write_json(OUT/'interface_integrity_audit.json', {
        'status': 'PASS', 'rows': len(rows), 'questions': len(questions), 'source_groups': len({q['source_group'] for q in questions}),
        'per_method_counts': dict(Counter(r['method'] for r in rows)),
        'actual_calls': sum(r['execution_kind']=='actual_call' for r in rows),
        'duplicate_aliases': sum(r['execution_kind']=='duplicate_alias' for r in rows),
        'rank_aliases_raw_response_identical_to_balanced_fair': True,
        'canonical_prebudget_top10_sets_identical': True, 'context_token_ceiling_3000': True,
        'generation_token_ceiling_128': True,
        'prefix_claim_boundary': 'Source/prompt token overlap in a hypothetical one-last-request per-interface stream; eligibility only, not real GPU hit, runtime savings or controlled cache performance.',
        'support_claim_boundary': 'Known source-ID/validated quote retention is an input support proxy, not independent applicability or authorization certification.'})
    print(summary.to_string(index=False), flush=True)
    print('Interface integrity audit PASS', flush=True)


if __name__ == '__main__':
    main()
