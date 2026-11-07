"""Post-hoc, source-traceable illustrations from the completed native KV run.

Examples are deliberately chosen to explain contrasting mechanisms. They are
not additional validation, random samples, or semantic admissibility judgments.
"""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd
from build_kv_prompts import ROOT

BASE = Path(__file__).resolve().parents[1]
LAYOUTS = ['query_first', 'canonical_state_first', 'hierarchical_state_first']
CASES = [
    ('shared_span_and_early_index_break', 'QV21E_0086', 'IntentRAG balanced fair'),
    ('stable_ancestor_index', 'QV21E_0086', 'Profile-resolved'),
    ('common_header_only_opportunity', 'QV21E_0316', 'Long-term fixed'),
]


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parts(text):
    source = text.split('Source excerpts:\n', 1)[1]
    body = source.split('\nQuestion: ', 1)[0]
    body = body.split('<|im_end|>', 1)[0]
    tuples = []
    if 'Persistent routing index (metadata, not answer evidence):\n' in text:
        index = text.split('Persistent routing index (metadata, not answer evidence):\n', 1)[1].split('Source excerpts:\n', 1)[0].strip()
        tuples = [line.split('|') for line in index.splitlines()]
    question = re.search(r'Question: ([^\n]+)', text).group(1).split('<|im_end|>', 1)[0]
    headers = re.findall(r'\[S(\d+)\] id=([^;]+); document=([^;]+); page=([^;]+); role=([^\n]+)', body)
    return {'question':question, 'ancestor_tuples':tuples, 'source_body_sha256':sha(body),
            'source_body_characters':len(body), 'emitted_source_headers':[{'label':'S'+label, 'evidence_id':eid,
             'document_id':doc, 'page':page, 'role':role} for label,eid,doc,page,role in headers]}


def common_prefix(a, b):
    for n, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return n
    return min(len(a), len(b))


def location(text, index):
    source_start = text.index('Source excerpts:\n')
    marker = 'Persistent routing index (metadata, not answer evidence):\n'
    if marker in text and text.index(marker) <= index < source_start:
        return 'hierarchy routing metadata'
    q = text.index('Question: ')
    if q < source_start and q <= index < source_start:
        return 'query before evidence'
    if index >= source_start and (q < source_start or index < q):
        return 'source-header or source-body sequence'
    if index >= q:
        return 'query after evidence'
    return 'common system/scope header'


def main():
    prompts = [json.loads(l) for l in (BASE/'data/real_kv_prompts.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup = {(p['query_id'],p['method'],p['namespace']):p for p in prompts}
    runs = pd.read_csv(BASE/'results/real_kv/kv_pairs.csv').set_index(['query_id','method','namespace'])
    retrieval = [json.loads(l) for l in (BASE/'results/all_retrieval_runs.jsonl').read_text(encoding='utf-8').splitlines()]
    original_ids = {(r['question_id'],r['method']):r['top_evidence_ids'].split(';') for r in retrieval}
    corpus = pd.read_parquet(ROOT/'01_processed_corpus/v5/evidence_objects.parquet',
        columns=['evidence_id','granularity','document_id','page_number','detected_section_id','component_family','text']).fillna('')
    corpus['original_source_characters'] = corpus.text.str.len()
    source_metadata = corpus.drop(columns='text').set_index('evidence_id').to_dict('index')
    cases, rows = [], []
    for case_id, qid, method in CASES:
        item = {'case_id':case_id, 'target_query_id':qid, 'method':method, 'layouts':[]}
        for layout in LAYOUTS:
            p = lookup[qid,method,layout]
            row = runs.loc[qid,method,layout]
            assert sha(p['prompt']) == row.prompt_sha256
            assert sha(p['seed_prompt']) == row.seed_prompt_sha256
            assert row.warm_timing_cache_n + row.warm_timing_prompt_n == row.prompt_token_count_native
            target, seed = parts(p['prompt']), parts(p['seed_prompt'])
            for state in [target, seed]:
                for header in state['emitted_source_headers']:
                    header.update(source_metadata[header['evidence_id']])
            lcp = common_prefix(p['prompt'],p['seed_prompt'])
            detail = {'layout':layout, 'seed_query_id':p['seed_query_id'], 'aircraft_model':p['aircraft_model'],
                'pair_index':p['pair_index'], 'topic':p['topic'], 'target':target, 'seed':seed,
                'target_retrieved_ids':original_ids[qid,method], 'seed_retrieved_ids':original_ids[p['seed_query_id'],method],
                'shared_retrieved_ids':p['shared_evidence_ids'], 'target_source_tokens':p['target_source_tokens'],
                'seed_source_tokens':p['seed_source_tokens'], 'character_lcp':lcp,
                'first_divergence_location':location(p['prompt'],lcp),
                'target_at_first_divergence':p['prompt'][max(0,lcp-30):lcp+90],
                'seed_at_first_divergence':p['seed_prompt'][max(0,lcp-30):lcp+90],
                'native_prompt_tokens':int(row.prompt_token_count_native),
                'native_prefix_lcp_tokens':int(row.seed_target_lcp_native),
                'actual_reused_input_tokens':int(row.warm_timing_cache_n),
                'newly_evaluated_input_tokens':int(row.warm_timing_prompt_n),
                'actual_reuse_fraction':float(row.warm_timing_cache_n/row.prompt_token_count_native),
                'cold_cache_tokens':int(row.cold_timing_cache_n), 'cold_warm_text_identical':bool(row.outputs_identical),
                'source_body_identical':target['source_body_sha256']==seed['source_body_sha256'],
                'ancestor_index_identical':target['ancestor_tuples']==seed['ancestor_tuples']}
            item['layouts'].append(detail)
            rows.append({k:v for k,v in detail.items() if k not in {'target','seed','target_retrieved_ids','seed_retrieved_ids','shared_retrieved_ids'}} |
                        {'case_id':case_id,'target_query_id':qid,'method':method,
                         'shared_evidence_count':len(p['shared_evidence_ids']),
                         'target_emitted_source_count':len(target['emitted_source_headers']),
                         'seed_emitted_source_count':len(seed['emitted_source_headers']),
                         'target_ancestor_tuples':json.dumps(target['ancestor_tuples'],ensure_ascii=False),
                         'seed_ancestor_tuples':json.dumps(seed['ancestor_tuples'],ensure_ascii=False)})
        cases.append(item)
    out = BASE/'reports'
    out.mkdir(exist_ok=True)
    (out/'real_kv_mechanism_examples.json').write_text(json.dumps({
        'selection':'Post-hoc illustrative selection after inspecting completed counters; no further experiment, random sampling or independent validation.',
        'case_selection_rule':'One high canonical prefix with early index mismatch, one stable index on the same pair, one empty evidence-ID intersection.',
        'boundaries':['Counters include system/scope/metadata tokens, not evidence-only support.',
                      'Same aircraft/topic does not establish maintenance applicability; revision and authority are unverified.',
                      'Native 32-token probes measure engine reuse, not full-answer correctness or isolated throughput.',
                      'Early-changing metadata breaks sequential prefix caching; this is not a test of all hierarchical-memory designs.'],
        'cases':cases},ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(out/'real_kv_mechanism_examples.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rows)[['case_id','layout','native_prompt_tokens','actual_reused_input_tokens','newly_evaluated_input_tokens','ancestor_index_identical','first_divergence_location']].to_string(index=False))


if __name__ == '__main__':
    main()
