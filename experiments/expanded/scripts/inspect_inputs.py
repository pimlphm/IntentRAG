from pathlib import Path
import pandas as pd
import json

ROOT = Path(r'${INTENTRAG_PROJECT_ROOT}')
OUT = Path(__file__).resolve().parents[1]
def main():
    e = pd.read_parquet(ROOT / '01_processed_corpus/v5/evidence_objects.parquet').fillna('')
    b = pd.read_csv(ROOT / '03_benchmark/benchmark_v21_evidence_sufficient_locked/qa_warehouse_v21_full.csv').fillna('')
    print('root', ROOT)
    print('evidence', len(e), 'bad_unicode_rows', e.text.str.contains('\ufffd').sum())
    for aircraft in e.aircraft_model.unique():
        sub = e[e.aircraft_model.eq(aircraft)]
        print(aircraft, 'unique_text', sub.text.nunique(), 'pages', sub.page_id.nunique())
        for _, r in sub[sub.text.str.len().between(180, 900)].iloc[10:12].iterrows():
            print(r.evidence_id, r.role, r.components, r.text[:600])
    print('benchmark', len(b), b.groupby('split').size().to_dict())
    for name in ['qa_warehouse_full.csv']:
        frame = pd.read_csv(ROOT / '04_results' / name).fillna('')
        print(name, len(frame), list(frame.columns))
    print('runtime')
    import urllib.request
    with urllib.request.urlopen('http://localhost:11434/api/tags', timeout=10) as r:
        data = json.load(r)
    print([(m['name'], round(m.get('size',0)/1e9,2)) for m in data['models'] if 'qwen' in m['name'].lower() or 'embed' in m['name'].lower()])
if __name__ == '__main__':
    main()
