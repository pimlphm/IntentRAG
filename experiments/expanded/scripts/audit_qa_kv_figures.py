"""Preserve source-chain and exported-PDF QA for the two independent KV plots.

This does not reanalyse observations or redraw scientific content. The effective
source preflight includes the imported exporter because the source-only linter
does not resolve imports. The raw linter result remains preserved, not relabelled.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
FIG = BASE / 'figures_qa_kv'
QA = FIG / 'qa'
SKILL = Path(r'${NATURE_FIGURE_SKILL}')
NAMES = ['actual_kv_reuse_by_prefix_layout', 'native_prefix_overlap_vs_actual_kv']


def run(script, args, dest):
    p = subprocess.run([sys.executable, '-B', str(SKILL / script), *map(str, args)],
                       text=True, capture_output=True, encoding='utf-8')
    dest.write_text(p.stdout, encoding='utf-8')
    if p.stderr:
        dest.with_suffix('.stderr.txt').write_text(p.stderr, encoding='utf-8')
    return {'exit_code': p.returncode, 'output': dest.name}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=BASE/'scripts/make_qa_kv_figures.py')
    parser.add_argument('--figures',nargs='+',default=NAMES)
    parser.add_argument('--manifest',default='actual_kv_qa_manifest.json')
    args=parser.parse_args()
    QA.mkdir(exist_ok=True)
    source = args.source
    dependency = BASE / 'scripts/make_expanded_figures.py'
    raw = source.read_text(encoding='utf-8')
    dep = dependency.read_text(encoding='utf-8')
    tree = ast.parse(dep)
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {'Exporter', 'axes_arrows'}]
    suffix='' if args.manifest=='actual_kv_qa_manifest.json' else '_materialization'
    effective = QA / ('effective_source_for_preflight'+suffix+'.py')
    effective.write_text(raw + '\n\n' + '\n\n'.join(ast.get_source_segment(dep, node)
                                                    for node in selected) + '\n', encoding='utf-8')
    results = {'raw_source': run('validate_figure.py', [source, '--json'], QA/('source_preflight_raw'+suffix+'.json')),
               'resolved_source_chain': run('validate_figure.py', [effective, '--json'], QA/('source_preflight_resolved'+suffix+'.json'))}
    metadata = {}
    import fitz
    for name in args.figures:
        pdf = FIG / (name+'.pdf')
        results[name] = {
            'glyph': run('audit_pdf_text.py', [pdf, '--min-pt', '5', '--json'], QA/(name+'.glyph.json')),
            'collision': run('audit_figure_collisions.py', [pdf, '--json-out', QA/(name+'.collision.json'),
                              '--overlay-pdf', QA/(name+'.collision.pdf')], QA/(name+'.collision.txt'))}
        with fitz.open(pdf) as document:
            p = document[0]
            spans = [s for b in p.get_text('dict')['blocks'] if 'lines' in b
                     for line in b['lines'] for s in line['spans']]
            width_mm = p.rect.width/72*25.4
            min_pt = min(s['size'] for s in spans)
            metadata[name] = {'native_width_mm': width_mm, 'native_height_mm': p.rect.height/72*25.4,
                              'minimum_text_pt': min_pt, 'minimum_at_183mm_width_pt': min_pt*183/width_mm,
                              'minimum_at_89mm_width_pt': min_pt*89/width_mm,
                              'pdf_text_selectable': bool(spans), 'one_axes': True,
                              'sha256_pdf': hashlib.sha256(pdf.read_bytes()).hexdigest()}
            # Final-size proof, 183-mm width at 150 dpi, without modifying content.
            p.get_pixmap(matrix=fitz.Matrix(183/width_mm*150/72, 183/width_mm*150/72),
                         alpha=False).save(QA/(name+'.183mm_preview.png'))
    report = {'backend':'Python', 'archetype':'quantitative single-axes chart',
              'reuse_level':'exact source and statistic reuse; caption clarification only',
              'raw_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'export_dependency_sha256':hashlib.sha256(dependency.read_bytes()).hexdigest(),
              'all_observations':594, 'profiles_per_layout':198, 'independent_query_pairs':6,
              'methods':33, 'exclusions':0, 'results':results, 'physical_size':metadata,
              'submission_size_advisory':'Use double-column/183-mm placement; the individual glyph audit records minimum sizes at 183 and 89 mm. Single-column placement is not certified here.',
              'source_warning_resolutions':{
                  'export':'Raw linter cannot resolve the imported Exporter. Resolved-source-chain audit and existing three export files verify actual SVG/PDF/600-dpi PNG export.',
                  'tiff':'Editable SVG/PDF are the authoritative scientific exports and 600-dpi PNG is the Word/preview asset. No TIFF is required for this draft; no final journal submission certification is claimed.',
                  'width':'Large editable master exports honor requested larger text. Evaluate final typeset sizing at 183 mm, not native width or 89 mm.',
                  'random':'Only seeded horizontal jitter is random. No values, row selection or vertical coordinates are simulated.',
                  'uncertainty':'Raw correlated profiles and descriptive median/IQR are shown, not independent-run mean estimates. Pair-cluster bootstrap contrasts are in the separate result CSV.'},
              'scientific_boundary':'Native engine cache counters, not evidence-only KV tokens, semantic admissibility, GPU speedup under isolation, or full QA correctness.'}
    (QA/args.manifest).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
