"""Run source, final-PDF text, and collision checks for standalone exports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--figures',type=Path,required=True)
    ap.add_argument('--plot-source',type=Path,default=Path(__file__).resolve().with_name('make_expanded_figures.py'))
    ap.add_argument('--skill-scripts',type=Path,default=Path('${NATURE_FIGURE_SKILL}'))
    args=ap.parse_args();qa=args.figures/'qa';qa.mkdir(parents=True,exist_ok=True)
    source=args.plot_source
    cmd=[sys.executable,str(args.skill_scripts/'validate_figure.py'),str(source),'--backend','python','--json']
    run=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8')
    source_report=json.loads(run.stdout)
    (qa/'source_preflight.json').write_text(json.dumps(source_report,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    manifest_path=args.figures/'figure_manifest.json'
    if manifest_path.exists():
        catalog=json.loads(manifest_path.read_text(encoding='utf-8'))['figures']
        pdfs=[args.figures/(entry['figure']+'.pdf') for entry in catalog]
    else:pdfs=list(args.figures.glob('*.pdf'))
    for pdf in sorted(pdfs):
        collision=qa/(pdf.stem+'.collision.json')
        c=subprocess.run([sys.executable,str(args.skill_scripts/'audit_figure_collisions.py'),str(pdf),
            '--json-out',str(collision)],capture_output=True,text=True,encoding='utf-8')
        t=subprocess.run([sys.executable,str(args.skill_scripts/'audit_pdf_text.py'),str(pdf),'--min-pt','5','--json'],
            capture_output=True,text=True,encoding='utf-8')
        text_report=json.loads(t.stdout)
        (qa/(pdf.stem+'.text.json')).write_text(json.dumps(text_report,ensure_ascii=False,indent=2),encoding='utf-8')
        col=json.loads(collision.read_text(encoding='utf-8')) if collision.exists() else {'error':c.stderr}
        rows.append({'figure':pdf.stem,'collision_exit':c.returncode,'text_exit':t.returncode,
                     'collision_summary':col.get('summary',col.get('verdict',{})),
                     'text_summary':text_report.get('summary',{})})
        print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    report={'source_ready':source_report['summary']['ready'],'figures':rows,
            'n_collision_blockers':sum(r['collision_exit']!=0 for r in rows),
            'n_text_blockers':sum(r['text_exit']!=0 for r in rows),
            'manual_review':'Required at final physical size; automated checks do not establish scientific validity'}
    (qa/'automated_qa_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'n_figures':len(rows),'collision_blockers':report['n_collision_blockers'],'text_blockers':report['n_text_blockers']}))


if __name__=='__main__':main()
