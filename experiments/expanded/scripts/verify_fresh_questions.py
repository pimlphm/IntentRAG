"""Second-model support screening before any fresh retrieval evaluation."""
from pathlib import Path
import argparse,json,re,time,urllib.request
import pandas as pd

OUT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='qwen2.5:7b-instruct')
    ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    questions=pd.read_csv(OUT/'data/fresh_questions.csv').fillna('')
    rawfile=OUT/'data/fresh_question_support_checks.jsonl'
    old=[json.loads(l) for l in rawfile.read_text(encoding='utf-8').splitlines()] if rawfile.exists() else []
    known={(x['question_id'],x['question_sha']) for x in old}
    import hashlib
    for n,(_,r) in enumerate(questions.iterrows()):
        if args.limit and n>=args.limit:break
        checksum=hashlib.sha256(r.question.encode()).hexdigest()
        if (r.question_id,checksum) in known:continue
        evidence=json.loads(r.answer_quotes_json)
        prompt='''Decide whether the quoted source statements directly answer the specific question.
The question and quotations may be English or Chinese. A statement merely
repeating the question's symptom does not answer a request for its cause,
component, action, threshold, or condition. Reject invented detail or a missing
part of a combined question. Do not use outside aircraft knowledge.
Return JSON: {"fully_supported": true/false, "reason": "short reason"}.
Question: '''+r.question+'\nQuoted statements:\n'+'\n'.join(a['quote'] for a in evidence)
        payload=dict(model=args.model,prompt=prompt,stream=False,keep_alive='5m',
            format={'type':'object','properties':{'fully_supported':{'type':'boolean'},'reason':{'type':'string'}},
                    'required':['fully_supported','reason']},
            options=dict(temperature=0,seed=20261014,num_ctx=2048,num_predict=100))
        started=time.perf_counter()
        result=dict(question_id=r.question_id,question_sha=checksum,model=args.model)
        try:
            req=urllib.request.Request('http://localhost:11434/api/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as f:data=json.load(f)
            data.pop('context',None)
            result.update(json.loads(data['response']),raw_response=data,elapsed_s=time.perf_counter()-started)
        except Exception as ex:result.update(fully_supported=False,error=repr(ex))
        with rawfile.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(f'SUPPORT CHECK {n+1}/{len(questions)} pass={result["fully_supported"]}',flush=True)
    checks=pd.DataFrame([json.loads(l) for l in rawfile.read_text(encoding='utf-8').splitlines()])
    checks=checks.drop_duplicates(['question_id','question_sha'],keep='last')
    ok=set(checks[checks.fully_supported.eq(True)].question_id)
    selected=questions[questions.question_id.isin(ok)].copy()
    selected['support_screen_model']=args.model
    selected.to_csv(OUT/'data/fresh_questions_verified.csv',index=False,encoding='utf-8-sig')
    print('VERIFIED',len(selected),'RAW_ACCEPTED',len(questions),flush=True)

if __name__=='__main__':main()
