# -*- coding: utf-8 -*-
"""PyTRIO 0.2.9 starter: supervised fine-tune for the team's ACTUAL
submission-model role: review a completed experiment and propose a next test.

Default: local-only --check. Cloud training requires explicit --train.
No GitHub or PyTRIO token is hardcoded here. Not an official evaluation.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SYSTEM=(
    '你是工业模型训练系统的实验复核助手。输入包含已完成的实验假设、改动、'
    '证伪条件、父模型与候选模型的公开验证指标和程序已经作出的裁决。'
    '你不能更改裁决或伪造没有测量的分组指标。'
    '只输出一个JSON对象，键为verdict、reason、next_test。'
    'verdict只能为partial_support或not_supported；全局指标改善不等于已证实特定机制。'
    'reason应给出可核对的指标方向；next_test提出一个尚未验证的具体实验。'
)

def read_rows(name):
    rows=[]
    for i,line in enumerate((ROOT/name).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        r=json.loads(line)
        if not isinstance(r.get('input'),str) or not isinstance(r.get('output'),dict):
            raise ValueError(f'{name}:{i} input/output format is invalid')
        if set(('verdict','reason','next_test'))-set(r['output']):
            raise ValueError(f'{name}:{i} output missing fields')
        rows.append(r)
    return rows

def check_data():
    t=read_rows('train_repo.jsonl');e=read_rows('eval_repo.jsonl')
    assert t and e
    a={(r['provenance']['task'],r['provenance']['round']) for r in t}
    b={(r['provenance']['task'],r['provenance']['round']) for r in e}
    assert not a & b, 'Identical task-round appears in both sets'
    print('训练案例:',len(t),'  保留案例:',len(e))
    print('示例输入：\n',t[0]['input'][:850])
    print('示例输出：',json.dumps(t[0]['output'],ensure_ascii=False))
    print('注意：来源为同一次公开离线演练，不能作为隐藏数据泛化证明。')
    print('注意：输出是根据真实分数生成的候选文本标签，尚未经队友人工审核。')
    return t,e

def run_training(train,evals):
    import numpy as np
    import pytrio as trio
    client=trio.ServiceClient()
    available=client.get_supported_models()
    wanted=os.environ.get('PYTRIO_MODEL','') or next((m for m in available if '4B' in m),None)
    if not wanted or wanted not in available:
        raise RuntimeError('No supported 4B model; set PYTRIO_MODEL to one in the account-supported list')
    epochs=int(os.environ.get('SFT_EPOCHS','1'))
    batch_size=int(os.environ.get('SFT_BATCH_SIZE','4'))
    if not (1<=epochs<=10 and 1<=batch_size<=64):
        raise ValueError('SFT_EPOCHS or SFT_BATCH_SIZE out of range')
    print('使用模型:',wanted,'训练条数:',len(train),'Epoch:',epochs)
    print('这一步开始消耗云端训练额度。')
    trainer=client.create_lora_training_client(base_model=wanted,rank=16,seed=42)
    tok=trainer.get_tokenizer()

    def prompt_ids(text):
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':text}]
        kwargs={'tokenize':False,'add_generation_prompt':True}
        try:s=tok.apply_chat_template(messages,enable_thinking=False,**kwargs)
        except TypeError:s=tok.apply_chat_template(messages,**kwargs)
        return tok.encode(s,add_special_tokens=False)

    def datum(row):
        prefix=prompt_ids(row['input'])
        suffix=json.dumps(row['output'],ensure_ascii=False,separators=(',',':'))+(tok.eos_token or '')
        target=tok.encode(suffix,add_special_tokens=False)
        tokens=prefix+target
        weights=[0]*len(prefix)+[1]*len(target)
        return trio.Datum(
            model_input=trio.ModelInput.from_ints(tokens=tokens[:-1]),
            loss_fn_inputs={'weights':np.array(weights[1:],dtype=np.float32),
                            'target_tokens':np.array(tokens[1:],dtype=np.int32)})

    examples=[datum(x) for x in train]
    losses=[]
    for ep in range(epochs):
        for i in range(0,len(examples),batch_size):
            batch=examples[i:i+batch_size]
            res=trainer.forward_backward(batch,'cross_entropy').result()
            trainer.optim_step(trio.AdamParams(learning_rate=1e-4)).result()
            loss=(res.metrics.get('loss_mean') if hasattr(res.metrics,'get') else None)
            if loss is None:
                lp=np.concatenate([np.asarray(o['logprobs'],dtype=float).ravel() for o in res.loss_fn_outputs])
                w=np.concatenate([x.loss_fn_inputs['weights'] for x in batch])
                loss=-float(np.dot(lp,w)/max(w.sum(),1))
            losses.append(float(loss))
            print(f'epoch {ep+1}/{epochs}, batch {i//batch_size+1}, loss {float(loss):.4f}',flush=True)

    from datetime import datetime
    weight_name='jyzx-review-sft-'+datetime.now().strftime('%Y%m%d%H%M%S')
    save=trainer.save_weights_for_sampler(name=weight_name).result()
    print('已保存 LoRA 适配器：',save.path)
    base=client.create_sampling_client(base_model=wanted)
    tuned=client.create_sampling_client(base_model=wanted,model_path=save.path)

    def ask(sampler,row):
        sampled=sampler.sample(
           prompt=trio.ModelInput.from_ints(tokens=prompt_ids(row['input'])),
           sampling_params=trio.SamplingParams(max_tokens=400,temperature=0.0),
           num_samples=1).result()
        return sampled.sequences[0].text

    def parse(raw):
        txt=raw.strip()
        try:return json.loads(txt)
        except json.JSONDecodeError:
            m=re.search(r'\{[\s\S]*?\}',txt)
            if m:
                try:return json.loads(m.group())
                except json.JSONDecodeError:pass
        return None

    result=[]
    for idx,row in enumerate(evals,1):
        original=ask(base,row); finetuned=ask(tuned,row)
        a,b=parse(original),parse(finetuned)
        gold=row['output']['verdict']
        entry={
          'case':idx,'task':row['provenance']['task'],'hypothesis':row['provenance']['hypothesis'],
          'expected_verdict':gold,
          'base_valid_json':isinstance(a,dict) and set(('verdict','reason','next_test'))<=set(a),
          'tuned_valid_json':isinstance(b,dict) and set(('verdict','reason','next_test'))<=set(b),
          'base_verdict':a.get('verdict') if isinstance(a,dict) else None,
          'tuned_verdict':b.get('verdict') if isinstance(b,dict) else None,
          'base_reply':original,'tuned_reply':finetuned,
        }
        result.append(entry)
        print(f'case {idx}: expected={gold}, base={entry["base_verdict"]}, tuned={entry["tuned_verdict"]}')
    def rate(key):return sum(x[key]==x['expected_verdict'] for x in result)/len(result)
    report={
       'WARNING':'Six examples from same public offline rehearsal; NOT independent benchmark or competition result.',
       'IMPORTANT':'Answer labels and next_test were constructed by templates; review them before official SFT.',
       'model':wanted,'adapter_path':save.path,'train':len(train),'eval':len(evals),
       'losses':losses,'base_verdict_match':rate('base_verdict'),
       'tuned_verdict_match':rate('tuned_verdict'),'cases':result,
    }
    outfile=ROOT/'sft_repo_report.json'
    outfile.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('已写入：',outfile)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--check',action='store_true',help='only inspect dataset: default')
    p.add_argument('--train',action='store_true',help='execute paid cloud SFT')
    args=p.parse_args()
    train,ev=check_data()
    if args.train:run_training(train,ev)
    else:print('尚未调用云端；如果团队确认标签正确并愿意消耗额度，才运行 --train')

if __name__=='__main__':main()
