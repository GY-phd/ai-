# -*- coding: utf-8 -*-
"""Reproducible preparation of a provisional SFT dataset from the team's
2026-11-22 GitHub main offline rehearsal ledger.
Numbers below were copied from `runs/harness-offline-10/ledger.jsonl`
at Git commit 26bbeed2f08c5dcfb3de2cde87420f848b9db439.
Prompt pattern is the literal `_model_brief()` in
`contest_baselines/layers/controller/round.py` at the same commit.
The editorial answer labels/next tests are constructed, not recorded reviews.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
COMMIT='26bbeed2f08c5dcfb3de2cde87420f848b9db439'
REPO='https://github.com/GY-phd/ai-'

# task, round, hypothesis, parent metric, candidate metric, recorded adoption
ROWS=[
('A',1,'more-data-helps',0.7140297220788709,0.7192279935410042,1),
('A',2,'the-parent-was-left-unconverged',0.7140297220788709,0.6677076605328264,0),
('A',3,'capacity-is-overfitting',0.7140297220788709,0.7205004230794619,1),
('A',4,'rare-class-is-underweighted',0.7140297220788709,0.5737384253870629,0),
('A',5,'the-batch-is-mixing-scales',0.7140297220788709,0.6589315119242508,0),
('A',6,'more-data-helps',0.7140297220788709,0.6394103770687803,0),
('A',7,'more-data-helps',0.7140297220788709,0.6355182365183002,0),
('A',8,'more-data-helps',0.7140297220788709,0.6084383503085725,0),
('A',9,'more-data-helps',0.7140297220788709,0.6192884105162336,0),
('A',10,'more-data-helps',0.7140297220788709,0.6018515410633108,0),
('B',1,'more-data-helps',681.1598326043113,663.9084520308958,1),
('B',2,'the-parent-was-left-unconverged',681.1598326043113,580.5755283063273,1),
('B',3,'the-tail-is-underweighted',681.1598326043113,722.5069075001514,0),
('B',4,'the-unit-is-the-confounder',681.1598326043113,670.8637110695691,1),
('B',5,'capacity-is-overfitting',681.1598326043113,663.2124306558998,1),
('B',6,'more-data-helps',681.1598326043113,660.9807155240039,1),
('B',7,'more-data-helps',681.1598326043113,640.8684501831862,1),
('B',8,'more-data-helps',681.1598326043113,630.0104933431837,1),
('B',9,'more-data-helps',681.1598326043113,626.3719770862434,1),
('B',10,'more-data-helps',681.1598326043113,648.479366774969,1),
('C',1,'more-data-helps',52.86857256611151,51.351144976747236,1),
('C',2,'the-parent-was-left-unconverged',52.86857256611151,51.0362055533061,1),
('C',3,'the-tail-is-underweighted',52.86857256611151,56.04976609404453,0),
('C',4,'the-unit-is-the-confounder',52.86857256611151,51.395827046195755,1),
('C',5,'capacity-is-overfitting',52.86857256611151,51.07113639729663,1),
('C',6,'more-data-helps',52.86857256611151,51.07113639729663,1),
('C',7,'more-data-helps',52.86857256611151,51.18008705888577,1),
('C',8,'more-data-helps',52.86857256611151,51.23359160428948,1),
('C',9,'more-data-helps',52.86857256611151,52.31317325640131,1),
('C',10,'more-data-helps',52.86857256611151,51.19116957331227,1),
]

# Claim/change/falsifier as documented in the repository hypothesis bank.
H={
'more-data-helps':(
'The parent was stopped early, so the model is data-limited rather than capacity-limited.',
"Continue the parent's weights and Adam state for more epochs on the same public training rows.",
'If the held-out score does not improve as the epoch count grows, the model is not data-limited.'),
'the-parent-was-left-unconverged':(
"The parent's learning rate was set for a short schedule and its own validation score was still falling when it stopped, so a larger learning rate reaches a better point.",
'Continue with a larger learning rate.',
"If the parent's validation score was already flat at the end of its own schedule, the schedule - not the rate - was the limit."),
'capacity-is-overfitting':(
'The parent has more capacity than the public rows can support, so it memorises the units it was trained on.',
"Migrate the weights to a narrower hidden layer. The migration keeps the surviving units' weights and Adam state, so it is a continuation and not a re-initialisation.",
"If the parent's training score is not near-perfect while its held-out score is much worse, the problem is not capacity but the target distribution itself."),
'rare-class-is-underweighted':(
'The training mix is not the deployment mix, so a class that is common in training but rare in deployment dominates the Macro-F1 loss.',
'Oversample the rare classes so every class carries the same total mass in the loss.',
"If the rare class's per-class F1 does not rise, the oversampling did not reach it."),
'the-batch-is-mixing-scales':(
"The parent's rows come from batches measured at different times, and the array ages between them, so the loss is dominated by the scale shift rather than by the chemistry.",
"Oversample the rows of the batch that is rarest in training, so the minority batch's own decision boundary is fitted rather than averaged away.",
"If the rare batch's own accuracy does not rise, the batch was not being mixed away."),
'the-tail-is-underweighted':(
'The training rows are dominated by a few long-lived units, so the loss is dominated by the easy rows and the short-lived units - which are the ones the metric punishes hardest - are fitted badly.',
'Oversample the units with the least remaining life, so every unit contributes a comparable mass to the loss.',
'If the per-unit RMSE spread does not shrink, the tail was not underweighted.'),
'the-unit-is-the-confounder':(
'The model cannot tell which unit a row came from, so it predicts the population-average lifetime for every unit and is wrong by the spread of lifetimes.',
"Give every unit the same total mass in the loss, so the population average stops being the objective the model is fitted to. The change is a training-distribution change only: the feature schema, the forward pass and the checkpoint format are untouched, because the platform's prediction request carries no sample id and so cannot reproduce an id-derived column at scoring time.",
'If the per-unit RMSE spread is no larger than the within-unit RMSE, the unit is not the confounder.'),
}

NEXT={
'more-data-helps':'固定其他设置，只改变继续训练轮数，在未参与调参的数据上比较指标走势，确认收益是否具有可重复性。',
'the-parent-was-left-unconverged':'固定训练数据、轮数与随机种子，仅对学习率做对照消融，并检查学习曲线。',
'capacity-is-overfitting':'对比父模型与缩窄模型的训练/留出集误差，检查是否真有容量过拟合而非分布偏移。',
'rare-class-is-underweighted':'查看各类别 F1 与召回率，特别检验少数类是否改善，并用新的独立批次复核。',
'the-batch-is-mixing-scales':'按数据批次统计准确率，核查弱势批次是否获益，再用未参与调参的批次验证。',
'the-tail-is-underweighted':'逐设备/寿命分组复算 RMSE，重点核查短寿命单元是否真正受益。',
'the-unit-is-the-confounder':'按设备单元分组统计 RMSE 的组间离散度，检验是否确实减小，而不只看总体 RMSE。',
}

# Reserve complete task+round records; this is NOT independent external evaluation.
EVAL={('A',4),('A',5),('B',3),('B',4),('C',3),('C',4)}

def make(r):
    task,idx,h,before,after,adopted=r
    claim,change,falsifier=H[h]
    kind='classification' if task=='A' else 'regression'
    metric='Macro-F1（越高越好）' if task=='A' else 'RMSE（越低越好）'
    delta=(after-before) if task=='A' else (before-after)
    improved=delta>1e-10
    assert bool(adopted)==improved, (task,idx,h,adopted,improved)
    user=(
        f'本轮实验记录（数据集 {task}，第 {idx} 轮，任务 {kind}）。\n'
        f'假设：{claim}\n'
        f'所做的改动：{change}\n'
        f'证伪条件：{falsifier}\n'
        f'公开验证集上的主指标：父模型 {before:.4f} -> 候选 {after:.4f}。\n'
        f'当前决策：{"采纳候选" if adopted else "保留父模型"}。\n'
        '请只依据以上真实测量，判断这个读数是否支持该假设，并指出下一步最值得验证的一个改动。'
    )
    score=f'{before:.4f} → {after:.4f}'
    # Crucial limitation: global metric improvement alone does not prove the
    # hypothesis's mechanistic claim. Hence partial_support, never proven.
    conclusion='总体指标改善，但尚不能单凭这一结果证实该假设机制。' if improved else '总体指标未改善，本轮结果不支持该改动。'
    if improved:
        reason=f'{metric}从 {score}，方向改善；本轮记录采纳候选。但要检验具体证伪条件，还需分组或对照实验。'
    else:
        reason=f'{metric}从 {score}，方向退化；本轮保留父模型，不能把该方案当作已验证改进。'
    return {
        'input':user,
        'output':{
            'verdict':'partial_support' if improved else 'not_supported',
            'reason':reason,
            'next_test':NEXT[h],
        },
        'provenance':{
            'repo':REPO,'commit':COMMIT,
            'ledger_path':'runs/harness-offline-10/ledger.jsonl',
            'prompt_template':'contest_baselines/layers/controller/round.py::_model_brief',
            'hypothesis_bank':'contest_baselines/adapters/hypotheses.py',
            'task':task,'round':idx,'hypothesis':h,'metric':metric,
            'before':before,'after':after,'adopted':bool(adopted),
            'output_label_origin':'template_inferred_from_metric_and_recorded_decision; editorial next_test (NOT a logged expert response)'
        }
    }

def main():
    records=[make(r) for r in ROWS]
    assert len(records)==30
    train=[r for r in records if (r['provenance']['task'],r['provenance']['round']) not in EVAL]
    ev=[r for r in records if (r['provenance']['task'],r['provenance']['round']) in EVAL]
    assert len(train)==24 and len(ev)==6
    for name,rs in [('train_repo.jsonl',train),('eval_repo.jsonl',ev)]:
        (HERE/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs),encoding='utf-8')
    manifest={
        'repo':REPO,'commit':COMMIT,'source_paths':[
         'runs/harness-offline-10/ledger.jsonl',
         'contest_baselines/layers/controller/round.py',
         'contest_baselines/adapters/hypotheses.py'],
        'source_type':'single_offline_rehearsal—not independent production or hidden-stream evaluation',
        'derived_count':30,'train_count':len(train),'eval_count':len(ev),
        'split':{'method':'held-out task-round pairs','eval_task_round':sorted([list(x) for x in EVAL]),
         'limitation':'Same offline rehearsal and public data; related experiments and one baseline repeated; NOT generalization test'},
        'labels':'Measured metric & adopted decision copied from ledger; verdict labels and next tests constructed by assistant, not observed human annotations.',
        'safety':'No secret/API key required in files. Review proposed labels with teammates before spending training credits.',
        'counts_train':dict(Counter(r['provenance']['task'] for r in train)),
        'counts_eval':dict(Counter(r['provenance']['task'] for r in ev)),
    }
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
