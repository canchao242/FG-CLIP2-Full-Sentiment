"""Recalculate all released predictions without models, private data or sklearn."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42,123,3407,2026,2027]

def metrics(labels, logits):
    labels, logits = np.asarray(labels), np.asarray(logits)
    if labels.ndim != 1 or logits.shape != (len(labels),3) or not np.isfinite(logits).all():
        raise ValueError('Invalid logits/label shape or nonfinite logits')
    if not np.isin(labels,[0,1,2]).all():
        raise ValueError('Invalid class labels')
    predicted = logits.argmax(axis=1)
    matrix = np.zeros((3,3),dtype=np.int64)
    np.add.at(matrix,(labels.astype(int),predicted),1)
    denom = matrix.sum(axis=0)+matrix.sum(axis=1)
    f1 = np.divide(2*matrix.diagonal(),denom,out=np.zeros(3,dtype=float),where=denom!=0)
    return float(f1.mean()),float(np.mean(labels==predicted))

def verify(root=ROOT):
    manifest = pd.read_csv(root/'data/split_manifest_public.csv',keep_default_na=False)
    if len(manifest)!=41362 or not manifest.sample_id.is_unique:
        raise ValueError('Public manifest count/identity differs')
    if manifest.groupby('duplicate_group').split.nunique().max()!=1:
        raise ValueError('Cross-split duplicate group')
    counts = {f'{lang}/{split}':int(n) for (lang,split),n in manifest.groupby(['language','split']).size().items()}
    if counts != json.loads((root/'data/split_counts.json').read_text()):
        raise ValueError('Split counts differ')
    index = json.loads((root/'results/prediction_index.json').read_text())
    if len(index)!=20 or len({r['file'] for r in index})!=20:
        raise ValueError('Prediction manifest must contain 20 unique files')
    expected = {f'results/predictions/{variant}_seed_{seed}_{lang}_test.npz'
                for variant in ('full_v5','no_cross_control') for seed in SEEDS for lang in ('zh','en')}
    if {r['file'] for r in index} != expected:
        raise ValueError('Unexpected/missing prediction files')
    for row in index:
        digest = hashlib.sha256((root/row['file']).read_bytes()).hexdigest()
        if digest != row['release_sha256']:
            raise ValueError('Released prediction file hash mismatch')
    historical = json.loads((root/'results/test_summary.json').read_text())
    scores = {}
    for variant in ('full_v5','no_cross_control'):
        rows = {k:[] for k in ('lb_mf1','zh_macro_f1','en_macro_f1','zh_accuracy','en_accuracy')}
        for seed in SEEDS:
            language_scores = []
            for lang in ('zh','en'):
                target = manifest[(manifest.language==lang)&(manifest.split=='test')].sort_values('split_row')
                with np.load(root/f'results/predictions/{variant}_seed_{seed}_{lang}_test.npz',allow_pickle=False) as data:
                    if set(data.files)-{'labels','logits','row_indices','probabilities'}:
                        raise ValueError('Non-allowlisted prediction content')
                    np.testing.assert_array_equal(data['labels'],target.label.to_numpy())
                    np.testing.assert_array_equal(data['row_indices'],target.split_row.to_numpy())
                    f1,acc = metrics(data['labels'],data['logits'])
                rows[f'{lang}_macro_f1'].append(f1)
                rows[f'{lang}_accuracy'].append(acc)
                language_scores.append(f1)
            rows['lb_mf1'].append(float(np.mean(language_scores)))
        for key,values in rows.items():
            for field,actual in [('values',values),('mean',np.mean(values)),('sd',np.std(values,ddof=1))]:
                np.testing.assert_allclose(actual,historical[variant][key][field],rtol=0,atol=1e-12)
        scores[variant]=np.array(rows['lb_mf1'])
    differences=scores['full_v5']-scores['no_cross_control']
    for field,actual in [('values',differences),('mean',differences.mean()),('sd',differences.std(ddof=1))]:
        np.testing.assert_allclose(actual,historical['paired_full_minus_no_cross'][field],rtol=0,atol=1e-12)
    if historical['independent_confirmatory_test'] is not False:
        raise ValueError('Test independence must not be relabelled')
    result={k:dict(mean=float(v.mean()),sample_sd=float(v.std(ddof=1))) for k,v in scores.items()}
    result.update(prediction_files=20,rows=int(len(manifest)),
        paired_difference_mean=float(differences.mean()),paired_difference_sd=float(differences.std(ddof=1)),
        full_wins=int((differences>1e-12).sum()),ties=int((abs(differences)<=1e-12).sum()),
        full_losses=int((differences < -1e-12).sum()),passed=True)
    return result

if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
