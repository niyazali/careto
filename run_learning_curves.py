"""Exploratory whole-study learning curves; specified before these fits."""
import argparse
import json
from pathlib import Path
import numpy as np
import run_validation as v

def main(output):
    B = Path(__file__).resolve().parent
    O = output
    O.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((B / 'analysis_config.json').read_text())
    rows = [r for r in json.loads((B / 'COHORT.json').read_text())['rows'] if r['cohort'] == 'core']
    spec = {'models': ['RandomForest', 'SVR', 'training_median'], 'fractions': [0.25, 0.5, 0.75, 1.0], 'subset_repeats': 3, 'seed': 20261010, 'outer_scheme': 'original_study', 'selection': 'Nested target-independent subsets of whole training studies within each repeat and outer fold', 'uncertainty': 'Observed training-subset variation; not confidence intervals or prospective experimental validation', 'timing': 'After main analysis was specified; before learning curve fits', 'inputs': {n: v.hash_file(B / n) for n in ['COHORT.json', 'analysis_config.json']}}
    v.save_json(O / 'SPECIFICATION.json', spec)
    out = {'specification': spec, 'folds': [], 'predictions': [], 'status': 'running'}
    for fraction in spec['fractions']:
        for repeat in range(1 if fraction == 1 else spec['subset_repeats']):
            for fold, tr, te in v.splits(rows, 'original_study', cfg):
                pool = [rows[i] for i in tr]
                test = [rows[i] for i in te]
                groups = sorted({r['study_group'] for r in pool})
                order = np.random.default_rng(spec['seed'] + repeat * 100 + int(fold.split('_')[-1])).permutation(groups)
                chosen = set(order[:max(2, round(len(groups) * fraction))])
                train = [r for r in pool if r['study_group'] in chosen]
                assert not chosen & {r['study_group'] for r in test}
                log = {'fraction': fraction, 'repeat': repeat, 'fold': fold, 'training_studies': len(chosen), 'training_rows': len(train), 'train_row_ids': [r['row_id'] for r in train], 'test_row_ids': [r['row_id'] for r in test], 'tuning': {}}
                for model in spec['models']:
                    fitted, tuning = v.fit_tuned(train, model, cfg)
                    pred = fitted.predict(v.matrix(test, cfg))
                    log['tuning'][model] = tuning
                    out['predictions'].extend([{'row_id': r['row_id'], 'study_group': r['study_group'], 'target': r['target'], 'prediction': float(p), 'fraction': fraction, 'repeat': repeat, 'model': model, 'fold': fold} for r, p in zip(test, pred)])
                out['folds'].append(log)
                v.save_json(O / 'CHECKPOINT.json', out)
            print(f'fraction {fraction} repeat {repeat} complete', flush=True)
    summary = []
    for fraction in spec['fractions']:
        for model in spec['models']:
            maes = []
            for repeat in range(1 if fraction == 1 else spec['subset_repeats']):
                p = [r for r in out['predictions'] if r['fraction'] == fraction and r['repeat'] == repeat and (r['model'] == model)]
                assert len(p) == len({r['row_id'] for r in p}) == len(rows)
                maes.append(v.metrics([r['target'] for r in p], [r['prediction'] for r in p], [r['study_group'] for r in p])['study_MAE'])
            folds = [r for r in out['folds'] if r['fraction'] == fraction]
            summary.append({'fraction': fraction, 'model': model, 'mean_training_studies': float(np.mean([r['training_studies'] for r in folds])), 'mean_study_MAE': float(np.mean(maes)), 'repeat_MAEs': maes})
    out['summary'] = summary
    out['status'] = 'complete'
    v.save_json(O / 'RESULTS.json', out)
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results' / 'learning_curves')
    main(parser.parse_args().output)
