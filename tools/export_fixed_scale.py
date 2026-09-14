"""Create fixed-scale manifests for a NEW local run; never select using test."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print(f'Plan only: verify local checkpoints under {args.run_root}; export Full=0.5 and pooled=0.25 into fixed_scale_manifests. Add --run.')
        return
    sys.path.insert(0,str(ROOT/'src'))
    import train_full_model_v5 as runner
    from full_model_v5_scaled import load_scaled_checkpoint
    run=args.run_root.resolve()
    protocol=json.loads((run/'protocol.json').read_text())
    cfg=json.loads((ROOT/'configs/full_v5.json').read_text())
    if set(protocol['variants']) != {'full_v5','pooled_v5'} or protocol['limit_samples']:
        raise ValueError('Requires completed non-smoke paired Full-v5/pooled-v5 protocol')
    cache=runner.ROOT/'checkpoints/fgclip2_submission_v1_feature_cache'
    meta=json.loads((cache/'cache_ready.json').read_text())
    if runner.shared.digest(meta)!=protocol['cache_metadata_sha256']:
        raise ValueError('Feature cache identity mismatch')
    output=run/'fixed_scale_manifests'
    candidates=[]
    for seed in protocol['seeds']:
        for variant in protocol['variants']:
            directory=run/f'seed_{seed}'/variant
            result=json.loads((directory/'run_result.json').read_text())
            configuration=json.loads((directory/'config.json').read_text())
            if configuration['protocol']!=protocol or result['fingerprint']!=runner.shared.digest(configuration):
                raise ValueError('Completed result/protocol mismatch')
            checkpoint=directory/'best_head.ckpt'
            alpha=cfg['full_inference_scale' if variant=='full_v5' else 'pooled_inference_scale']
            manifest=dict(format='fgclip2_v5_fixed_scale_v1',variant=variant,seed=seed,
                checkpoint_path=str(checkpoint),checkpoint_sha256=runner.cached.file_sha256(checkpoint),
                residual_scale=alpha,feature_dims=meta['dims'],cache_metadata_sha256=protocol['cache_metadata_sha256'],
                selected_epoch=result['selected']['epoch'],zero_initial_branch=result['selected']['epoch']==0,
                selected_on_validation=True,test_evaluated=False,new_training_performed=False,
                selection_rule='Checkpoint chosen by this run validation at alpha=1; inference alpha fixed from published protocol, not retuned here.')
            path=output/f'{variant}_seed_{seed}.json'
            if path.exists():
                raise FileExistsError('Refusing to overwrite an existing manifest')
            candidates.append((path,manifest))
    output.mkdir(exist_ok=True)
    for path,manifest in candidates:
        runner.shared.write_json(path,manifest)
        # Only trusted locally produced checkpoints; torch pickle is not safe for untrusted weights.
        model,_=load_scaled_checkpoint(path,'cpu')
        del model
    print(f'Created and CPU-loaded {len(candidates)} fixed-scale manifests; no test inputs read.')

if __name__ == '__main__':
    main()
