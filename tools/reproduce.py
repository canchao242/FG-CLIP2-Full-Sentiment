"""Print a fixed training stage; execute only with --run. No test-directed search."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'src'
def unique_protocol(folder):
    matches = list(folder.glob('*/protocol.json'))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one completed protocol under {folder}; found {len(matches)}')
    return matches[0].parent

def command(stage,seeds,parents=None,controls=None,device='cuda'):
    cfg = json.loads((ROOT/'configs/full_v5.json').read_text(encoding='utf-8'))
    out = SRC/'checkpoints/public_reproduction'
    common = ['--seeds',*map(str,seeds),'--steps-per-epoch',str(cfg['steps_per_epoch']),
              '--eval-batch-size',str(cfg['eval_batch_size']),'--device',device]
    if stage == 'parents':
        return [sys.executable,'-B',str(SRC/'run_full_model_v2.py'),'--variants','no_cross_control',
                '--epochs','20','--patience','4','--cross-dropout','0.1','--save-root',str(out/stage),*common]
    parents = parents or unique_protocol(out/'parents')
    if stage == 'controls':
        return [sys.executable,'-B',str(SRC/'run_full_model_v2.py'),'--variants','no_cross_control',
                '--epochs','5','--patience','4','--cross-dropout','0','--base-lr','0.00001',
                '--cross-lr','0.0001','--warm-start-root',str(parents),'--save-root',str(out/stage),*common]
    if stage != 'v5':
        raise ValueError('Unknown stage')
    controls = controls or unique_protocol(out/'controls')
    return [sys.executable,'-B',str(SRC/'train_full_model_v5.py'),'--variants','pooled_v5','full_v5',
            '--epochs','5','--freeze-epochs','3','--base-lr','0.00001','--branch-lr','0.0003',
            '--joint-branch-lr','0.0001','--branch-loss-weight','0.1','--warm-start-root',str(parents),
            '--reference-root',str(controls),'--save-root',str(out/stage),*common]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['parents','controls','v5'],required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,123,3407,2026,2027])
    parser.add_argument('--parents',type=Path)
    parser.add_argument('--controls',type=Path)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--run',action='store_true')
    group.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate seeds')
    cmd=command(args.stage,args.seeds,args.parents,args.controls,args.device)
    if args.preflight:
        cmd.append('--preflight')
    print(subprocess.list2cmdline(cmd),flush=True)
    if args.run or args.preflight:
        subprocess.run(cmd,cwd=SRC,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},check=True)
    else:
        print('Plan only. Add --preflight for read-only input checks, or --run to train.')

if __name__ == '__main__':
    main()
