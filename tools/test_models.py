"""Run synthetic CPU model tests; no downloads or research-data loading."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
     'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
     'PYTHONPATH':os.pathsep.join([str(ROOT/'src'),str(ROOT/'tests')])}
if __name__ == '__main__':
    subprocess.run([sys.executable,'-B','-m','unittest','test_full_model_v2','test_full_model_v4','test_full_model_v5','-v'],
                   cwd=ROOT,env=env,check=True)
