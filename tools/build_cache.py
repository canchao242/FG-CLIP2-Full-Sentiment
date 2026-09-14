"""Build only frozen static features; do not launch any classifier experiments."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print('Plan only: reads src/data_submission_v1; builds/reuses src/checkpoints/fgclip2_submission_v1_feature_cache. Add --run explicitly.')
        return
    sys.path.insert(0,str(ROOT/'src'))
    import run_fgclip2_submission_cached_repeated as cached
    cached.build_cache(cached.CACHE_ROOT,cached.SPLIT_ROOT)

if __name__ == '__main__':
    main()
