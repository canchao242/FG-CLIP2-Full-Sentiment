"""Reconstruct exact split membership/order from legally obtained local CSVs."""
from __future__ import annotations
import argparse
import hashlib
from collections import defaultdict, deque
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def text_sha256(s):
    return hashlib.sha256(str(s).encode('utf-8')).hexdigest()
def filename(s):
    return str(s).replace('\\','/').rsplit('/',1)[-1]

def reconstruct(manifest, sources):
    required = {'sample_id','image_file','language','split','split_row','label','duplicate_group','text_sha256'}
    if set(manifest.columns) != required or not manifest.sample_id.is_unique:
        raise ValueError('Invalid or duplicate public manifest fields')
    if set(manifest.language)-{'zh','en'} or set(manifest.split)-{'train','val','test'}:
        raise ValueError('Unexpected language or split')
    if manifest.groupby('duplicate_group').split.nunique().max() != 1:
        raise ValueError('Duplicate-connected group crosses splits')
    records = defaultdict(deque)
    for lang, paths in sources.items():
        for path in paths:
            frame = pd.read_csv(path,dtype=str,keep_default_na=False)
            if not {'image_path','text','label'} <= set(frame.columns):
                raise ValueError('Local CSV requires image_path,text,label columns')
            for row in frame.to_dict('records'):
                label = int(row['label'])
                if label not in (0,1,2):
                    raise ValueError('Labels must be 0=negative, 1=neutral, 2=positive')
                image = Path(row['image_path']).expanduser()
                if not image.is_absolute():
                    image = Path(path).resolve().parent/image
                if not image.is_file():
                    raise ValueError('A local image file is missing; do not silently drop rows')
                key = (lang,filename(row['image_path']),label,text_sha256(row['text']))
                records[key].append(dict(image_path=str(image.resolve()),text=row['text'],label=label))
    result = {}
    for (lang,split),group in manifest.groupby(['language','split']):
        group = group.sort_values('split_row')
        if group.split_row.astype(int).tolist() != list(range(len(group))):
            raise ValueError('Split row indices are not a complete ordered sequence')
        rows = []
        for row in group.to_dict('records'):
            key = (lang,row['image_file'],int(row['label']),row['text_sha256'])
            if not records[key]:
                raise ValueError('Local sample identity/content/count differs from the fixed manifest')
            rows.append(records[key].popleft())
        result[(lang,split)] = pd.DataFrame(rows,columns=['image_path','text','label'])
    if any(records.values()):
        raise ValueError('Extra local samples remain; supply the study usable subset, not a different split')
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zh-csv',type=Path,action='append',required=True)
    parser.add_argument('--en-csv',type=Path,action='append',required=True)
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/split_manifest_public.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'src/data_submission_v1')
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest,keep_default_na=False)
    frames = reconstruct(manifest,{'zh':args.zh_csv,'en':args.en_csv})
    targets = [args.output/lang/(split+'.csv') for lang,split in frames]
    if any(p.exists() for p in targets):
        raise FileExistsError('Refusing to overwrite existing private splits; use an empty output directory')
    for (lang,split),frame in frames.items():
        target = args.output/lang/(split+'.csv')
        target.parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(target,index=False,encoding='utf-8',lineterminator='\n')
    print(f'Validated and wrote {sum(len(f) for f in frames.values())} private rows. Raw data stay local.')

if __name__ == '__main__':
    main()
