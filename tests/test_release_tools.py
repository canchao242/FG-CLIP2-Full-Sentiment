"""Data-free safety and result checks for the publication wrappers."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

splitter=load('build_private_splits')
verifier=load('verify_results')
reproduce=load('reproduce')

class ReleaseTests(unittest.TestCase):
    def test_all_published_predictions(self):
        result=verifier.verify()
        self.assertEqual((result['full_wins'],result['ties'],result['full_losses']),(1,2,2))
        self.assertLess(result['paired_difference_mean'],0)

    def test_macro_f1_definition(self):
        labels=np.array([0,1,2,2])
        logits=np.eye(3)[[0,1,0,2]]
        self.assertAlmostEqual(verifier.metrics(labels,logits)[0],(2/3+1+2/3)/3)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            verifier.metrics(np.array([0]),np.array([[np.nan,0,1]]))

    def fixture(self,root):
        rows=[dict(image_path='a.jpg',text='same filename, second text',label=2),
              dict(image_path='a.jpg',text='first text',label=0)]
        (root/'a.jpg').touch()
        path=root/'local.csv'
        pd.DataFrame(rows).to_csv(path,index=False)
        manifest=pd.DataFrame([dict(sample_id=f'zh-{i}',image_file='a.jpg',language='zh',split='train',
            split_row=i,label=row['label'],duplicate_group=i,text_sha256=splitter.text_sha256(row['text']))
            for i,row in enumerate(reversed(rows))])
        return path,manifest

    def test_split_identity_and_exact_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,manifest=self.fixture(Path(tmp))
            result=splitter.reconstruct(manifest,{'zh':[path]})[('zh','train')]
            self.assertEqual(result.text.tolist(),['first text','same filename, second text'])
            self.assertEqual(result.label.tolist(),[0,2])

    def test_changed_text_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,manifest=self.fixture(Path(tmp))
            manifest.loc[0,'text_sha256']='0'*64
            with self.assertRaises(ValueError):
                splitter.reconstruct(manifest,{'zh':[path]})

    def test_cross_split_group_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,manifest=self.fixture(Path(tmp))
            manifest.loc[1,'split']='test'
            manifest.loc[1,'duplicate_group']=0
            with self.assertRaises(ValueError):
                splitter.reconstruct(manifest,{'zh':[path]})

    def test_fixed_commands_and_no_test_arguments(self):
        for stage in ('parents','controls','v5'):
            cmd=reproduce.command(stage,[42,123,3407,2026,2027],Path('parent'),Path('control'))
            self.assertNotIn('--test',cmd)
            self.assertIn('--steps-per-epoch',cmd)
        v5=reproduce.command('v5',[42],Path('parent'),Path('control'))
        self.assertEqual(v5[v5.index('--freeze-epochs')+1],'3')
        self.assertIn('full_v5',v5)
        self.assertIn('pooled_v5',v5)

if __name__ == '__main__':
    unittest.main()
