import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import full_model_v2 as v2
from full_model_v5 import StagedResidualFullClassifier
import train_full_model_v5 as runner

core, shared = runner.core, runner.shared


class FullV5Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        self.kwargs = dict(img_dim=8, txt_dim=8, patch_dim=8, token_dim=8,
                           proj_dim=8, hidden_dim=16, num_heads=2, dropout=.4)
        self.inputs = dict(img_global=torch.randn(3, 8), txt_global=torch.randn(3, 8),
                          flat_patches=torch.randn(3, 6, 8), token_embeds=torch.randn(3, 5, 16),
                          patch_mask=torch.tensor([[1,1,1,0,0,0],[1,1,1,1,0,0],[1,1,1,1,1,1]]).bool(),
                          attention_mask=torch.tensor([[0,1,1,0,0],[0,1,1,1,0],[0,1,1,1,1]]).bool(),
                          language_ids=torch.tensor([0,1,0]))
        self.args = SimpleNamespace(base_lr=1e-5, branch_lr=3e-4, joint_branch_lr=1e-4,
                                    freeze_epochs=1, branch_loss_weight=.1)

    def model(self, attention=True):
        core.set_seed(42)
        return StagedResidualFullClassifier(use_attention=attention, cross_dropout=0., **self.kwargs)

    def test_initial_logits_exactly_match_unmodified_no_cross(self):
        core.set_seed(42)
        control = v2.ResidualFullClassifier(use_cross=False, **self.kwargs).eval()
        static = {**self.inputs, "token_embeds": self.inputs["token_embeds"][..., :8]}
        expected, _ = control(**static)
        for attention in (True, False):
            model = self.model(attention).set_base_frozen(True).train()
            actual, _ = model(**self.inputs)
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            for name, parameter in control.state_dict().items():
                torch.testing.assert_close(parameter, model.state_dict()[name], atol=0, rtol=0)

    def test_frozen_base_disables_dropout_and_preserves_weights(self):
        model = self.model().set_base_frozen(True).train()
        before = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("cross_attention.")}
        optimizer = runner.make_optimizer(model, self.args)
        first, aux = model(**self.inputs)
        again, _ = model(**self.inputs)
        torch.testing.assert_close(first, again, atol=0, rtol=0)
        self.assertFalse(model.classifier.training)
        self.assertTrue(model.cross_attention.training)
        first.square().mean().backward()
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n in before))
        optimizer.step()
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        self.assertGreater(model.cross_attention.classifier[-1].weight.abs().sum().item(), 0)

    def test_joint_stage_restores_base_gradients_and_rates(self):
        model = self.model()
        optimizer = runner.make_optimizer(model, self.args)
        self.assertEqual(runner.configure_stage(model, optimizer, 1, self.args), "branch_only")
        self.assertFalse(model.img_global_proj[1].weight.requires_grad)
        self.assertEqual(runner.configure_stage(model, optimizer, 2, self.args), "joint")
        model.train()
        self.assertTrue(model.classifier.training)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model(**self.inputs)[0].square().mean().backward()
        self.assertIsNotNone(model.img_global_proj[1].weight.grad)
        self.assertEqual({g["name"]: g["lr"] for g in optimizer.param_groups}, {"base":1e-5,"cross":1e-4})

    def test_fp32_addition_keeps_sub_fp16_resolution_increment(self):
        base = torch.tensor([10.], dtype=torch.float16)
        delta = torch.tensor([.001], dtype=torch.float16, requires_grad=True)
        self.assertEqual((base + delta).item(), 10.)
        combined = StagedResidualFullClassifier.add_correction(base, delta)
        self.assertEqual(combined.dtype, torch.float32)
        self.assertGreater(combined.item(), 10.)
        combined.sum().backward()
        self.assertEqual(delta.grad.item(), 1.)

    def test_direct_branch_loss_finite_with_empty_text(self):
        model = self.model().set_base_frozen(True)
        inp = {**self.inputs, "attention_mask": torch.zeros_like(self.inputs["attention_mask"])}
        logits, aux = model(**inp)
        criterion = core.SmoothedFocalLoss(torch.ones(3), .05, 1.2)
        loss, final, branch = runner.training_loss(logits, aux, torch.tensor([0,1,2]), criterion, 1, .1)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(branch.item(), 0.)
        torch.testing.assert_close(loss, final)

    def test_masked_values_cannot_affect_active_branch(self):
        model = self.model().eval()
        with torch.no_grad():
            model.cross_attention.classifier[-1].weight.normal_(std=.1)
        changed = {k: v.clone() for k, v in self.inputs.items()}
        changed["token_embeds"][~changed["attention_mask"]] = 200
        changed["flat_patches"][~changed["patch_mask"]] = -200
        torch.testing.assert_close(model(**self.inputs)[0], model(**changed)[0])

    def test_shared_branch_initialization_matches_pooled(self):
        full, pooled = self.model(), self.model(False)
        for name, value in full.cross_attention.state_dict().items():
            if name.startswith(("context_projection.","context_score.","classifier.")):
                torch.testing.assert_close(value, pooled.cross_attention.state_dict()[name], atol=0, rtol=0)

    def test_summary_retains_all_pairs_including_losses(self):
        rows = [dict(seed=123, variant="full_v5",best_val_lb_mf1=.7),
                dict(seed=42,variant="full_v5",best_val_lb_mf1=.9)]
        refs = [dict(seed=42,score=.8),dict(seed=123,score=.85)]
        result = runner.summarize(rows, refs)["paired"]["full_v5_minus_no_cross_control_reused"]
        self.assertEqual(result["seeds"], [42,123])
        np.testing.assert_allclose(result["values"], [.1,-.15])
        self.assertLess(result["mean"], 0.)

    def test_resume_before_and_after_unfreeze_is_exact(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, mock.patch.multiple(
            core, DEVICE="cpu", PROJ_DIM=8, HIDDEN_DIM=16, NUM_HEADS=2, BATCH_SIZE=4,
            MAX_NUM_PATCHES=6, MAX_TEXT_LEN=5,
        ), redirect_stdout(io.StringIO()):
            root = Path(tmp)
            dims = dict(img_dim=8,txt_dim=8,patch_dim=8,token_dim=8)
            masks = {}
            generator = np.random.default_rng(77)
            for split in shared.SPLITS:
                directory = root / "cache" / split
                arrays = runner.cached.create_arrays(directory, 8, dims)
                for name, array in arrays.items():
                    if np.issubdtype(array.dtype, np.floating):
                        array[:] = generator.normal(size=array.shape)
                    elif name.endswith("mask"):
                        array[:] = 1
                    elif name == "labels":
                        array[:] = np.arange(8) % 3
                    else:
                        array[:] = int(split.startswith("en"))
                    array.flush()
                del arrays, array
                shared.write_json(directory / "metadata.json", dict(rows=8,dims=dims))
                masks[split] = np.ones((8,5), dtype=np.uint8)
                ctx = root / "context" / split
                ctx.mkdir(parents=True)
                np.save(ctx / "tokens.npy", generator.normal(size=(8,5,8)).astype(np.float16))
            protocol = dict(core_settings={},cache_metadata_sha256="fixture",content_mask_sha256={},limit_samples=8)
            parent = v2.ResidualFullClassifier(use_cross=False, **self.kwargs)
            source = root / "parent.ckpt"
            shared.save_torch(source, dict(config=dict(variant="no_cross_control",seed=42,protocol=protocol),clf_state_dict=parent.state_dict()))
            protocol = {**protocol,"warm_start_checkpoints":{"42":{"path":str(source),"sha256":runner.cached.file_sha256(source)}}}
            args = SimpleNamespace(**vars(self.args),cache_root=root/"cache",contextual_cache_root=root/"context",
                                   limit_samples=8,steps_per_epoch=1,eval_batch_size=8,epochs=3,grad_clip=1.)
            baseline = root / "uninterrupted"
            runner.run_one(args,dict(dims=dims),masks,protocol,baseline,"full_v5",42)
            for cut in (1,2):
                out = root / f"cut_{cut}"
                original_save = shared.save_torch
                def interrupt(path, value):
                    original_save(path,value)
                    if path.name == "last_state.ckpt" and value["epoch"] == cut:
                        raise InterruptedError("disconnect")
                with mock.patch.object(shared,"save_torch",side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        runner.run_one(args,dict(dims=dims),masks,protocol,out,"full_v5",42)
                runner.run_one(args,dict(dims=dims),masks,protocol,out,"full_v5",42)
                a = torch.load(baseline/"seed_42/full_v5/last_state.ckpt",weights_only=False)
                b = torch.load(out/"seed_42/full_v5/last_state.ckpt",weights_only=False)
                self.assertEqual(a["history"],b["history"])
                for name in a["model"]:
                    torch.testing.assert_close(a["model"][name],b["model"][name],atol=0,rtol=0)


ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    unittest.main()
