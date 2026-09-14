import unittest
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from contextlib import redirect_stdout
import io

import numpy as np

import torch

import full_model_v2 as v2
import run_full_model_v2 as runner
import train_fgclip2_emotion_enhanced_v8_3_ablation_suite_summary_plus as core


class FullV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        self.kwargs = dict(img_dim=8, txt_dim=8, patch_dim=8, token_dim=8,
                           proj_dim=8, hidden_dim=16, num_heads=2, dropout=0.)
        self.inputs = dict(
            img_global=torch.randn(3, 8), txt_global=torch.randn(3, 8),
            flat_patches=torch.randn(3, 6, 8), token_embeds=torch.randn(3, 5, 8),
            patch_mask=torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]).bool(),
            attention_mask=torch.tensor([[0, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]]).bool(),
            language_ids=torch.tensor([0, 1, 0]),
        )

    def model(self, use_cross):
        core.set_seed(42)
        return v2.ResidualFullClassifier(use_cross=use_cross, cross_dropout=0., **self.kwargs).eval()

    def test_all_shared_initial_parameters_are_identical(self):
        full, control = self.model(True), self.model(False)
        for key, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, full.state_dict()[key]), key)
        self.assertEqual(full.gate_input_features, 4)
        self.assertFalse(any("cross" in key for key in control.state_dict()))

    def test_zero_alpha_eval_matches_control_and_legacy_no_cross(self):
        full, control = self.model(True), self.model(False)
        legacy = core.EmotionGatedFusionClassifier(ablation_cfg=core.get_ablation_config("no_cross"), **self.kwargs).eval()
        legacy.load_state_dict(control.state_dict())
        with torch.no_grad():
            expected, _ = control(**self.inputs)
            actual, _ = full(**self.inputs)
            old, _ = legacy(**self.inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, old, rtol=0, atol=0)

    def test_cross_receives_gradients_after_zero_alpha_first_step(self):
        full = self.model(True)
        optimizer = torch.optim.SGD(full.parameters(), lr=.1)
        logits, _ = full(**self.inputs)
        logits.square().mean().backward()
        self.assertGreater(abs(full.cross_alpha.grad.item()), 0)
        self.assertEqual(full.cross_attention.attention.in_proj_weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        logits, _ = full(**self.inputs)
        logits.square().mean().backward()
        grad = full.cross_attention.attention.in_proj_weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_cross_changes_output_when_enabled(self):
        full = self.model(True)
        with torch.no_grad():
            zero, _ = full(**self.inputs)
            full.cross_alpha.fill_(.5)
            active, _ = full(**self.inputs)
        self.assertFalse(torch.allclose(zero, active))

    def test_masked_patches_and_tokens_cannot_change_output(self):
        full = self.model(True)
        altered = {key: value.clone() for key, value in self.inputs.items()}
        altered["flat_patches"][~altered["patch_mask"]] = 200
        altered["token_embeds"][~altered["attention_mask"]] = -200
        with torch.no_grad():
            full.cross_alpha.fill_(.5)
            logits, aux = full(**self.inputs, return_attention=True)
            other, _ = full(**altered)
        torch.testing.assert_close(logits, other)
        self.assertTrue((aux["token_weights"][~self.inputs["attention_mask"]] == 0).all())
        attention = aux["cross_attn"]
        masked = ~self.inputs["patch_mask"][:, None, None, :].expand_as(attention)
        self.assertTrue((attention[masked] == 0).all())

    def test_empty_content_returns_zero_local_and_cross_features(self):
        full = self.model(True)
        inputs = {**self.inputs, "attention_mask": torch.zeros_like(self.inputs["attention_mask"])}
        with torch.no_grad():
            full.cross_alpha.fill_(.5)
            logits, aux = full(**inputs)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(aux["text_local"].abs().sum().item(), 0)
        self.assertEqual(aux["cross_vec"].abs().sum().item(), 0)

    def test_empty_patch_mask_fails_explicitly(self):
        full = self.model(True)
        inputs = {**self.inputs, "patch_mask": torch.zeros_like(self.inputs["patch_mask"])}
        with self.assertRaisesRegex(ValueError, "valid patch"):
            full(**inputs)

    def test_eval_batch_composition_invariant(self):
        full = self.model(True)
        with torch.no_grad():
            full.cross_alpha.fill_(.5)
            whole, _ = full(**self.inputs)
            one, _ = full(**{key: value[:1] for key, value in self.inputs.items()})
        torch.testing.assert_close(whole[:1], one, atol=1e-6, rtol=1e-5)

    def test_content_offsets_exclude_prompt_special_and_padding(self):
        encoded = {"offset_mapping": [[(0, 3), (3, 8), (7, 10), (10, 12), (0, 0), (0, 0)]],
                   "special_tokens_mask": [[0, 0, 0, 0, 1, 1]],
                   "attention_mask": [[1, 1, 1, 1, 1, 0]]}
        self.assertEqual(v2.make_content_mask(encoded, [8]), [[0, 0, 1, 1, 0, 0]])

    def test_empty_content_offsets_mask_everything(self):
        encoded = {"offset_mapping": [[(0, 8), (0, 0)]],
                   "special_tokens_mask": [[0, 1]], "attention_mask": [[1, 1]]}
        self.assertEqual(v2.make_content_mask(encoded, [8]), [[0, 0]])

    def test_protocol_fingerprint_detects_configuration_changes(self):
        a = {"steps": 1200, "code": "abc", "dropout": .1}
        for key, value in (("steps", 100), ("code", "def"), ("dropout", .2)):
            self.assertNotEqual(runner.digest(a), runner.digest({**a, key: value}))
        self.assertEqual(runner.digest(a), runner.digest(dict(reversed(list(a.items())))))

    def test_summary_pairs_by_seed_not_row_order(self):
        rows = [dict(variant="full_v2", seed=123, best_val_lb_mf1=.7),
                dict(variant="no_cross_control", seed=42, best_val_lb_mf1=.8),
                dict(variant="full_v2", seed=42, best_val_lb_mf1=.9)]
        result = runner.summarize(rows)
        self.assertEqual(list(result["paired_full_v2_minus_control"]), ["42"])
        self.assertAlmostEqual(result["paired_full_v2_minus_control"]["42"], .1)
        self.assertIsNone(result["variants"]["no_cross_control"]["sample_std"])

    def test_resume_restores_all_random_streams(self):
        sampler = torch.Generator().manual_seed(12)
        batches = torch.Generator().manual_seed(34)
        state = runner.rng_state(sampler, batches)

        def draw():
            return (torch.rand(4), np.random.rand(4), random.random(),
                    torch.rand(4, generator=sampler), torch.rand(4, generator=batches))

        expected = draw()
        draw()
        runner.restore_rng(state, sampler, batches)
        actual = draw()
        for index in (0, 3, 4):
            torch.testing.assert_close(actual[index], expected[index], atol=0, rtol=0)
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])

    def test_zero_cross_dropout_preserves_shared_training_rng_at_zero_alpha(self):
        kwargs = {**self.kwargs, "dropout": .4}
        core.set_seed(42)
        full = v2.ResidualFullClassifier(use_cross=True, cross_dropout=0., **kwargs).train()
        core.set_seed(42)
        control = v2.ResidualFullClassifier(use_cross=False, **kwargs).train()
        # Reset after construction to isolate forward-pass stochastic streams.
        torch.manual_seed(1234)
        full_logits, _ = full(**self.inputs)
        full_rng = torch.get_rng_state()
        torch.manual_seed(1234)
        control_logits, _ = control(**self.inputs)
        control_rng = torch.get_rng_state()
        torch.testing.assert_close(full_logits, control_logits, atol=0, rtol=0)
        self.assertTrue(torch.equal(full_rng, control_rng))

    def test_nonzero_cross_dropout_consumes_extra_training_rng(self):
        kwargs = {**self.kwargs, "dropout": .4}
        core.set_seed(42)
        full = v2.ResidualFullClassifier(use_cross=True, cross_dropout=.1, **kwargs).train()
        core.set_seed(42)
        control = v2.ResidualFullClassifier(use_cross=False, **kwargs).train()
        torch.manual_seed(1234)
        full(**self.inputs)
        full_rng = torch.get_rng_state()
        torch.manual_seed(1234)
        control(**self.inputs)
        self.assertFalse(torch.equal(full_rng, torch.get_rng_state()))

    def test_json_rejects_nonfinite_diagnostics(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as temporary:
            path = Path(temporary) / "result.json"
            with self.assertRaises(ValueError):
                runner.write_json(path, {"grad_norm": float("inf")})
            self.assertFalse(path.exists())
            runner.write_json(path, {"finite_grad_norm_mean": None, "amp_skipped_steps": 1})
            self.assertIn('"amp_skipped_steps": 1', path.read_text())

    def test_interrupted_training_matches_uninterrupted_final_weights(self):
        variant = getattr(self, "resume_variant", "full_v2")
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as temporary, mock.patch.multiple(
            core, DEVICE="cpu", PROJ_DIM=8, HIDDEN_DIM=16, NUM_HEADS=2, BATCH_SIZE=4,
        ), redirect_stdout(io.StringIO()):
            root = Path(temporary)
            self.assertTrue(root.resolve().is_relative_to(runner.ROOT.resolve()))
            dims = dict(img_dim=8, txt_dim=8, patch_dim=8, token_dim=8)
            cache_root = root / "cache"
            generator = np.random.default_rng(77)
            masks = {}
            for split in runner.SPLITS:
                directory = cache_root / split
                arrays = runner.cached.create_arrays(directory, 8, dims)
                for name, array in arrays.items():
                    if np.issubdtype(array.dtype, np.floating):
                        array[:] = generator.normal(size=array.shape).astype(array.dtype)
                    elif name.endswith("mask"):
                        array[:] = 1
                    elif name == "labels":
                        array[:] = np.arange(8) % 3
                    else:
                        array[:] = int(split.startswith("en"))
                    array.flush()
                del arrays, array
                runner.write_json(directory / "metadata.json", {"rows": 8, "dims": dims})
                masks[split] = np.ones((8, core.MAX_TEXT_LEN), dtype=np.uint8)
            args = SimpleNamespace(cache_root=cache_root, limit_samples=8, cross_dropout=0.,
                                   steps_per_epoch=1, eval_batch_size=8, epochs=2,
                                   patience=4, grad_clip=1.)
            if variant in ("full_v4", "pooled_v4"):
                context_root = root / "context"
                for split in runner.SPLITS:
                    directory = context_root / split
                    directory.mkdir(parents=True)
                    np.save(directory / "tokens.npy", generator.normal(
                        size=(8, core.MAX_TEXT_LEN, dims["token_dim"])).astype(np.float16))
                args.contextual_cache_root = context_root
            protocol = {"fixture": "resume_integration", "smoke_only": True}
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"
            runner.run_one(args, {"dims": dims}, masks, protocol, uninterrupted, variant, 42)
            original_save = runner.save_torch

            def interrupt_after_first_epoch(path, value):
                original_save(path, value)
                if path.name == "last_state.ckpt" and value["epoch"] == 1:
                    raise InterruptedError("simulated disconnection")

            with mock.patch.object(runner, "save_torch", side_effect=interrupt_after_first_epoch):
                with self.assertRaises(InterruptedError):
                    runner.run_one(args, {"dims": dims}, masks, protocol, resumed, variant, 42)
            runner.run_one(args, {"dims": dims}, masks, protocol, resumed, variant, 42)
            suffix = Path(f"seed_42/{variant}/last_state.ckpt")
            a = torch.load(uninterrupted / suffix, map_location="cpu", weights_only=False)
            b = torch.load(resumed / suffix, map_location="cpu", weights_only=False)
            self.assertEqual(a["history"], b["history"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)

    def test_warm_start_uses_identical_shared_weights_and_rejects_wrong_seed(self):
        parent, full = self.model(False), self.model(True)
        protocol = {"core_settings": {"fixture": True}, "cache_metadata_sha256": "cache",
                    "content_mask_sha256": {"fixture": "mask"}, "limit_samples": 8}
        with torch.no_grad():
            parent.classifier[-1].weight.add_(.2)
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as temporary:
            path = Path(temporary) / "parent.ckpt"
            runner.save_torch(path, {"config": {"variant": "no_cross_control", "seed": 42,
                                               "protocol": protocol},
                                      "clf_state_dict": parent.state_dict()})
            runner.load_shared_start(full, path, 42, protocol)
            with torch.no_grad():
                expected, _ = parent(**self.inputs)
                actual, _ = full(**self.inputs)
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "paired"):
                runner.load_shared_start(full, path, 123, protocol)
            with self.assertRaisesRegex(ValueError, "content_mask"):
                runner.load_shared_start(full, path, 42, {**protocol, "content_mask_sha256": {}})

    def test_refinement_optimizer_keeps_base_learning_rate_paired(self):
        settings = SimpleNamespace(base_lr=1e-5, cross_lr=1e-4, alpha_lr=1e-3)
        full = runner.make_optimizer(self.model(True), settings)
        control = runner.make_optimizer(self.model(False), settings)
        self.assertEqual({g["name"]: g["lr"] for g in full.param_groups},
                         {"base": 1e-5, "cross": 1e-4, "alpha": 1e-3})
        self.assertEqual({g["name"]: g["lr"] for g in control.param_groups}, {"base": 1e-5})

    def test_contextual_request_cannot_silently_reuse_embedding_cache(self):
        with mock.patch.object(core, "TOKEN_FEATURE_SOURCE", "contextual"):
            with self.assertRaisesRegex(ValueError, "embedding"):
                runner.verify_cache(runner.ROOT / "not_used", runner.ROOT / "not_used")


if __name__ == "__main__":
    unittest.main()
