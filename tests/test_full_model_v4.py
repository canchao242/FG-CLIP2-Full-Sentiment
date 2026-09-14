import unittest
import torch

import full_model_v2 as v2
import full_model_v4 as v4
import test_full_model_v2 as fixtures


class FullV4Tests(unittest.TestCase):
    def setUp(self):
        fixtures.FullV2Tests.setUp(self)
        self.static_inputs = self.inputs
        self.inputs = {**self.inputs, "token_embeds": torch.cat([
            self.inputs["token_embeds"], torch.randn_like(self.inputs["token_embeds"])], dim=-1)}

    def model(self, full=True):
        torch.manual_seed(42)
        return v4.ContextualResidualFullClassifier(use_attention=full, **self.kwargs).eval()

    def test_zero_correction_and_shared_rng_equal_no_cross(self):
        torch.manual_seed(42)
        control = v2.ResidualFullClassifier(use_cross=False, **self.kwargs).eval()
        state = torch.get_rng_state()
        for full in (False, True):
            model = self.model(full)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            torch.testing.assert_close(model(**self.inputs)[0], control(**self.static_inputs)[0], atol=0, rtol=0)
            for key, value in control.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)

    def test_contextual_common_modules_match_between_arms(self):
        full, pooled = self.model(True), self.model(False)
        for key, value in full.cross_attention.state_dict().items():
            if key.startswith(("context_projection.", "context_score.", "classifier.")):
                torch.testing.assert_close(value, pooled.cross_attention.state_dict()[key], atol=0, rtol=0)

    def test_valid_context_changes_active_correction(self):
        full = self.model()
        other = {**self.inputs, "token_embeds": self.inputs["token_embeds"].clone()}
        other["token_embeds"][..., 8:] += torch.randn_like(other["token_embeds"][..., 8:])
        with torch.no_grad():
            full.cross_attention.classifier[-1].weight.normal_(std=.1)
            a, _ = full(**self.inputs)
            b, _ = full(**other)
        self.assertFalse(torch.allclose(a, b))

    def test_masked_context_and_patches_do_not_change_logits(self):
        full = self.model()
        other = {k: v.clone() for k, v in self.inputs.items()}
        other["token_embeds"][~other["attention_mask"]] = 400
        other["flat_patches"][~other["patch_mask"]] = -400
        with torch.no_grad():
            full.cross_attention.classifier[-1].weight.normal_(std=.1)
            a, aux = full(**self.inputs, return_attention=True)
            b, _ = full(**other)
        torch.testing.assert_close(a, b)
        self.assertTrue((aux["contextual_token_weights"][~other["attention_mask"]] == 0).all())

    def test_empty_text_has_zero_correction_even_with_output_bias(self):
        for full in (False, True):
            model = self.model(full)
            with torch.no_grad():
                model.cross_attention.classifier[-1].bias.fill_(1.)
            logits, aux = model(**{**self.inputs, "attention_mask": torch.zeros_like(self.inputs["attention_mask"])})
            self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(aux["correction_logits"].abs().sum().item(), 0)
            self.assertEqual(aux["contextual_token_weights"].abs().sum().item(), 0)

    def test_context_and_cross_receive_gradients_after_output_update(self):
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        model(**self.inputs)[0].square().mean().backward()
        self.assertGreater(model.cross_attention.classifier[-1].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        model(**self.inputs)[0].square().mean().backward()
        for parameter in (model.cross_attention.context_projection[1].weight,
                          model.cross_attention.context_score.weight,
                          model.cross_attention.interaction.attention.in_proj_weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_missing_context_and_invalid_images_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "separately verified"):
            self.model()(**self.static_inputs)
        for full in (False, True):
            with self.assertRaisesRegex(ValueError, "valid patch"):
                self.model(full)(**{**self.inputs, "patch_mask": torch.zeros_like(self.inputs["patch_mask"])})


    def test_interrupted_training_matches_uninterrupted(self):
        import test_full_model_v2 as fixtures
        self.resume_variant = "full_v4"
        fixtures.FullV2Tests.test_interrupted_training_matches_uninterrupted_final_weights(self)


if __name__ == "__main__":
    unittest.main()
