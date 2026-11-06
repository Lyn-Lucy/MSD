import unittest

import torch

from eagle.model.visual_compression import (
    build_shifted_visual_padding_mask,
    compress_visual_pair,
    compress_visual_sequence,
)


class VisualCompressionTest(unittest.TestCase):
    def test_shifted_visual_padding_mask_preserves_absolute_text_position(self):
        attention_mask = build_shifted_visual_padding_mask(
            sequence_length=7,
            visual_start=1,
            original_visual_tokens=4,
            kept_visual_tokens=2,
        )
        self.assertEqual(attention_mask.tolist(), [[True, True, False, False, True, True, True]])

    def test_kmeans_matches_cdist_reference_and_keeps_streams_aligned(self):
        visual_inputs = torch.randn(97, 16)
        visual_hidden = visual_inputs + 100
        num_tokens = 7

        torch.manual_seed(1234)
        center_indices = torch.randperm(visual_inputs.shape[0])[:num_tokens]
        centers = visual_inputs[center_indices]
        assignments = torch.cdist(visual_inputs, centers, p=2).argmin(dim=-1)
        counts = torch.bincount(assignments, minlength=num_tokens).clamp_min(1).float().unsqueeze(1)
        expected = torch.zeros(num_tokens, visual_inputs.shape[1])
        expected.index_add_(0, assignments, visual_inputs)
        expected = expected / counts

        torch.manual_seed(1234)
        compressed_inputs, compressed_hidden = compress_visual_pair(
            visual_inputs, visual_hidden, num_tokens
        )

        self.assertTrue(torch.equal(compressed_inputs, expected))
        self.assertTrue(torch.allclose(compressed_hidden, compressed_inputs + 100, atol=2e-5))

    def test_training_and_inference_sequence_alignment(self):
        input_ids = torch.tensor([[10, 32000, 32000, 32000, 32000, 11, 12]])
        inputs_embeds = torch.arange(14, dtype=torch.float32).view(1, 7, 2)
        hidden_states = inputs_embeds + 100
        loss_mask = torch.tensor([[False, False, False, False, False, True, True]])

        output_ids, output_inputs, output_hidden, output_loss_mask, reduced_len = compress_visual_sequence(
            input_ids,
            inputs_embeds,
            hidden_states,
            image_token_id=32000,
            num_tokens=2,
            loss_mask=loss_mask,
        )

        self.assertEqual(output_ids.tolist(), [[10, 32000, 32000, 11, 12]])
        self.assertEqual(output_inputs.shape, output_hidden.shape)
        self.assertEqual(output_inputs.shape, (1, 5, 2))
        self.assertEqual(output_loss_mask.tolist(), [[False, False, False, True, True]])
        self.assertEqual(reduced_len, 2)

    def test_padded_compression_restores_visual_span_and_text_positions(self):
        input_ids = torch.tensor([[10, 32000, 32000, 32000, 32000, 11, 12]])
        inputs_embeds = torch.arange(14, dtype=torch.float32).view(1, 7, 2)
        hidden_states = inputs_embeds + 100
        loss_mask = torch.tensor([[False, False, False, False, False, True, True]])

        output_ids, output_inputs, output_hidden, output_loss_mask, reduced_len = compress_visual_sequence(
            input_ids,
            inputs_embeds,
            hidden_states,
            image_token_id=32000,
            num_tokens=2,
            loss_mask=loss_mask,
            pad_to_original=True,
        )

        self.assertEqual(output_ids.tolist(), [[10, 32000, 32000, 0, 0, 11, 12]])
        self.assertEqual(output_inputs.shape, inputs_embeds.shape)
        self.assertEqual(output_hidden.shape, hidden_states.shape)
        self.assertTrue(torch.equal(output_inputs[:, 3:5], torch.zeros(1, 2, 2)))
        self.assertTrue(torch.equal(output_hidden[:, 3:5], torch.zeros(1, 2, 2)))
        self.assertTrue(torch.equal(output_inputs[:, 5:], inputs_embeds[:, 5:]))
        self.assertTrue(torch.equal(output_hidden[:, 5:], hidden_states[:, 5:]))
        self.assertEqual(output_loss_mask.tolist(), [[False, False, False, False, False, True, True]])
        self.assertEqual(reduced_len, 2)

    def test_padded_compression_rejects_noncontiguous_visual_tokens(self):
        input_ids = torch.tensor([[10, 32000, 11, 32000, 12]])
        features = torch.ones(1, 5, 3)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            compress_visual_sequence(
                input_ids,
                features,
                features,
                32000,
                1,
                pad_to_original=True,
            )

    def test_sequence_compression_preserves_non_visual_input(self):
        input_ids = torch.tensor([[10, 11]])
        features = torch.ones(1, 2, 3)
        output = compress_visual_sequence(input_ids, features, features, 32000, 1)
        self.assertTrue(torch.equal(output[0], input_ids))
        self.assertEqual(output[4], 0)

    def test_inference_defaults_to_kmeans_after_reset(self):
        from unittest.mock import Mock
        from eagle.model.utils import initialize_tree, temp_cache

        temp_cache.reset()
        self.addCleanup(temp_cache.reset)
        self.assertEqual(temp_cache.visual_tokens, 32)
        input_ids = torch.tensor([[10] + [32000] * 40 + [11]])
        features = torch.randn(1, 42, 4)
        model = Mock(return_value=(None, torch.zeros(1, 42, 16), features + 100))
        model.ea_layer.topK_genrate.return_value = (None, None, None, None)
        result = initialize_tree(input_ids, model, None, None, inputs_embeds=features)
        self.assertEqual((result[-1] == 32000).sum().item(), 32)
        self.assertEqual(temp_cache.reduced_len, 8)
        self.assertEqual(model.ea_layer.topK_genrate.call_args.args[0].shape[1], 34)

    def test_invalid_output_length_is_rejected(self):
        features = torch.ones(4, 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            compress_visual_pair(features, features, 0)


if __name__ == "__main__":
    unittest.main()
