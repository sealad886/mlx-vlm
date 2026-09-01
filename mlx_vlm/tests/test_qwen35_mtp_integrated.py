"""Native tests for integrated Qwen3.5 multimodal MTP contracts."""

from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_vlm.models.qwen3_5.qwen3_5 import sanitize_key
from mlx_vlm.models.qwen3_5_moe.config import TextConfig
from mlx_vlm.models.qwen3_5_moe.language import LanguageModel, MTPMoeDecoderLayer


class TestMTPNamespace(unittest.TestCase):
    def test_raw_and_converted_namespaces_are_idempotent(self) -> None:
        self.assertEqual("language_model.mtp.fc.weight", sanitize_key("mtp.fc.weight"))
        self.assertEqual(
            "language_model.mtp.fc.weight",
            sanitize_key("language_model.mtp.fc.weight"),
        )
        self.assertEqual(
            "language_model.mtp.fc.weight",
            sanitize_key("model.language_model.mtp.fc.weight"),
        )

    def test_namespace_test_runs_with_native_mlx(self) -> None:
        value = mx.array([1.0], dtype=mx.float32)
        mx.eval(value)
        self.assertEqual(1.0, value.item())


class TestMTPMoeConstruction(unittest.TestCase):
    def test_moe_mtp_model_does_not_require_dense_intermediate_size(self) -> None:
        config = TextConfig(
            model_type="qwen3_5_moe_text",
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            num_experts=2,
            num_experts_per_tok=1,
            shared_expert_intermediate_size=8,
            moe_intermediate_size=8,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=1,
            max_position_embeddings=128,
            head_dim=8,
            full_attention_interval=1,
            mtp_num_hidden_layers=1,
        )

        self.assertFalse(hasattr(config, "intermediate_size"))
        model = LanguageModel(config)

        self.assertIsInstance(model.mtp.layers[0], MTPMoeDecoderLayer)
        self.assertFalse(model.mtp.layers[0].is_linear)


if __name__ == "__main__":
    unittest.main()
