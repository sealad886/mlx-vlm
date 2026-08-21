"""Native tests for integrated Qwen3.5 multimodal MTP contracts."""

from __future__ import annotations

import unittest

import mlx.core as mx
from mlx_vlm.models.qwen3_5.qwen3_5 import sanitize_key


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


if __name__ == "__main__":
    unittest.main()
